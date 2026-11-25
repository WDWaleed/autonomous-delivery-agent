"""Provider-agnostic LLM Client with Record-Replay Cache (§7.6, M8.1).

Supports:
- LLMClient protocol
- MockClient: deterministic scripted responses for offline unit tests
- ReplayClient: reads sanitized results/llm_cache.jsonl keyed by
  (provider, model, messages, tools, temperature, max_tokens)
- OpenAICompatibleClient: connects to hosted OpenAI-compatible APIs or
  local servers (Ollama/vLLM/LM Studio), reading secrets strictly from
  environment variables (.env) without ever logging, displaying, or caching keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / "results" / "llm_cache.jsonl"


def sanitize_secrets(text: str, extra_secrets: Sequence[str] = ()) -> str:
    """Redacts API keys or bearer tokens before logging or writing traces/cache."""
    if not text:
        return text
    cleaned = re.sub(r"sk-[A-Za-z0-9_\-]{10,}", "[REDACTED_KEY]", text)
    cleaned = re.sub(r"AIza[A-Za-z0-9_\-]{10,}", "[REDACTED_KEY]", cleaned)
    cleaned = re.sub(r"gsk_[A-Za-z0-9_\-]{10,}", "[REDACTED_KEY]", cleaned)
    cleaned = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]{10,}", "Bearer [REDACTED_TOKEN]", cleaned)
    for sec in extra_secrets:
        if sec and len(sec) >= 6:
            cleaned = cleaned.replace(sec, "[REDACTED_KEY]")
    return cleaned


def compute_cache_key(
    provider: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Computes deterministic SHA-256 cache key across provider, model, prompt, schema, and settings."""
    payload = {
        "provider": provider,
        "model": model,
        "messages": messages,
        "tools": tools or [],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class LLMResponse:
    """Standardized LLM completion record."""

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = "mock"
    model: str = "mock-model"
    model_version: str = "1.0"
    cached: bool = False
    latency_seconds: float = 0.0
    cache_key: str = ""
    raw_usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "cached": self.cached,
            "latency_seconds": self.latency_seconds,
            "cache_key": self.cache_key,
            "raw_usage": self.raw_usage,
        }


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for provider-agnostic LLM clients."""

    provider: str
    model: str

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        ...


class MockClient:
    """Deterministic scripted LLM client for offline unit testing."""

    def __init__(
        self,
        responses: Optional[Sequence[Any]] = None,
        provider: str = "mock",
        model: str = "mock-dispatcher-v1",
    ):
        self.provider = provider
        self.model = model
        self.responses = list(responses or [])
        self.call_idx = 0
        self.history: List[Dict[str, Any]] = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        key = compute_cache_key(self.provider, self.model, messages, tools, temperature, max_tokens)
        self.history.append({"messages": messages, "tools": tools, "cache_key": key})

        if self.call_idx < len(self.responses):
            item = self.responses[self.call_idx]
            self.call_idx += 1
        elif self.responses:
            item = self.responses[-1]
        else:
            item = "FINAL_ANSWER: {\"status\": \"NO_MOCK_RESPONSE\"}"

        if isinstance(item, LLMResponse):
            return LLMResponse(
                content=item.content,
                tool_calls=list(item.tool_calls),
                provider=self.provider,
                model=self.model,
                model_version=item.model_version,
                cached=False,
                latency_seconds=0.001,
                cache_key=key,
                raw_usage=dict(item.raw_usage),
            )
        if isinstance(item, dict):
            return LLMResponse(
                content=str(item.get("content", "")),
                tool_calls=list(item.get("tool_calls", [])),
                provider=self.provider,
                model=self.model,
                cached=False,
                latency_seconds=0.001,
                cache_key=key,
            )
        return LLMResponse(
            content=str(item),
            provider=self.provider,
            model=self.model,
            cached=False,
            latency_seconds=0.001,
            cache_key=key,
        )


class ReplayClient:
    """Offline replay client reading sanitized completions from results/llm_cache.jsonl."""

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        provider: str = "openai-compat",
        model: str = "gpt-4o-mini",
    ):
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.provider = provider
        self.model = model
        self._records: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        self._records.clear()
        if not self.cache_path.exists():
            return
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    k = rec.get("cache_key")
                    if k:
                        self._records[k] = rec
                except json.JSONDecodeError:
                    continue

    def save_record(self, resp: LLMResponse) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "cache_key": resp.cache_key,
            "provider": resp.provider,
            "model": resp.model,
            "model_version": resp.model_version,
            "content": sanitize_secrets(resp.content),
            "tool_calls": resp.tool_calls,
            "raw_usage": resp.raw_usage,
        }
        self._records[resp.cache_key] = entry
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        key = compute_cache_key(self.provider, self.model, messages, tools, temperature, max_tokens)
        if not bypass_cache and key in self._records:
            rec = self._records[key]
            return LLMResponse(
                content=rec.get("content", ""),
                tool_calls=list(rec.get("tool_calls", [])),
                provider=rec.get("provider", self.provider),
                model=rec.get("model", self.model),
                model_version=rec.get("model_version", "cached"),
                cached=True,
                latency_seconds=0.0,
                cache_key=key,
                raw_usage=dict(rec.get("raw_usage", {})),
            )
        raise KeyError(f"Cache miss in ReplayClient for key {key[:12]}... (provider={self.provider}, model={self.model})")


class OpenAICompatibleClient:
    """Client for hosted OpenAI-compatible APIs or local servers (Ollama, vLLM, LM Studio).

    Reads API key from environment variables (.env) and never prints, logs, or caches secrets.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "openai-compat",
        cache_path: Optional[Path] = None,
        timeout_seconds: float = 60.0,
    ):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        self.provider = provider
        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        self._api_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.timeout_seconds = float(timeout_seconds)
        self.replay_cache = ReplayClient(
            cache_path=cache_path,
            provider=self.provider,
            model=self.model,
        )

    def __repr__(self) -> str:
        return f"OpenAICompatibleClient(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        bypass_cache: bool = False,
    ) -> LLMResponse:
        key = compute_cache_key(self.provider, self.model, messages, tools, temperature, max_tokens)

        if not bypass_cache:
            try:
                return self.replay_cache.generate(
                    messages, tools=tools, temperature=temperature, max_tokens=max_tokens, bypass_cache=False
                )
            except KeyError:
                pass

        if not self._api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise RuntimeError(
                "Missing LLM_API_KEY / OPENAI_API_KEY in environment/.env for hosted endpoint."
            )

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if tools:
            body["tools"] = tools

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp_obj:
                raw_json = json.loads(resp_obj.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = sanitize_secrets(e.read().decode("utf-8", errors="replace"), [self._api_key])
            raise RuntimeError(f"LLM HTTP {e.code}: {err_body}") from None
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {sanitize_secrets(str(e), [self._api_key])}") from None
        elapsed = time.perf_counter() - t0

        choice = raw_json.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = sanitize_secrets(str(msg.get("content") or ""), [self._api_key])
        tool_calls = list(msg.get("tool_calls") or [])
        model_ver = str(raw_json.get("model", self.model))
        usage = dict(raw_json.get("usage", {}))

        llm_resp = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            provider=self.provider,
            model=self.model,
            model_version=model_ver,
            cached=False,
            latency_seconds=elapsed,
            cache_key=key,
            raw_usage=usage,
        )

        if not bypass_cache:
            self.replay_cache.save_record(llm_resp)

        return llm_resp
