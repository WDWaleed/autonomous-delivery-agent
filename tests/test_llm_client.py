"""Tests for Provider-Agnostic LLM Client and Record-Replay Cache (§7.6, M8.1)."""

import json
from pathlib import Path

import pytest

from delivery.llm.client import (
    LLMResponse,
    MockClient,
    OpenAICompatibleClient,
    ReplayClient,
    compute_cache_key,
    sanitize_secrets,
)


def test_mock_client_scripted_responses():
    client = MockClient(responses=["first reply", "second reply"])
    r1 = client.generate([{"role": "user", "content": "hi"}])
    r2 = client.generate([{"role": "user", "content": "next"}])
    assert r1.content == "first reply"
    assert r2.content == "second reply"
    assert r1.cached is False


def test_cache_key_changes_with_settings_and_tools():
    msgs = [{"role": "user", "content": "route to stop 0"}]
    k1 = compute_cache_key("openai", "gpt-4o-mini", msgs, tools=None, temperature=0.0)
    k2 = compute_cache_key("openai", "gpt-4o-mini", msgs, tools=[{"name": "plan_route"}], temperature=0.0)
    k3 = compute_cache_key("openai", "gpt-4o-mini", msgs, tools=None, temperature=0.5)
    assert k1 != k2
    assert k1 != k3


def test_replay_client_roundtrip_and_secret_redaction(tmp_path: Path):
    cache_file = tmp_path / "llm_cache.jsonl"
    msgs = [{"role": "user", "content": "plan delivery"}]
    key = compute_cache_key("openai", "gpt-4o-mini", msgs, temperature=0.0)

    replay = ReplayClient(cache_path=cache_file, provider="openai", model="gpt-4o-mini")
    resp = LLMResponse(
        content="Using key sk-secret1234567890abcdef for plan",
        provider="openai",
        model="gpt-4o-mini",
        model_version="2026-01",
        cache_key=key,
    )
    replay.save_record(resp)

    # Reload from disk
    replay2 = ReplayClient(cache_path=cache_file, provider="openai", model="gpt-4o-mini")
    loaded = replay2.generate(msgs, temperature=0.0)
    assert loaded.cached is True
    assert "sk-secret" not in loaded.content
    assert "[REDACTED_KEY]" in loaded.content

    # Bypass cache raises KeyError in ReplayClient
    with pytest.raises(KeyError):
        replay2.generate(msgs, temperature=0.0, bypass_cache=True)


def test_openai_client_repr_hides_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-verysecretkey123456")
    client = OpenAICompatibleClient()
    assert "sk-verysecret" not in repr(client)
    assert sanitize_secrets("Bearer tok1234567890abc") == "Bearer [REDACTED_TOKEN]"
