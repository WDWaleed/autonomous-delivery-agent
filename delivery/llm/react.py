"""ReAct Dispatcher Loop with Safe Trace Logging (§7.6, M8.3).

Implements:
- System prompt generator embedding instance facts (stops, terrain costs, twists, tools)
- Structured Sense -> Reason (concise summary) -> Action (tool call) -> Observation cycle
- Robust parser supporting both OpenAI tool_calls and structured text Reason/Action blocks
- Optional ArgumentValidator hook (wired in M8.5) with bounded self-repair retries
- Stamped, secret-sanitized JSONL and text trace logging to results/traces/
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from delivery.instance import Instance
from delivery.llm.client import LLMClient, LLMResponse, sanitize_secrets
from delivery.llm.tools import TOOL_SCHEMAS, DispatcherToolbox
from delivery.stamp import STAMP

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRACES_DIR = PROJECT_ROOT / "results" / "traces"
PROMPT_VERSION = "v1.0"


def build_system_prompt(instance: Instance) -> str:
    """Builds grounded system prompt with instance facts, twists, and tool rules."""
    stops_str = ", ".join(f"Stop {i} at {pos}" for i, pos in enumerate(instance.delivery_stops))
    twists_str = ", ".join(instance.twists) if instance.twists else "none"
    return (
        f"You are the Autonomous Delivery Dispatcher for city instance '{instance.id}' "
        f"({instance.n}x{instance.n} grid, start depot {instance.start}).\n"
        f"Delivery Stops: {stops_str}.\n"
        f"Terrain Entry Costs: {instance.terrain_cost} (obstacles = -1).\n"
        f"Active Twists: {twists_str}.\n"
        "Rules:\n"
        "1. Follow a structured Reason -> Action -> Observation loop.\n"
        "2. Provide a concise 1-sentence Reason summary (no hidden chain-of-thought).\n"
        "3. Use the provided tools (plan_route, simulate, inspect_twists) to verify feasibility and cost.\n"
        "4. If a request is ambiguous or infeasible/conflicting, state so clearly and refuse or ask for clarification.\n"
        "5. End with FINAL_ANSWER: <JSON summary with status, plan, cost, reason>."
    )


@dataclass
class ReActStep:
    """Single Sense-Reason-Action-Observation step in a ReAct trace."""

    step: int
    reason_summary: str
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    observation: Optional[Dict[str, Any]]
    validation_error: Optional[str] = None
    malformed_parse: bool = False
    latency_seconds: float = 0.0
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "reason_summary": sanitize_secrets(self.reason_summary),
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "observation": self.observation,
            "validation_error": self.validation_error,
            "malformed_parse": self.malformed_parse,
            "latency_seconds": self.latency_seconds,
            "cached": self.cached,
        }


@dataclass
class ReActResult:
    """Outcome of a ReAct dispatcher session."""

    request_id: str
    status: str  # COMPLETED, STEP_LIMIT, ERROR
    final_answer: str
    final_payload: Dict[str, Any] = field(default_factory=dict)
    steps: List[ReActStep] = field(default_factory=list)
    tool_calls_count: int = 0
    validation_interventions: int = 0
    malformed_count: int = 0
    total_latency_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stamp": STAMP,
            "request_id": self.request_id,
            "status": self.status,
            "final_answer": sanitize_secrets(self.final_answer),
            "final_payload": self.final_payload,
            "tool_calls_count": self.tool_calls_count,
            "validation_interventions": self.validation_interventions,
            "malformed_count": self.malformed_count,
            "total_latency_seconds": self.total_latency_seconds,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "steps": [s.to_dict() for s in self.steps],
        }


def parse_llm_action(resp: LLMResponse) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[str], bool]:
    """Parses LLMResponse into (reason_summary, tool_name, tool_args, final_answer, is_malformed).

    Supports native OpenAI tool_calls as well as text-based Reason/Action/Action Input blocks.
    """
    content = resp.content or ""
    reason_match = re.search(r"Reason:\s*(.+?)(?=\n(?:Action|FINAL_ANSWER):|$)", content, re.DOTALL | re.IGNORECASE)
    reason_summary = reason_match.group(1).strip() if reason_match else content.strip()[:200]

    # 1. Native tool_calls
    if resp.tool_calls:
        tc = resp.tool_calls[0]
        fn = tc.get("function", {})
        t_name = fn.get("name")
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, dict):
            return reason_summary, t_name, raw_args, None, False
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            if not isinstance(parsed, dict):
                return reason_summary, t_name, None, None, True
            return reason_summary, t_name, parsed, None, False
        except json.JSONDecodeError:
            return reason_summary, t_name, None, None, True

    # 2. FINAL_ANSWER in text
    final_match = re.search(r"FINAL_ANSWER:\s*(.+)$", content, re.DOTALL | re.IGNORECASE)
    if final_match:
        return reason_summary, None, None, final_match.group(1).strip(), False

    # 3. Text Action / Action Input
    act_match = re.search(r"Action:\s*([A-Za-z0-9_]+)", content, re.IGNORECASE)
    if act_match:
        t_name = act_match.group(1).strip()
        inp_match = re.search(r"Action Input:\s*(\{.*\})", content, re.DOTALL | re.IGNORECASE)
        if not inp_match:
            return reason_summary, t_name, None, None, True
        try:
            parsed = json.loads(inp_match.group(1).strip())
            if not isinstance(parsed, dict):
                return reason_summary, t_name, None, None, True
            return reason_summary, t_name, parsed, None, False
        except json.JSONDecodeError:
            return reason_summary, t_name, None, None, True

    return reason_summary, None, None, None, True


def _extract_json_payload(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def save_trace_files(result: ReActResult, trace_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Saves stamped, sanitized JSONL and rendered text traces."""
    out_dir = trace_dir or TRACES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", result.request_id)
    jsonl_path = out_dir / f"{safe_id}.jsonl"
    txt_path = out_dir / f"{safe_id}.txt"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    lines = [
        f"=== {STAMP} ===",
        f"Request ID: {result.request_id} | Status: {result.status} | Provider/Model: {result.provider}/{result.model}",
        f"Tool Calls: {result.tool_calls_count} | Validation Interventions: {result.validation_interventions}",
        "-" * 72,
    ]
    for st in result.steps:
        lines.append(f"[Step {st.step}] Reason: {sanitize_secrets(st.reason_summary)}")
        if st.tool_name:
            lines.append(f"  Action: {st.tool_name}({json.dumps(st.tool_args)})")
        if st.validation_error:
            lines.append(f"  Validation Error: {st.validation_error}")
        if st.observation is not None:
            obs_str = sanitize_secrets(json.dumps(st.observation))
            if len(obs_str) > 300:
                obs_str = obs_str[:297] + "..."
            lines.append(f"  Observation: {obs_str}")
    lines.append("-" * 72)
    lines.append(f"Final Answer: {sanitize_secrets(result.final_answer)}")
    lines.append(f"=== {STAMP} ===")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return jsonl_path, txt_path


def run_react_dispatcher(
    request_text: str,
    client: LLMClient,
    toolbox: DispatcherToolbox,
    request_id: str = "REQ-001",
    max_steps: int = 6,
    validator: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None,
    max_repair_attempts: int = 2,
    bypass_cache: bool = False,
    save_trace: bool = False,
    trace_dir: Optional[Path] = None,
) -> ReActResult:
    """Executes the bounded ReAct loop with optional argument validation and self-repair."""
    t_start = time.perf_counter()
    sys_prompt = build_system_prompt(toolbox.instance)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": request_text},
    ]

    steps_log: List[ReActStep] = []
    tool_calls_count = 0
    interventions = 0
    malformed_count = 0
    consecutive_repairs = 0

    for step_idx in range(1, max_steps + 1):
        resp = client.generate(
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.0,
            max_tokens=2048,
            bypass_cache=bypass_cache,
        )
        reason_summary, tool_name, tool_args, final_answer, is_malformed = parse_llm_action(resp)

        if final_answer is not None:
            payload = _extract_json_payload(final_answer)
            steps_log.append(
                ReActStep(
                    step=step_idx,
                    reason_summary=reason_summary,
                    tool_name=None,
                    tool_args=None,
                    observation=None,
                    latency_seconds=resp.latency_seconds,
                    cached=resp.cached,
                )
            )
            res = ReActResult(
                request_id=request_id,
                status="COMPLETED",
                final_answer=final_answer,
                final_payload=payload,
                steps=steps_log,
                tool_calls_count=tool_calls_count,
                validation_interventions=interventions,
                malformed_count=malformed_count,
                total_latency_seconds=time.perf_counter() - t_start,
                provider=resp.provider,
                model=resp.model,
            )
            if save_trace:
                save_trace_files(res, trace_dir=trace_dir)
            return res

        if is_malformed or tool_name is None or tool_args is None:
            malformed_count += 1
            err_msg = "Malformed or incomplete tool call. Please emit valid JSON tool arguments or FINAL_ANSWER."
            steps_log.append(
                ReActStep(
                    step=step_idx,
                    reason_summary=reason_summary,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation={"error": err_msg},
                    malformed_parse=True,
                    latency_seconds=resp.latency_seconds,
                    cached=resp.cached,
                )
            )
            messages.append({"role": "assistant", "content": resp.content or ""})
            messages.append({"role": "user", "content": f"Observation: {err_msg}"})
            continue

        # Optional validation layer (M8.5)
        if validator is not None:
            val_err = validator(tool_name, tool_args)
            if val_err is not None:
                interventions += 1
                consecutive_repairs += 1
                obs = {"status": "VALIDATION_ERROR", "error": val_err, "repair_attempt": consecutive_repairs}
                steps_log.append(
                    ReActStep(
                        step=step_idx,
                        reason_summary=reason_summary,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=obs,
                        validation_error=val_err,
                        latency_seconds=resp.latency_seconds,
                        cached=resp.cached,
                    )
                )
                messages.append({"role": "assistant", "content": resp.content or f"Action: {tool_name}"})
                messages.append({"role": "user", "content": f"Observation (VALIDATION_ERROR): {val_err}"})
                if consecutive_repairs > max_repair_attempts:
                    break
                continue

        consecutive_repairs = 0
        tool_calls_count += 1
        obs = toolbox.execute(tool_name, tool_args)
        steps_log.append(
            ReActStep(
                step=step_idx,
                reason_summary=reason_summary,
                tool_name=tool_name,
                tool_args=tool_args,
                observation=obs,
                latency_seconds=resp.latency_seconds,
                cached=resp.cached,
            )
        )
        messages.append({"role": "assistant", "content": resp.content or f"Action: {tool_name}"})
        messages.append({"role": "user", "content": f"Observation: {json.dumps(obs)}"})

    res = ReActResult(
        request_id=request_id,
        status="STEP_LIMIT",
        final_answer="Terminated at step budget without FINAL_ANSWER.",
        steps=steps_log,
        tool_calls_count=tool_calls_count,
        validation_interventions=interventions,
        malformed_count=malformed_count,
        total_latency_seconds=time.perf_counter() - t_start,
        provider=client.provider,
        model=client.model,
    )
    if save_trace:
        save_trace_files(res, trace_dir=trace_dir)
    return res
