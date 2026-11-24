"""Sense–Reason–Act execution runner (§6.1, §7.3, M3.2).

Governed by decisions D1, D6, D7, and D11.
Executes the closed loop: SENSE (percept) -> REASON (decision rationale) -> ACT (step),
logging structured step traces for benchmarks and LLM audit trails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from delivery.agents.base import Agent
from delivery.sim.env import ACTION_NAMES, DeliveryEnv


@dataclass
class StepTrace:
    """Detailed record of a single Sense-Reason-Act cycle."""

    step: int
    from_pos: Tuple[int, int]
    intended_action: int
    action_name: str
    decision_rationale: str
    slipped: bool
    sampled_action: Optional[int]
    realised_destination: Tuple[int, int]
    cost: float
    total_cost: float
    new_deliveries: List[int]
    delivered_mask: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "from_pos": list(self.from_pos),
            "intended_action": self.intended_action,
            "action_name": self.action_name,
            "decision_rationale": self.decision_rationale,
            "slipped": self.slipped,
            "sampled_action": self.sampled_action,
            "realised_destination": list(self.realised_destination),
            "cost": self.cost,
            "total_cost": self.total_cost,
            "new_deliveries": self.new_deliveries,
            "delivered_mask": self.delivered_mask,
        }


@dataclass
class RunnerResult:
    """Execution outcome summary for an episode."""

    success: bool
    total_cost: float
    total_steps: int
    trajectory: List[Tuple[int, int]]
    trace: List[StepTrace]
    delivered_mask: int
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_cost": self.total_cost,
            "total_steps": self.total_steps,
            "trajectory": [list(p) for p in self.trajectory],
            "trace": [t.to_dict() for t in self.trace],
            "delivered_mask": self.delivered_mask,
            "failure_reason": self.failure_reason,
        }


def run(
    agent: Agent,
    env: DeliveryEnv,
    max_steps: int = 200,
) -> RunnerResult:
    """Executes the Sense-Reason-Act loop until all stops are delivered or max_steps is reached."""
    agent.reset()
    trajectory: List[Tuple[int, int]] = [env.agent_pos]
    trace: List[StepTrace] = []

    # Check if already completed at start
    if env.is_done():
        return RunnerResult(
            success=True,
            total_cost=0.0,
            total_steps=0,
            trajectory=trajectory,
            trace=trace,
            delivered_mask=env.delivered_mask,
            failure_reason=None,
        )

    for step_num in range(1, max_steps + 1):
        # 1. SENSE
        percept = env.get_percept()
        from_pos = env.agent_pos

        # 2. REASON
        action, rationale = agent.choose_action(percept)

        # 3. ACT
        next_percept, cost, events = env.step(action)
        dest_pos = env.agent_pos
        trajectory.append(dest_pos)

        step_record = StepTrace(
            step=step_num,
            from_pos=from_pos,
            intended_action=action,
            action_name=ACTION_NAMES.get(action, f"ACTION_{action}"),
            decision_rationale=rationale,
            slipped=events.get("slipped", False),
            sampled_action=events.get("sampled_action"),
            realised_destination=dest_pos,
            cost=cost,
            total_cost=env.total_cost,
            new_deliveries=events.get("new_deliveries", []),
            delivered_mask=env.delivered_mask,
        )
        trace.append(step_record)

        if env.captured:
            return RunnerResult(
                success=False,
                total_cost=env.total_cost,
                total_steps=step_num,
                trajectory=trajectory,
                trace=trace,
                delivered_mask=env.delivered_mask,
                failure_reason="CAPTURED_BY_RIVAL",
            )

        if env.is_done():
            return RunnerResult(
                success=True,
                total_cost=env.total_cost,
                total_steps=step_num,
                trajectory=trajectory,
                trace=trace,
                delivered_mask=env.delivered_mask,
                failure_reason=None,
            )

        if env.rival_pos is not None:
            env.step_rival()
            if env.captured:
                return RunnerResult(
                    success=False,
                    total_cost=env.total_cost,
                    total_steps=step_num,
                    trajectory=trajectory,
                    trace=trace,
                    delivered_mask=env.delivered_mask,
                    failure_reason="CAPTURED_BY_RIVAL",
                )

    # Reached step cap
    failure_reason = "STEP_CAP_EXCEEDED"
    # Check if stuck in a cycle (oscillation between 2-3 positions in the last 20 steps)
    if len(trajectory) >= 20:
        tail = trajectory[-10:]
        unique_tail = set(tail)
        if len(unique_tail) <= 3:
            failure_reason = "LOCAL_MINIMUM_OSCILLATION"

    return RunnerResult(
        success=False,
        total_cost=env.total_cost,
        total_steps=max_steps,
        trajectory=trajectory,
        trace=trace,
        delivered_mask=env.delivered_mask,
        failure_reason=failure_reason,
    )
