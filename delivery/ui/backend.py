"""UI Backend Helpers, Configuration Hashing, and Map Rendering (§7.7, M10.1–M10.3).

All UI actions route through the exact same validated backend modules as the CLI
(delivery.overrides, delivery.search, delivery.sim, delivery.game, delivery.llm).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from delivery.agents.hybrid import HybridAgent
from delivery.heuristics.base import ZeroHeuristic
from delivery.heuristics.baselines import ManhattanNearestStopHeuristic, MaxRelaxedDistanceHeuristic
from delivery.heuristics.own import CombinedMSTHeuristic
from delivery.instance import Instance, get_canonical_instance
from delivery.overrides import apply_overrides
from delivery.search.astar import astar_search
from delivery.search.bfs import breadth_first_search
from delivery.search.core import Limits, SearchResult, SearchStatus
from delivery.search.dls import depth_limited_search
from delivery.search.greedy import greedy_best_first_search
from delivery.search.ids import iterative_deepening_search
from delivery.search.problem import ACTION_NAMES, DeliveryProblem
from delivery.search.ucs import uniform_cost_search
from delivery.sim.env import DeliveryEnv
from delivery.stamp import STAMP
from delivery.twists.base import TwistStack
from delivery.twists.registry import build_twist_stack
from delivery.viz.render import NORM, TERRAIN_CMAP
from delivery.world.grid import GridWorld


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Computes a canonical SHA-256 hash of UI sidebar configuration for cache keying."""
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_configured_instance(
    slip_probability: float,
    start: Tuple[int, int],
    stops: List[Tuple[int, int]],
    twists: List[str],
    rival_depth: int,
    seed: int,
) -> Instance:
    """Builds and validates a configured Instance through delivery.overrides."""
    base = get_canonical_instance()
    overrides = {
        "slip_probability": float(slip_probability),
        "start": tuple(map(int, start)),
        "delivery_stops": [tuple(map(int, s)) for s in stops],
        "twists": list(twists),
        "adversary_search_depth": int(rival_depth),
        "seed": int(seed),
    }
    return apply_overrides(base, overrides)


def run_ui_plan(
    instance: Instance,
    algo: str = "astar",
    heuristic: str = "own",
    depth_limit: int = 100,
    max_expanded: int = 200_000,
    max_time_seconds: float = 15.0,
) -> Dict[str, Any]:
    """Runs the selected Part B planner on the configured Instance."""
    world = GridWorld(instance)
    twists = build_twist_stack(instance, world)
    problem = DeliveryProblem(instance, world=world, twist_stack=twists)
    limits = Limits(max_expanded=max_expanded, max_time_seconds=max_time_seconds)

    algo_l = algo.lower()
    h_l = heuristic.lower()
    h_obj = None
    if algo_l in ("astar", "greedy"):
        if h_l == "zero":
            h_obj = ZeroHeuristic()
        elif h_l == "b1":
            h_obj = ManhattanNearestStopHeuristic()
        elif h_l == "b2":
            h_obj = MaxRelaxedDistanceHeuristic(problem)
        else:
            h_obj = CombinedMSTHeuristic(problem)

    if algo_l == "bfs":
        res = breadth_first_search(problem, limits=limits)
    elif algo_l == "ucs":
        res = uniform_cost_search(problem, limits=limits)
    elif algo_l == "dls":
        res = depth_limited_search(problem, depth_limit=depth_limit, limits=limits)
    elif algo_l == "ids":
        res = iterative_deepening_search(problem, max_depth=depth_limit, limits=limits)
    elif algo_l == "greedy":
        res = greedy_best_first_search(problem, heuristic=h_obj, limits=limits)
    elif algo_l == "astar":
        res = astar_search(problem, heuristic=h_obj, limits=limits)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return {
        "status": res.status.value,
        "success": res.status == SearchStatus.SUCCESS,
        "cost": float(res.cost) if np.isfinite(res.cost) else None,
        "length": len(res.plan) if res.plan is not None else 0,
        "plan": list(res.plan) if res.plan is not None else [],
        "trajectory": list(res.trajectory) if res.trajectory is not None else [],
        "expanded": res.metrics.expanded_nodes,
        "generated": res.metrics.generated_nodes,
        "peak_stored": res.metrics.peak_stored,
        "duration_s": res.metrics.duration_seconds,
        "failure_reason": res.failure_reason,
    }


def render_map_figure(
    instance: Instance,
    agent_pos: Optional[Tuple[int, int]] = None,
    rival_pos: Optional[Tuple[int, int]] = None,
    trajectory: Optional[List[Tuple[int, int]]] = None,
    delivered_mask: int = 0,
    title: str = "Delivery City Map",
) -> plt.Figure:
    """Renders a stamped Matplotlib grid figure for display in Streamlit."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(instance.grid, cmap=TERRAIN_CMAP, norm=NORM, origin="upper")

    sr, sc = instance.start
    ax.plot(sc, sr, marker="s", color="#e91e63", markersize=10, label="Depot")

    for idx, (dr, dc) in enumerate(instance.delivery_stops):
        done = bool(delivered_mask & (1 << idx))
        color = "#9e9e9e" if done else "#ffd700"
        ax.plot(dc, dr, marker="*", color=color, markersize=13, markeredgecolor="black")
        ax.text(dc + 0.3, dr - 0.3, f"S{idx}", color="white", weight="bold", fontsize=8)

    if trajectory and len(trajectory) > 1:
        ax.plot(
            [p[1] for p in trajectory],
            [p[0] for p in trajectory],
            color="#00e676",
            linewidth=2.2,
            label="Route",
        )

    if agent_pos is not None:
        ax.plot(agent_pos[1], agent_pos[0], marker="o", color="#00bcd4", markersize=10, markeredgecolor="black", label="Agent")

    if rival_pos is not None:
        ax.plot(rival_pos[1], rival_pos[0], marker="X", color="#d50000", markersize=11, markeredgecolor="white", label="Chaser")

    ax.set_title(title, fontsize=11, weight="bold")
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.5, 0.01, STAMP, ha="center", va="bottom", fontsize=7.5, color="#555555")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


class UISimSession:
    """Interactive Step/Play/Reset execution session for DeliveryEnv + HybridAgent."""

    def __init__(
        self,
        instance: Instance,
        enable_rival: bool = False,
        window_radius: int = 4,
    ):
        self.instance = instance
        self.world = GridWorld(instance)
        self.twists = build_twist_stack(instance, self.world)
        self.enable_rival = enable_rival
        self.window_radius = int(window_radius)
        self.env = DeliveryEnv(
            world=self.world,
            twists=self.twists,
            slip_p=instance.slip_probability,
            seed=instance.seed,
            enable_rival=enable_rival,
            rival_depth=min(4, int(instance.adversary_search_depth)),
        )
        self.agent = HybridAgent(
            instance,
            world=self.world,
            twists=self.twists,
            window_radius=self.window_radius,
            tactical_depth=min(3, int(instance.adversary_search_depth)),
        )
        self.trajectory: List[Tuple[int, int]] = [self.env.agent_pos]
        self.event_log: List[Dict[str, Any]] = []

    def step_once(self) -> Dict[str, Any]:
        """Executes a single Sense-Reason-Act step (+ rival move if enabled)."""
        if self.env.is_done() or self.env.captured:
            return {"done": True}

        percept = self.env.get_percept()
        action, rationale = self.agent.choose_action(percept)
        _, step_cost, ev = self.env.step(action)
        self.trajectory.append(self.env.agent_pos)

        rival_act = None
        if self.enable_rival and not self.env.is_done() and not self.env.captured:
            rival_act = self.env.step_rival()

        entry = {
            "step": self.env.step_count,
            "action": ACTION_NAMES.get(action, str(action)),
            "pos": list(self.env.agent_pos),
            "slipped": bool(ev.get("slipped", False)),
            "cost": float(step_cost),
            "total_cost": float(self.env.total_cost),
            "rival_pos": list(self.env.rival_pos) if self.env.rival_pos else None,
            "rival_action": ACTION_NAMES.get(rival_act, "-") if rival_act is not None else "-",
            "rationale": rationale,
            "replans": self.agent.replan_counts.total_replans(),
            "captured": self.env.captured,
            "done": self.env.is_done(),
        }
        self.event_log.append(entry)
        return entry

    def play_all(self, max_steps: int = 120) -> List[Dict[str, Any]]:
        """Runs until done, captured, or max_steps."""
        while not self.env.is_done() and not self.env.captured and self.env.step_count < max_steps:
            self.step_once()
        return self.event_log


def run_ui_rival_game(
    instance: Instance,
    window_radius: int = 4,
    use_tuned_weights: bool = True,
    max_steps: int = 150,
) -> Dict[str, Any]:
    """Plays a full adversarial game against the assigned Chaser rival using HybridAgent."""
    import json
    from pathlib import Path
    from delivery.game.evaluation import EvalWeights, make_default_evaluator
    from delivery.game.window import WindowGame, compute_canonical_baseline
    from delivery.sim.runner import run as run_sim

    world = GridWorld(instance)
    twists = build_twist_stack(instance, world)
    c_star, l_star, _ = compute_canonical_baseline(instance, world=world, twists=twists)

    weights = EvalWeights()
    tuned_path = Path(__file__).resolve().parent.parent.parent / "config" / "tuned_weights.json"
    if use_tuned_weights and tuned_path.exists():
        with open(tuned_path, "r", encoding="utf-8") as f:
            w = json.load(f).get("weights", {})
        weights = EvalWeights(
            w_delivered=float(w.get("w_delivered", 350.0)),
            w_remaining_route=float(w.get("w_remaining_route", -300.0)),
            w_rival_dist=float(w.get("w_rival_dist", 180.0)),
            w_mobility=float(w.get("w_mobility", 70.0)),
            w_cost_margin=float(w.get("w_cost_margin", 50.0)),
        )

    env = DeliveryEnv(
        world=world,
        twists=twists,
        slip_p=instance.slip_probability,
        seed=instance.seed,
        enable_rival=True,
        rival_depth=min(4, int(instance.adversary_search_depth)),
    )
    game = WindowGame(
        instance,
        world=world,
        twists=twists,
        window_radius=window_radius,
        c_star=c_star,
        l_star=l_star,
    )
    evaluator = make_default_evaluator(game, weights=weights)
    agent = HybridAgent(
        instance,
        world=world,
        twists=twists,
        window_radius=window_radius,
        tactical_depth=min(3, int(instance.adversary_search_depth)),
        eval_fn=evaluator,
        c_star=c_star,
        l_star=l_star,
    )
    res = run_sim(agent, env, max_steps=max_steps)

    if res.success:
        outcome = "WIN"
    elif res.failure_reason == "CAPTURED_BY_RIVAL":
        outcome = "LOSS (CAPTURED)"
    else:
        outcome = f"DRAW ({res.failure_reason or 'STEP_CAP'})"

    return {
        "outcome": outcome,
        "success": res.success,
        "total_cost": float(res.total_cost),
        "total_steps": int(res.total_steps),
        "trajectory": list(res.trajectory),
        "rival_pos": env.rival_pos,
        "delivered_mask": env.delivered_mask,
        "replans": agent.replan_counts.to_dict(),
        "tactical_moves": agent.tactical_moves,
        "strategic_moves": agent.strategic_moves,
    }


def get_provider_status() -> Dict[str, Any]:
    """Returns sanitized LLM provider status without exposing API keys."""
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=False)
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    has_openai = bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    provider = "groq" if (has_groq and not has_openai) else ("openai-compat" if has_openai else "unconfigured")
    model = os.environ.get("LLM_MODEL") or ("openai/gpt-oss-120b" if provider == "groq" else "gpt-4o-mini")
    return {
        "provider": provider,
        "model": model,
        "key_configured": has_groq or has_openai,
    }


def run_ui_dispatcher(
    instance: Instance,
    request_text: str,
    llm_mode: str = "replay",
    use_validation: bool = True,
    client_override: Any = None,
) -> Dict[str, Any]:
    """Executes the validated ReAct LLM dispatcher for the UI Dispatcher tab."""
    from delivery.llm.client import MockClient, OpenAICompatibleClient, ReplayClient
    from delivery.llm.react import run_react_dispatcher
    from delivery.llm.tools import DispatcherToolbox
    from delivery.llm.validation import ArgumentValidator

    world = GridWorld(instance)
    twists = build_twist_stack(instance, world)
    toolbox = DispatcherToolbox(instance=instance, world=world, twists=twists)
    validator = ArgumentValidator(instance) if use_validation else None

    if client_override is not None:
        client = client_override
    elif llm_mode == "live":
        client = OpenAICompatibleClient()
    elif llm_mode == "mock":
        client = MockClient(
            responses=[
                {
                    "content": "Reason: Computing optimal route via A* with own heuristic.",
                    "tool_calls": [{"id": "ui_1", "function": {"name": "plan_route", "arguments": '{"algorithm": "astar", "heuristic": "own"}'}}],
                },
                'Reason: Route planned.\nFINAL_ANSWER: {"status": "SUCCESS", "message": "Mock dispatcher route complete."}',
            ]
        )
    else:
        client = ReplayClient()

    res = run_react_dispatcher(
        request_text=request_text,
        client=client,
        toolbox=toolbox,
        request_id="UI-DISPATCH",
        max_steps=6,
        validator=validator,
        bypass_cache=(llm_mode == "live"),
    )
    return res.to_dict()

