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
