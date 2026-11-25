"""Command-line interface for the delivery agent system.

Usage:
    python -m delivery.cli instance [--cms ID ...] [--json]
    python -m delivery.cli plan --algo {bfs,ucs,dls,ids,greedy,astar} --heuristic {zero,b1,b2,own} [--set k=v ...]
    python -m delivery.cli certify [--algo {astar,ucs}] [--json]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from delivery.certificate import FeasibilityCertificate, FeasibilityStatus, certify_feasibility
from delivery.heuristics.base import ZeroHeuristic
from delivery.heuristics.baselines import ManhattanNearestStopHeuristic, MaxRelaxedDistanceHeuristic
from delivery.instance import get_canonical_instance, load_instance, save_instance
from delivery.overrides import parse_cli_overrides, apply_overrides
from delivery.search.astar import astar_search
from delivery.search.bfs import breadth_first_search
from delivery.search.core import Limits, SearchResult, SearchStatus
from delivery.search.dls import depth_limited_search
from delivery.search.greedy import greedy_best_first_search
from delivery.search.ids import iterative_deepening_search
from delivery.search.problem import ACTION_NAMES, DeliveryProblem
from delivery.search.ucs import uniform_cost_search
from delivery.stamp import banner, print_stamped_table
from delivery.twists.registry import build_twist_stack
from delivery.viz.render import render_grid_world
from delivery.world.grid import GridWorld


def cmd_instance(args):
    print(banner("CANONICAL PROBLEM INSTANCE"))

    if args.cms:
        inst = load_instance(args.cms)
        if args.save:
            save_instance(inst)
    else:
        inst = get_canonical_instance()

    if args.json:
        print(json.dumps(inst.to_dict(), indent=2))
        return

    print(f"Instance ID:            {inst.id}")
    print(f"Grid Size:              {inst.n} x {inst.n} ({inst.n * inst.n} cells)")
    print(f"Terrain Costs:          Road={inst.terrain_cost['road']}, Grass={inst.terrain_cost['grass']}, Mud={inst.terrain_cost['mud']}, Ford={inst.terrain_cost['ford']}")
    print(f"Obstacle Density:       {inst.obstacle_density * 100:.1f}%")
    print(f"Depot / Start:          {inst.start}")
    print(f"Delivery Stops ({len(inst.delivery_stops)}):     {inst.delivery_stops}")
    print(f"Adversary / Rival:      {inst.adversary} (search depth: {inst.adversary_search_depth} plies)")
    print(f"Simulator Slip Rate:    {inst.slip_probability * 100:.1f}%")
    print(f"Assigned Twists:        {', '.join(inst.twists)}")
    print(f"Configured Members:     {', '.join(inst.cms_ids)}")


def cmd_plan(args):
    print(banner(f"OFFLINE SEARCH PLANNER: {args.algo.upper()}"))

    # Load canonical instance and apply CLI overrides if any
    inst = get_canonical_instance()
    if args.set:
        overrides = parse_cli_overrides(args.set)
        inst = apply_overrides(inst, overrides)
        print(f"Applied Overrides:      {overrides}")

    world = GridWorld(inst)
    twists = build_twist_stack(inst.twists, world=world)
    problem = DeliveryProblem(inst, world=world, twist_stack=twists)

    limits = Limits(
        max_expanded=args.max_expanded,
        max_time_seconds=args.max_time,
        max_stored_records=args.max_stored,
    )

    algo = args.algo.lower()
    h_name = args.heuristic.lower()

    # Heuristic selection
    heuristic = None
    if algo in ["astar", "greedy"]:
        if h_name == "zero":
            heuristic = ZeroHeuristic()
        elif h_name == "b1":
            heuristic = ManhattanNearestStopHeuristic()
        elif h_name == "b2":
            heuristic = MaxRelaxedDistanceHeuristic(problem)
        elif h_name == "own":
            from delivery.heuristics.own import CombinedMSTHeuristic
            heuristic = CombinedMSTHeuristic(problem)
        else:
            raise ValueError(f"Unknown heuristic: {args.heuristic}")

    # Search execution
    if algo == "bfs":
        result = breadth_first_search(problem, limits=limits)
    elif algo == "ucs":
        result = uniform_cost_search(problem, limits=limits)
    elif algo == "dls":
        result = depth_limited_search(problem, depth_limit=args.depth_limit, limits=limits)
    elif algo == "ids":
        result = iterative_deepening_search(problem, max_depth=args.depth_limit, limits=limits)
    elif algo == "greedy":
        result = greedy_best_first_search(problem, heuristic=heuristic, limits=limits)
    elif algo == "astar":
        result = astar_search(problem, heuristic=heuristic, limits=limits)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Print stamped summary table
    summary_headers = ["Metric", "Value"]
    summary_rows = [
        ["Algorithm", algo.upper()],
        ["Heuristic", heuristic.name if heuristic else "N/A"],
        ["Search Status", result.status.value],
        ["Success", str(result.status == SearchStatus.SUCCESS)],
        ["Plan Length (Actions)", str(len(result.plan) if result.plan else 0)],
        ["Path Cost", f"{result.cost:.2f}" if result.cost < float("inf") else "inf"],
        ["Nodes Expanded", f"{result.metrics.expanded_nodes:,}"],
        ["Nodes Generated", f"{result.metrics.generated_nodes:,}"],
        ["Peak Stored Records", f"{result.metrics.peak_stored:,}"],
        ["Search Duration", f"{result.metrics.duration_seconds:.4f} s"],
    ]
    print_stamped_table(summary_headers, summary_rows, title="SEARCH EXECUTION RESULTS")

    if result.status == SearchStatus.SUCCESS and result.plan:
        action_seq_str = " -> ".join(ACTION_NAMES.get(a, str(a)) for a in result.plan[:15])
        if len(result.plan) > 15:
            action_seq_str += f" ... (+{len(result.plan) - 15} more)"
        print(f"\nAction Sequence: {action_seq_str}")

        # Render route figure
        fig_name = f"plan_{algo}_{h_name}"
        png_path, pdf_path = render_grid_world(
            inst,
            path=result.trajectory,
            name=fig_name,
            title=f"Plan Route ({algo.upper()} + {h_name.upper()} | Cost: {result.cost:.1f})",
        )
        print(f"\nSaved route figures:")
        print(f"  PNG: {png_path}")
        print(f"  PDF: {pdf_path}")
    else:
        print(f"\n[FAILURE/LIMIT] Plan not found. Reason: {result.failure_reason or result.status.value}")


def cmd_certify(args):
    print(banner("CANONICAL INSTANCE FEASIBILITY CERTIFICATE"))
    inst = get_canonical_instance()
    limits = Limits(max_time_seconds=args.max_time, max_expanded=args.max_expanded)

    cert = certify_feasibility(inst, limits=limits, algo=args.algo)

    if args.json:
        print(json.dumps(cert.to_dict(), indent=2))
        return

    cert_headers = ["Field", "Certificate Outcome"]
    cert_rows = [
        ["Instance ID", cert.instance_id],
        ["Certification Status", cert.status.value],
        ["Feasible", str(cert.status == FeasibilityStatus.FEASIBLE)],
        ["Search Status", cert.search_status],
        ["Solution Cost", f"{cert.cost:.2f}" if cert.cost is not None else "N/A"],
        ["Plan Length", str(len(cert.plan)) if cert.plan else "N/A"],
        ["Expanded Nodes", f"{cert.expanded_nodes:,}"],
        ["Generated Nodes", f"{cert.generated_nodes:,}"],
        ["Evaluation Time", f"{cert.duration_seconds:.4f} s"],
        ["Detailed Reason", cert.reason],
    ]
    print_stamped_table(cert_headers, cert_rows, title="INSTANCE SOLVABILITY CERTIFICATE")


def cmd_rerun(args):
    """Lightweight override-and-rerun parity command matching UI backend."""
    from delivery.ui.backend import run_ui_plan

    inst = get_canonical_instance()
    if args.set:
        overrides = parse_cli_overrides(args.set)
        inst = apply_overrides(inst, overrides)

    res = run_ui_plan(
        inst,
        algo=args.algo,
        heuristic=args.heuristic,
        depth_limit=args.depth_limit,
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return

    print(banner(f"LIVE-CHANGE RERUN: {args.algo.upper()}"))
    rows = [
        ["Status", res["status"]],
        ["Cost", f"{res['cost']:.2f}" if res["cost"] is not None else "N/A"],
        ["Actions", str(res["length"])],
        ["Expanded", f"{res['expanded']:,}"],
        ["Duration", f"{res['duration_s']:.4f} s"],
    ]
    print_stamped_table(["Metric", "Value"], rows, title="RERUN SUMMARY")


def main():
    parser = argparse.ArgumentParser(description="Delivery Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: instance
    inst_parser = subparsers.add_parser("instance", help="Display or generate problem instance.")
    inst_parser.add_argument("--cms", nargs="+", help="Explicit CMS IDs to override group defaults.")
    inst_parser.add_argument("--save", action="store_true", help="Save generated instance to instances/.")
    inst_parser.add_argument("--json", action="store_true", help="Output raw JSON format.")
    inst_parser.set_defaults(func=cmd_instance)

    # Subcommand: plan
    plan_parser = subparsers.add_parser("plan", help="Execute offline search planner.")
    plan_parser.add_argument("--algo", choices=["bfs", "ucs", "dls", "ids", "greedy", "astar"], default="astar", help="Search algorithm.")
    plan_parser.add_argument("--heuristic", choices=["zero", "b1", "b2", "own"], default="b2", help="Heuristic function.")
    plan_parser.add_argument("--set", nargs="*", help="Key=value overrides (e.g. --set slip_probability=0.0).")
    plan_parser.add_argument("--depth-limit", type=int, default=100, help="Depth limit for DLS and IDS.")
    plan_parser.add_argument("--max-expanded", type=int, default=2_000_000, help="Max expanded nodes limit.")
    plan_parser.add_argument("--max-time", type=float, default=60.0, help="Max execution time in seconds.")
    plan_parser.add_argument("--max-stored", type=int, default=None, help="Max stored records limit.")
    plan_parser.set_defaults(func=cmd_plan)

    # Subcommand: certify
    cert_parser = subparsers.add_parser("certify", help="Produce honest instance feasibility certificate.")
    cert_parser.add_argument("--algo", choices=["astar", "ucs"], default="astar", help="Algorithm for certification.")
    cert_parser.add_argument("--max-time", type=float, default=60.0, help="Max certification time in seconds.")
    cert_parser.add_argument("--max-expanded", type=int, default=1_000_000, help="Max expanded nodes.")
    cert_parser.add_argument("--json", action="store_true", help="Output certificate in raw JSON format.")
    cert_parser.set_defaults(func=cmd_certify)

    # Subcommand: rerun
    rerun_parser = subparsers.add_parser("rerun", help="Apply live overrides and rerun planner (UI parity).")
    rerun_parser.add_argument("--algo", choices=["bfs", "ucs", "dls", "ids", "greedy", "astar"], default="astar")
    rerun_parser.add_argument("--heuristic", choices=["zero", "b1", "b2", "own"], default="own")
    rerun_parser.add_argument("--depth-limit", type=int, default=80)
    rerun_parser.add_argument("--set", nargs="*", help="Key=value overrides.")
    rerun_parser.add_argument("--json", action="store_true", help="Output JSON.")
    rerun_parser.set_defaults(func=cmd_rerun)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
