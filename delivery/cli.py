"""Command-line interface for the delivery agent system.

Usage:
    python -m delivery.cli instance [--cms ID ...] [--json]
"""

import argparse
import json
import sys
from pathlib import Path

from delivery.instance import get_canonical_instance, load_instance, save_instance
from delivery.stamp import banner


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


def main():
    parser = argparse.ArgumentParser(description="Delivery Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: instance
    inst_parser = subparsers.add_parser("instance", help="Display or generate problem instance.")
    inst_parser.add_argument("--cms", nargs="+", help="Explicit CMS IDs to override group defaults.")
    inst_parser.add_argument("--save", action="store_true", help="Save generated instance to instances/.")
    inst_parser.add_argument("--json", action="store_true", help="Output raw JSON format.")
    inst_parser.set_defaults(func=cmd_instance)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
