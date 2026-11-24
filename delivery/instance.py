"""Instance dataclass, factory, and serialization helpers.

Integrates with oel_instance.py to produce reproducible problem instances.
Governed by decisions D1, D3, D8, D14, and D15.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import oel_instance
from delivery.stamp import MEMBERS, load_members

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = PROJECT_ROOT / "instances"
INSTRUCTOR_SCRIPT = PROJECT_ROOT / "oel_instance.py"
PINNED_SCRIPT_SHA256 = "56a2dcdcff64ae5c7d9ebe138f70921360fa6b15904f3d562fd91c98f0c09d4b"


@dataclass
class Instance:
    """Problem instance specification."""

    id: str
    n: int
    grid: np.ndarray  # int8: 0..3 for terrain, -1 for blocked obstacle
    terrain_cost: Dict[str, int]
    obstacle_density: float
    start: Tuple[int, int]
    delivery_stops: List[Tuple[int, int]]
    adversary: str
    slip_probability: float
    adversary_search_depth: int
    twists: List[str]
    seed: int
    cms_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance attributes (excluding raw numpy grid) to serializable dict."""
        return {
            "instance_id": self.id,
            "grid_size": self.n,
            "terrain_cost": self.terrain_cost,
            "obstacle_density": self.obstacle_density,
            "start": list(self.start),
            "delivery_stops": [list(p) for p in self.delivery_stops],
            "adversary": self.adversary,
            "slip_probability": self.slip_probability,
            "adversary_search_depth": self.adversary_search_depth,
            "twists": self.twists,
            "seed": self.seed,
            "cms_ids": self.cms_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], grid: np.ndarray) -> Instance:
        return cls(
            id=data["instance_id"],
            n=data["grid_size"],
            grid=grid.astype(np.int8),
            terrain_cost=data["terrain_cost"],
            obstacle_density=data["obstacle_density"],
            start=tuple(data["start"]),
            delivery_stops=[tuple(p) for p in data["delivery_stops"]],
            adversary=data["adversary"],
            slip_probability=data["slip_probability"],
            adversary_search_depth=data["adversary_search_depth"],
            twists=list(data["twists"]),
            seed=data["seed"],
            cms_ids=list(data["cms_ids"]),
        )


def verify_instructor_script_hash() -> str:
    """Computes and validates the SHA-256 of oel_instance.py."""
    if not INSTRUCTOR_SCRIPT.exists():
        raise FileNotFoundError(f"Instructor script missing at {INSTRUCTOR_SCRIPT}")
    with open(INSTRUCTOR_SCRIPT, "rb") as f:
        computed = hashlib.sha256(f.read()).hexdigest()
    if computed != PINNED_SCRIPT_SHA256:
        raise ValueError(
            f"oel_instance.py SHA-256 mismatch! Expected {PINNED_SCRIPT_SHA256}, got {computed}"
        )
    return computed


def derive_seed_from_ids(cms_ids: List[str]) -> Tuple[str, int]:
    """Derives key hash and 64-bit integer seed matching oel_instance.py."""
    key = "|".join(sorted(r.strip().upper() for r in cms_ids))
    h = hashlib.sha256(key.encode()).hexdigest()
    seed = int(h[:16], 16)
    return h, seed


def load_instance(
    cms_ids: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Instance:
    """Generates an Instance deterministically using oel_instance.make_instance()."""
    if cms_ids is None:
        members = load_members()
        ids = [cms for _, cms in members]
    else:
        ids = list(cms_ids)

    h, seed = derive_seed_from_ids(ids)
    raw_inst, grid = oel_instance.make_instance(ids)

    # Invariants verification
    grid_int8 = grid.astype(np.int8)
    n = raw_inst["grid_size"]
    start = tuple(map(int, raw_inst["start"]))
    stops = [tuple(map(int, p)) for p in raw_inst["delivery_stops"]]

    inst = Instance(
        id=raw_inst["instance_id"],
        n=n,
        grid=grid_int8,
        terrain_cost=dict(raw_inst["terrain_cost"]),
        obstacle_density=float(raw_inst["obstacle_density"]),
        start=start,
        delivery_stops=stops,
        adversary=str(raw_inst["adversary"]),
        slip_probability=float(raw_inst["slip_probability"]),
        adversary_search_depth=int(raw_inst["adversary_search_depth"]),
        twists=list(raw_inst["twists"]),
        seed=seed,
        cms_ids=sorted(r.strip().upper() for r in ids),
    )

    if overrides:
        for k, v in overrides.items():
            if hasattr(inst, k):
                setattr(inst, k, v)

    return inst


def save_instance(instance: Instance, target_dir: Optional[Path] = None) -> Path:
    """Saves instance.json, grid.npy, twists.json, and meta.json to instances/<id>/."""
    out_dir = target_dir or (INSTANCES_DIR / instance.id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. instance.json
    inst_path = out_dir / "instance.json"
    with open(inst_path, "w", encoding="utf-8") as f:
        json.dump(instance.to_dict(), f, indent=2)

    # 2. grid.npy
    grid_path = out_dir / "grid.npy"
    np.save(grid_path, instance.grid)

    # 3. twists.json
    twists_path = out_dir / "twists.json"
    with open(twists_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "assigned_twists": instance.twists,
                "slip_probability": instance.slip_probability,
                "adversary": instance.adversary,
                "adversary_search_depth": instance.adversary_search_depth,
            },
            f,
            indent=2,
        )

    # 4. meta.json
    meta_path = out_dir / "meta.json"
    meta = {
        "timestamp": datetime.now().isoformat(),
        "instance_id": instance.id,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "oel_instance_sha256": PINNED_SCRIPT_SHA256,
        "cms_ids": instance.cms_ids,
        "seed": instance.seed,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def load_saved_instance(instance_id: str, instances_root: Optional[Path] = None) -> Instance:
    """Loads an existing saved instance from disk."""
    root = instances_root or INSTANCES_DIR
    target_dir = root / instance_id
    inst_path = target_dir / "instance.json"
    grid_path = target_dir / "grid.npy"

    if not inst_path.exists() or not grid_path.exists():
        raise FileNotFoundError(f"Saved instance {instance_id} not found in {target_dir}")

    with open(inst_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    grid = np.load(grid_path)

    return Instance.from_dict(data, grid)


def get_canonical_instance() -> Instance:
    """Returns the canonical instance for the configured group members, saving if needed."""
    members = load_members()
    ids = [cms for _, cms in members]
    _, seed = derive_seed_from_ids(ids)
    h, _ = derive_seed_from_ids(ids)
    canonical_id = h[:8]

    target_dir = INSTANCES_DIR / canonical_id
    if (target_dir / "instance.json").exists() and (target_dir / "grid.npy").exists():
        return load_saved_instance(canonical_id)

    # Generate and save
    instance = load_instance(ids)
    save_instance(instance, target_dir)
    return instance
