"""Compact scenario generator for memory-budget comparisons (§7.4, M5.5).

Generates deterministic windowed/compact instances with short solution depth (D_max <= 14-16)
and 2-3 stops, tailored for memory-bound search evaluations.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from delivery.instance import Instance


def make_compact_scenario(
    seed: int,
    n: int = 10,
    num_stops: int = 2,
    obstacle_density: float = 0.15,
) -> Instance:
    """Generates a compact, windowed instance with 2-3 stops and moderate branching."""
    rng = np.random.default_rng(seed)

    start = (1, 1)
    # Generate 2-3 stops close to start to ensure short solution depth (D_max <= 16)
    possible_stops = [
        (n - 2, n - 2),
        (1, n - 2),
        (n - 2, 1),
        (n // 2, n // 2),
    ]
    stops = possible_stops[:num_stops]

    # Grid generation: mix of road and grass, with obstacles
    cells = rng.choice(
        [-1, 0, 1],
        size=(n, n),
        p=[obstacle_density, (1.0 - obstacle_density) * 0.7, (1.0 - obstacle_density) * 0.3],
    ).astype(np.int8)

    # Ensure start, stops, and cardinal corridors are passable roads
    cells[start] = 0
    for s in stops:
        cells[s] = 0

    # Carve a clear path from start to each stop to guarantee reachability
    for sr, sc in stops:
        r, c = start
        while r != sr:
            r += 1 if sr > r else -1
            cells[r, c] = max(0, cells[r, c])
        while c != sc:
            c += 1 if sc > c else -1
            cells[r, c] = max(0, cells[r, c])

    return Instance(
        id=f"COMPACT-M55-{seed}",
        seed=seed,
        cms_ids=["513303", "522356", "514549"],
        n=n,
        grid=cells,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=obstacle_density,
        start=start,
        delivery_stops=stops,
        slip_probability=0.0,
        adversary="chaser",
        adversary_search_depth=1,
        twists=[],
    )
