"""Move ordering strategies for Minimax and Alpha-Beta (§7.5, M6.3).

Provides:
- natural_ordering: Canonical action order (N, E, S, W, WAIT, twist actions)
- make_random_ordering(seed): Seeded deterministic random permutation
- make_own_ordering(stop_fields): Domain-informed heuristic ordering:
  - For AGENT (MAX): prioritizes principal variation hint, then target progress
    via relaxed distance (or Manhattan fallback), then distance from rival/threat.
  - For RIVAL ('chaser', MIN): prioritizes moves minimizing distance to agent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from delivery.game.stats import OrderFn
from delivery.game.window import AGENT, GameState, WindowGame
from delivery.search.problem import ACTION_DELTAS, WAIT


def natural_ordering(
    state: GameState,
    actions: List[int],
    game: WindowGame,
    depth: int,
    pv_hint: Optional[int] = None,
) -> List[int]:
    """Canonical natural action ordering (already sorted by N, E, S, W, WAIT, twist)."""
    return list(actions)


def make_random_ordering(seed: int) -> OrderFn:
    """Returns a deterministic seeded random move-ordering callable."""

    def _random_order(
        state: GameState,
        actions: List[int],
        game: WindowGame,
        depth: int,
        pv_hint: Optional[int] = None,
    ) -> List[int]:
        if len(actions) <= 1:
            return list(actions)
        # Deterministic per-state/depth seed derived from base seed
        h = hash((seed, state.agent_pos, state.rival_pos, state.delivered_mask, state.turn, depth)) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        ordered = list(actions)
        rng.shuffle(ordered)
        return ordered

    return _random_order


def make_own_ordering(
    stop_fields: Optional[List[np.ndarray]] = None,
) -> OrderFn:
    """Returns domain-informed move ordering for AGENT and assigned RIVAL ('chaser').

    AGENT ordering:
      1. Principal variation hint from previous depth (if any)
      2. Progress toward nearest undelivered target (relaxed distance field if provided, else Manhattan)
      3. Distance from rival (preferring safer moves)
      4. Canonical action rank tie-break

    RIVAL ('chaser') ordering:
      1. Principal variation hint (if any)
      2. Minimize distance to agent_pos (aggressive chaser ordering)
      3. Canonical action rank tie-break
    """

    def _own_order(
        state: GameState,
        actions: List[int],
        game: WindowGame,
        depth: int,
        pv_hint: Optional[int] = None,
    ) -> List[int]:
        if len(actions) <= 1:
            return list(actions)

        if state.turn == AGENT:
            undelivered = [
                idx for idx in range(len(game.stops))
                if not (state.delivered_mask & (1 << idx))
            ]
            rr, rc = state.rival_pos
            ar, ac = state.agent_pos

            def _agent_key(act: int) -> Tuple[int, float, float, int]:
                is_pv = 0 if (pv_hint is not None and act == pv_hint) else 1
                if act == WAIT:
                    nr, nc = ar, ac
                elif act in ACTION_DELTAS:
                    dr, dc = ACTION_DELTAS[act]
                    nr, nc = ar + dr, ac + dc
                else:
                    # Twist action (e.g., TELEPORT)
                    nr, nc = game.tunnel_pairs.get((ar, ac), (ar, ac))

                # Target progress
                if undelivered:
                    if stop_fields is not None:
                        target_d = min(float(stop_fields[idx][nr, nc]) for idx in undelivered)
                    else:
                        target_d = min(
                            float(abs(nr - game.stops[idx][0]) + abs(nc - game.stops[idx][1]))
                            for idx in undelivered
                        )
                else:
                    target_d = 0.0

                # Rival distance (negated because larger distance is better)
                rival_d = float(abs(nr - rr) + abs(nc - rc))
                # Penalize stepping right next to or onto rival
                return (is_pv, target_d, -rival_d, act)

            return sorted(actions, key=_agent_key)

        # RIVAL ('chaser') ordering: minimize distance to agent
        ar, ac = state.agent_pos
        rr, rc = state.rival_pos

        def _rival_key(act: int) -> Tuple[int, float, int]:
            is_pv = 0 if (pv_hint is not None and act == pv_hint) else 1
            if act == WAIT:
                nr, nc = rr, rc
            else:
                dr, dc = ACTION_DELTAS[act]
                nr, nc = rr + dr, rc + dc
            dist_to_agent = float(abs(nr - ar) + abs(nc - ac))
            return (is_pv, dist_to_agent, act)

        return sorted(actions, key=_rival_key)

    return _own_order
