"""Tests for Minimax and Alpha-Beta Pruning (§7.5, M6.2)."""

from typing import Dict, List, Optional

import numpy as np
import pytest

from delivery.game.alphabeta import alphabeta_search
from delivery.game.minimax import minimax_search
from delivery.game.stats import placeholder_eval
from delivery.game.window import AGENT, RIVAL, UTILITY_M, GameState, WindowGame
from delivery.instance import Instance


class TextbookTreeGame:
    """Classic Russell & Norvig textbook 2-ply Minimax/Alpha-Beta tree.

    Root (MAX) -> A (MIN), B (MIN), C (MIN)
      A -> [3, 12, 8]    (min = 3)
      B -> [2, 4, 6]     (min = 2 -> beta cutoff after leaf 2!)
      C -> [14, 5, 2]    (min = 2 -> beta cutoff after leaf 2, wait: at 14,5,2 no cutoff until 2!)
    Wait:
      At A: min(3, 12, 8) = 3 -> alpha becomes 3.
      At B: first child is 2 <= alpha (3) -> beta cutoff! Leaves 4 and 6 are PRUNED!
      At C: 14 (>3), 5 (>3), 2 (<=3, last child).
    Total leaves = 9; evaluated leaves = 7 (2 pruned at B).
    """

    def __init__(self, tree: Dict[str, List], leaves: Dict[str, float]):
        self.tree = tree
        self.leaves = leaves
        self.evaluated_leaves: List[str] = []

    def _state(self, node_id: str, turn: int) -> GameState:
        return GameState(
            agent_pos=(0, 0),
            rival_pos=(0, 1),
            delivered_mask=0,
            extra_state=(node_id,),
            turn=turn,
        )

    def root_state(self) -> GameState:
        return self._state("ROOT", AGENT)

    def is_terminal(self, state: GameState, prev_state: Optional[GameState] = None) -> bool:
        return state.extra_state[0] in self.leaves

    def terminal_utility(self, state: GameState, prev_state: Optional[GameState] = None) -> float:
        node_id = state.extra_state[0]
        self.evaluated_leaves.append(node_id)
        return self.leaves[node_id]

    def legal_actions(self, state: GameState, player: int) -> List[int]:
        node_id = state.extra_state[0]
        return list(range(len(self.tree.get(node_id, []))))

    def step(self, state: GameState, action: int) -> GameState:
        node_id = state.extra_state[0]
        child_id = self.tree[node_id][action]
        next_turn = RIVAL if state.turn == AGENT else AGENT
        return self._state(child_id, next_turn)


def test_textbook_tree_exact_pruned_leaves():
    tree = {
        "ROOT": ["A", "B", "C"],
        "A": ["A1", "A2", "A3"],
        "B": ["B1", "B2", "B3"],
        "C": ["C1", "C2", "C3"],
    }
    leaves = {
        "A1": 3.0, "A2": 12.0, "A3": 8.0,
        "B1": 2.0, "B2": 4.0, "B3": 6.0,
        "C1": 14.0, "C2": 5.0, "C3": 2.0,
    }

    game_mm = TextbookTreeGame(tree, leaves)
    res_mm = minimax_search(game_mm, game_mm.root_state(), depth_limit=4)
    assert res_mm.value == 3.0
    assert res_mm.best_action == 0
    assert res_mm.stats.evaluations == 9

    game_ab = TextbookTreeGame(tree, leaves)
    res_ab = alphabeta_search(game_ab, game_ab.root_state(), depth_limit=4)
    assert res_ab.value == 3.0
    assert res_ab.best_action == 0
    # Cutoff 1 at B1 (2 <= alpha=3) prunes B2 & B3; Cutoff 2 at C3 (2 <= alpha=3)
    assert res_ab.stats.cutoffs == 2
    assert res_ab.stats.evaluations == 7
    assert "B2" not in game_ab.evaluated_leaves
    assert "B3" not in game_ab.evaluated_leaves


def test_deterministic_tie_handling():
    tree = {"ROOT": ["A", "B"], "A": ["A1"], "B": ["B1"]}
    leaves = {"A1": 5.0, "B1": 5.0}
    game = TextbookTreeGame(tree, leaves)

    res_mm = minimax_search(game, game.root_state(), depth_limit=2)
    res_ab = alphabeta_search(game, game.root_state(), depth_limit=2)
    assert res_mm.best_action == 0
    assert res_ab.best_action == 0

    # Reversed ordering should pick action 1 first on tie
    rev_order = lambda s, acts, g, d, h: list(reversed(acts))
    res_mm_rev = minimax_search(game, game.root_state(), depth_limit=2, order_fn=rev_order)
    res_ab_rev = alphabeta_search(game, game.root_state(), depth_limit=2, order_fn=rev_order)
    assert res_mm_rev.best_action == 1
    assert res_ab_rev.best_action == 1


def test_alphabeta_equals_minimax_on_200_seeded_positions():
    """Verifies Alpha-Beta == Minimax value across 200 seeded positions, depths 1-5, and orderings."""
    rng = np.random.default_rng(2026)
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[2, 2] = -1
    grid[3, 4] = -1

    inst = Instance(
        id="MM-AB-200",
        seed=2026,
        cms_ids=["513303", "522356", "514549"],
        n=6,
        grid=grid,
        terrain_cost={"road": 3, "grass": 4, "mud": 5, "ford": 7},
        obstacle_density=0.05,
        start=(0, 0),
        delivery_stops=[(0, 5), (5, 5)],
        slip_probability=0.0,
        adversary="chaser",
        adversary_search_depth=6,
        twists=[],
    )
    game = WindowGame(inst, window_radius=3, window_center=(2, 2), allow_wait=False)

    passable_cells = [(r, c) for r in range(6) for c in range(6) if grid[r, c] >= 0]

    for idx in range(200):
        a_pos = passable_cells[int(rng.integers(0, len(passable_cells)))]
        r_pos = passable_cells[int(rng.integers(0, len(passable_cells)))]
        while r_pos == a_pos:
            r_pos = passable_cells[int(rng.integers(0, len(passable_cells)))]

        mask = int(rng.integers(0, 4))
        turn = AGENT if (idx % 2 == 0) else RIVAL
        depth = int((idx % 5) + 1)  # depths 1..5

        state = game.initial_state(
            agent_pos=a_pos,
            rival_pos=r_pos,
            delivered_mask=mask,
            turn=turn,
        )

        # Ensure placeholder_eval is strictly within terminal bounds
        assert abs(placeholder_eval(state, game)) < UTILITY_M

        # Test natural, reversed, and seeded random orderings
        seed_val = 1000 + idx
        if idx % 3 == 0:
            order_fn = None
        elif idx % 3 == 1:
            order_fn = lambda s, acts, g, d, h: list(reversed(acts))
        else:
            def order_fn(s, acts, g, d, h, _sv=seed_val):
                local_rng = np.random.default_rng(_sv + d)
                shuffled = list(acts)
                local_rng.shuffle(shuffled)
                return shuffled

        res_mm = minimax_search(game, state, depth_limit=depth, order_fn=order_fn)
        res_ab = alphabeta_search(game, state, depth_limit=depth, order_fn=order_fn)

        assert res_ab.value == pytest.approx(res_mm.value), (
            f"Position {idx} (depth={depth}): AB={res_ab.value} != MM={res_mm.value}"
        )
        assert res_ab.best_action == res_mm.best_action
        assert res_ab.stats.evaluations <= res_mm.stats.evaluations
