"""Tests for Argument Validation Layer and Self-Repair (§7.6, M8.5)."""

from delivery.llm.client import MockClient
from delivery.llm.react import run_react_dispatcher
from delivery.llm.tools import DispatcherToolbox
from delivery.llm.validation import ArgumentValidator


def test_validator_rejects_invalid_and_contradictory_arguments():
    val = ArgumentValidator()

    # 1. Unknown algorithm
    assert val("plan_route", {"algorithm": "dijkstra_magic"}) is not None
    # 2. Invalid stop ID
    assert val("plan_route", {"algorithm": "astar", "constraints": {"target_stops": [99]}}) is not None
    # 3. Unknown terrain
    assert val("plan_route", {"algorithm": "astar", "constraints": {"avoid_terrains": ["lava"]}}) is not None
    # 4. Cell out of bounds
    assert val("plan_route", {"algorithm": "astar", "constraints": {"forbidden_cells": [[99, 99]]}}) is not None
    # 5. Negative max_cost
    assert val("plan_route", {"algorithm": "astar", "constraints": {"max_cost": -10}}) is not None
    # 6. Contradictory: forbid start depot
    assert val("plan_route", {"algorithm": "astar", "constraints": {"forbidden_cells": [list(val.instance.start)]}}) is not None
    # 7. Contradictory: avoid all 4 terrains
    assert val("plan_route", {"algorithm": "astar", "constraints": {"avoid_terrains": ["road", "grass", "mud", "ford"]}}) is not None
    # 8. Invalid simulate slip_p
    assert val("simulate", {"plan": [0], "slip_p": 1.5}) is not None
    # 9. Invalid inspect_twists pos
    assert val("inspect_twists", {"pos": [-1, 0]}) is not None

    assert val.interventions == 9

    # Valid call returns None
    assert val("plan_route", {"algorithm": "astar", "heuristic": "own"}) is None


def test_react_self_repair_blocks_invalid_calls_before_side_effects():
    tb = DispatcherToolbox()
    val = ArgumentValidator(tb.instance)

    # First call has invalid stop 99 -> blocked by validator (toolbox.call_history untouched!)
    # Second call repairs to stop 0 -> succeeds!
    client = MockClient(
        responses=[
            {
                "content": "Reason: Try stop 99.",
                "tool_calls": [{"id": "1", "function": {"name": "plan_route", "arguments": '{"algorithm": "astar", "constraints": {"target_stops": [99]}}'}}],
            },
            {
                "content": "Reason: Repair to stop 0.",
                "tool_calls": [{"id": "2", "function": {"name": "plan_route", "arguments": '{"algorithm": "astar", "heuristic": "own", "constraints": {"target_stops": [0]}}'}}],
            },
            'Reason: Route planned.\nFINAL_ANSWER: {"status": "SUCCESS"}',
        ]
    )

    res = run_react_dispatcher(
        request_text="Deliver to Stop 0.",
        client=client,
        toolbox=tb,
        request_id="TEST-REPAIR",
        validator=val,
        max_repair_attempts=2,
    )

    assert res.status == "COMPLETED"
    assert res.validation_interventions == 1
    assert val.interventions == 1
    assert res.tool_calls_count == 1
    # Toolbox was NEVER called with stop 99!
    assert len(tb.call_history) == 1
    assert tb.call_history[0]["arguments"]["constraints"]["target_stops"] == [0]
