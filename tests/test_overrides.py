"""Unit tests for parameter overrides parser and validator (M2.6)."""

import pytest

from delivery.instance import load_instance
from delivery.overrides import apply_overrides, parse_cli_overrides, validate_overrides


def test_validate_valid_overrides():
    raw = {
        "grid_size": 25,
        "slip_probability": 0.15,
        "obstacle_density": 0.20,
        "adversary": "chaser",
        "adversary_search_depth": 5,
        "start": (2, 2),
        "delivery_stops": [(5, 5), (10, 10)],
    }
    cleaned = validate_overrides(raw)
    assert cleaned["n"] == 25
    assert cleaned["slip_probability"] == 0.15
    assert cleaned["start"] == (2, 2)


def test_validate_negative_terrain_cost_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        validate_overrides({"terrain_cost": {"road": 0, "grass": 4}})

    with pytest.raises(ValueError, match="strictly positive"):
        validate_overrides({"terrain_cost": {"road": -1, "grass": 4}})


def test_validate_invalid_ranges():
    with pytest.raises(ValueError, match="slip_probability"):
        validate_overrides({"slip_probability": 1.5})

    with pytest.raises(ValueError, match="grid_size"):
        validate_overrides({"grid_size": 3})

    with pytest.raises(ValueError, match="adversary"):
        validate_overrides({"adversary": "unknown_ai"})


def test_parse_cli_overrides():
    args = ["--slip-probability=0.18", "--grid-size", "30", "--adversary=chaser"]
    cleaned = parse_cli_overrides(args)
    assert cleaned["slip_probability"] == 0.18
    assert cleaned["n"] == 30
    assert cleaned["adversary"] == "chaser"


def test_apply_overrides_to_instance():
    instance = load_instance()
    modified = apply_overrides(instance, {"slip_probability": 0.10, "start": (1, 1)})

    assert modified.slip_probability == 0.10
    assert modified.start == (1, 1)
    # Original unmodified
    assert instance.slip_probability == 0.23
    assert instance.start == (4, 20)
