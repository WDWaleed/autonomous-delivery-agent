"""Tests for Streamlit UI Layout, Overrides Validation, and Map Rendering (§7.7, M10.1)."""

import matplotlib.pyplot as plt
import pytest

from delivery.stamp import STAMP
from delivery.ui.backend import (
    build_configured_instance,
    compute_config_hash,
    render_map_figure,
)


def test_ui_overrides_validation_and_config_hash():
    inst = build_configured_instance(
        slip_probability=0.15,
        start=(4, 20),
        stops=[(5, 15), (16, 1)],
        twists=["one_way_streets"],
        rival_depth=4,
        seed=101,
    )
    assert inst.slip_probability == pytest.approx(0.15)
    assert inst.start == (4, 20)

    # Out-of-bounds start should raise ValueError via delivery.overrides
    with pytest.raises(ValueError):
        build_configured_instance(
            slip_probability=0.15,
            start=(99, 99),
            stops=[(5, 15)],
            twists=[],
            rival_depth=4,
            seed=101,
        )

    h1 = compute_config_hash({"algo": "astar", "slip_p": 0.1})
    h2 = compute_config_hash({"algo": "astar", "slip_p": 0.2})
    assert h1 != h2


def test_ui_map_figure_contains_stamp():
    inst = build_configured_instance(
        slip_probability=0.23,
        start=(4, 20),
        stops=[(5, 15)],
        twists=[],
        rival_depth=6,
        seed=42,
    )
    fig = render_map_figure(inst, agent_pos=(4, 20), rival_pos=(10, 10))
    texts = [t.get_text() for t in fig.texts]
    assert any(STAMP in txt for txt in texts)
    plt.close(fig)
