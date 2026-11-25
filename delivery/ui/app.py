"""Streamlit Interactive Dashboard for the Autonomous Delivery Agent (§7.7, M10.1).

Run with:
    streamlit run delivery/ui/app.py
"""

from __future__ import annotations

import streamlit as st

from delivery.instance import get_canonical_instance
from delivery.stamp import STAMP, ui_footer, ui_header
from delivery.ui.backend import (
    build_configured_instance,
    compute_config_hash,
    render_map_figure,
    run_ui_plan,
)

st.set_page_config(page_title="CS-370 Delivery Agent", layout="wide")


def _init_session_state() -> None:
    if "plan_cache" not in st.session_state:
        st.session_state["plan_cache"] = {}
    if "active_plan" not in st.session_state:
        st.session_state["active_plan"] = None
    if "active_config_hash" not in st.session_state:
        st.session_state["active_config_hash"] = None


def main() -> None:
    _init_session_state()
    ui_header(st)
    st.title("Autonomous Delivery Agent under Uncertainty & Competition")

    base_inst = get_canonical_instance()

    # Sidebar Controls (validated via delivery.overrides)
    with st.sidebar:
        st.header("Configuration & Overrides")
        algo = st.selectbox("Algorithm", ["astar", "ucs", "greedy", "bfs", "dls", "ids"], index=0)
        heuristic = st.selectbox("Heuristic", ["own", "b2", "b1", "zero"], index=0)
        depth_limit = st.slider("DLS / IDS Depth Limit", min_value=5, max_value=150, value=80, step=5)
        slip_p = st.slider("Slip Probability (p)", min_value=0.0, max_value=0.6, value=float(base_inst.slip_probability), step=0.01)

        st.subheader("Start & Stops")
        start_r = st.number_input("Start Row", min_value=0, max_value=base_inst.n - 1, value=int(base_inst.start[0]))
        start_c = st.number_input("Start Col", min_value=0, max_value=base_inst.n - 1, value=int(base_inst.start[1]))

        st.subheader("Twists")
        use_fog = st.checkbox("fog_partial_observability", value=("fog_partial_observability" in base_inst.twists))
        use_oneway = st.checkbox("one_way_streets", value=("one_way_streets" in base_inst.twists))
        use_fragile = st.checkbox("fragile_parcel", value=("fragile_parcel" in base_inst.twists))
        active_twists = [
            name for name, on in [
                ("fog_partial_observability", use_fog),
                ("one_way_streets", use_oneway),
                ("fragile_parcel", use_fragile),
            ] if on
        ]

        st.subheader("Adversary (Assigned)")
        st.text(f"Strategy: {base_inst.adversary} (assigned)")
        rival_depth = st.slider("Rival Search Depth (plies)", min_value=1, max_value=6, value=int(base_inst.adversary_search_depth))
        window_radius = st.slider("Window Radius (Chebyshev)", min_value=2, max_value=8, value=4)
        seed = st.number_input("RNG Seed", min_value=0, value=int(base_inst.seed), step=1)

    try:
        inst = build_configured_instance(
            slip_probability=slip_p,
            start=(int(start_r), int(start_c)),
            stops=list(base_inst.delivery_stops),
            twists=active_twists,
            rival_depth=int(rival_depth),
            seed=int(seed),
        )
    except ValueError as err:
        st.error(f"Invalid override configuration: {err}")
        ui_footer(st)
        return

    cfg = {
        "algo": algo,
        "heuristic": heuristic,
        "depth_limit": depth_limit,
        "slip_p": slip_p,
        "start": [int(start_r), int(start_c)],
        "twists": active_twists,
        "rival_depth": rival_depth,
        "window_radius": window_radius,
        "seed": int(seed),
    }
    cfg_hash = compute_config_hash(cfg)

    # Tabs (every tab renders stamp)
    tab_plan, tab_game, tab_llm = st.tabs(["1. Planning & Execution", "2. Rival Game", "3. LLM Dispatcher"])

    with tab_plan:
        st.caption(f"Tab Stamp: {STAMP}")
        col_map, col_info = st.columns([1.2, 1.0])
        with col_map:
            plan_data = st.session_state["plan_cache"].get(cfg_hash)
            traj = plan_data["trajectory"] if plan_data else None
            fig = render_map_figure(inst, agent_pos=inst.start, trajectory=traj, title=f"Map ({inst.id})")
            st.pyplot(fig)
        with col_info:
            st.subheader("Planner Controls")
            if st.button("Plan Route", type="primary"):
                res = run_ui_plan(inst, algo=algo, heuristic=heuristic, depth_limit=depth_limit)
                st.session_state["plan_cache"][cfg_hash] = res
                st.session_state["active_config_hash"] = cfg_hash
                st.rerun()

            plan_data = st.session_state["plan_cache"].get(cfg_hash)
            if plan_data:
                st.metric("Status", plan_data["status"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Cost", f"{plan_data['cost']:.1f}" if plan_data["cost"] is not None else "N/A")
                c2.metric("Actions", plan_data["length"])
                c3.metric("Expanded", plan_data["expanded"])
            else:
                st.info("Click 'Plan Route' to compute a route for the current configuration.")

    with tab_game:
        st.caption(f"Tab Stamp: {STAMP}")
        st.info("Rival Game panel (wired in M10.3).")

    with tab_llm:
        st.caption(f"Tab Stamp: {STAMP}")
        st.info("LLM Dispatcher panel (wired in M10.3).")

    ui_footer(st)


if __name__ == "__main__":
    main()
