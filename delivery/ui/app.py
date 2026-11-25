"""Streamlit Interactive Dashboard for the Autonomous Delivery Agent (§7.7, M10.1).

Run with:
    streamlit run delivery/ui/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when invoked via `streamlit run delivery/ui/app.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import streamlit as st

from delivery.instance import get_canonical_instance
from delivery.stamp import STAMP, ui_footer, ui_header
from delivery.ui.backend import (
    UISimSession,
    build_configured_instance,
    compute_config_hash,
    get_provider_status,
    render_map_figure,
    run_ui_dispatcher,
    run_ui_plan,
    run_ui_rival_game,
)

st.set_page_config(page_title="CS-370 Delivery Agent", layout="wide")


def _init_session_state() -> None:
    if "plan_cache" not in st.session_state:
        st.session_state["plan_cache"] = {}
    if "active_plan" not in st.session_state:
        st.session_state["active_plan"] = None
    if "active_config_hash" not in st.session_state:
        st.session_state["active_config_hash"] = None
    if "sim_session" not in st.session_state:
        st.session_state["sim_session"] = None
    if "sim_config_hash" not in st.session_state:
        st.session_state["sim_config_hash"] = None


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
        seed_str = st.text_input("RNG Seed", value=str(base_inst.seed))
        try:
            seed = int(seed_str.strip())
            if seed < 0:
                raise ValueError("Seed must be non-negative.")
        except ValueError:
            st.sidebar.error("RNG Seed must be a non-negative integer; using canonical seed.")
            seed = int(base_inst.seed)

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

        # Invalidate stale sim session when sidebar config changes
        if st.session_state["sim_config_hash"] != cfg_hash:
            st.session_state["sim_session"] = None
            st.session_state["sim_config_hash"] = cfg_hash

        col_map, col_info = st.columns([1.2, 1.0])
        with col_map:
            map_placeholder = st.empty()
            sim_sess = st.session_state["sim_session"]
            plan_data = st.session_state["plan_cache"].get(cfg_hash)
            if sim_sess is not None:
                fig = render_map_figure(
                    inst,
                    agent_pos=sim_sess.env.agent_pos,
                    rival_pos=sim_sess.env.rival_pos,
                    trajectory=sim_sess.trajectory,
                    delivered_mask=sim_sess.env.delivered_mask,
                    title=f"Live Execution ({inst.id} | Step {sim_sess.env.step_count})",
                )
            else:
                traj = plan_data["trajectory"] if plan_data else None
                fig = render_map_figure(inst, agent_pos=inst.start, trajectory=traj, title=f"Map ({inst.id})")
            map_placeholder.pyplot(fig)

        with col_info:
            st.subheader("1. Offline Planner")
            if st.button("Plan Route", type="primary"):
                res = run_ui_plan(inst, algo=algo, heuristic=heuristic, depth_limit=depth_limit)
                st.session_state["plan_cache"][cfg_hash] = res
                st.session_state["active_config_hash"] = cfg_hash
                st.rerun()

            plan_data = st.session_state["plan_cache"].get(cfg_hash)
            if plan_data:
                if not plan_data["success"]:
                    st.warning(f"Planner terminated without full solution: {plan_data['status']} ({plan_data['failure_reason'] or 'limit reached'})")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status", plan_data["status"])
                c2.metric("Cost", f"{plan_data['cost']:.1f}" if plan_data["cost"] is not None else "N/A")
                c3.metric("Actions", plan_data["length"])
                c4.metric("Expanded", f"{plan_data['expanded']:,}")
                st.caption(f"Generated: {plan_data['generated']:,} | Peak Stored: {plan_data['peak_stored']:,} | Time: {plan_data['duration_s']:.4f}s")
            else:
                st.info("Click 'Plan Route' to compute a route for the current configuration.")

            st.divider()
            st.subheader("2. Step-by-Step Execution")
            anim_delay_ms = st.slider("Live Animation Delay (ms/step)", min_value=0, max_value=500, value=40, step=10)
            b1, b2, b3 = st.columns(3)
            if b1.button("Step"):
                if st.session_state["sim_session"] is None:
                    st.session_state["sim_session"] = UISimSession(inst, enable_rival=False, window_radius=window_radius)
                st.session_state["sim_session"].step_once()
                st.rerun()
            play_clicked = b2.button("Play")
            if b3.button("Reset"):
                st.session_state["sim_session"] = None
                st.rerun()

            sim_stats_placeholder = st.empty()
            sim_log_placeholder = st.empty()

            if play_clicked:
                if st.session_state["sim_session"] is None:
                    st.session_state["sim_session"] = UISimSession(inst, enable_rival=False, window_radius=window_radius)
                sess = st.session_state["sim_session"]

                def _on_sim_step(s: UISimSession, _entry: dict) -> None:
                    live_fig = render_map_figure(
                        inst,
                        agent_pos=s.env.agent_pos,
                        rival_pos=s.env.rival_pos,
                        trajectory=s.trajectory,
                        delivered_mask=s.env.delivered_mask,
                        title=f"Live Execution ({inst.id} | Step {s.env.step_count})",
                    )
                    map_placeholder.pyplot(live_fig)
                    plt.close(live_fig)
                    sim_stats_placeholder.write(
                        f"**Steps:** {s.env.step_count} | **Cost:** {s.env.total_cost:.1f} | "
                        f"**Delivered:** {bin(s.env.delivered_mask).count('1')}/{len(inst.delivery_stops)} | "
                        f"**Replans:** {s.agent.replan_counts.total_replans()}"
                    )
                    sim_log_placeholder.dataframe(s.event_log[-8:], use_container_width=True)
                    if anim_delay_ms > 0:
                        time.sleep(anim_delay_ms / 1000.0)

                sess.play_all(max_steps=120, on_step=_on_sim_step)

            sim_sess = st.session_state["sim_session"]
            if sim_sess is not None:
                sim_stats_placeholder.write(
                    f"**Steps:** {sim_sess.env.step_count} | **Cost:** {sim_sess.env.total_cost:.1f} | "
                    f"**Delivered:** {bin(sim_sess.env.delivered_mask).count('1')}/{len(inst.delivery_stops)} | "
                    f"**Replans:** {sim_sess.agent.replan_counts.total_replans()}"
                )
                if sim_sess.event_log:
                    sim_log_placeholder.dataframe(sim_sess.event_log[-10:], use_container_width=True)

    with tab_game:
        st.caption(f"Tab Stamp: {STAMP}")
        g_col1, g_col2 = st.columns([1.2, 1.0])
        with g_col1:
            game_map_placeholder = st.empty()
        with g_col2:
            st.subheader("Adversarial Game vs Chaser")
            use_tuned = st.checkbox("Use Tuned Evaluation Weights (config/tuned_weights.json)", value=True)
            game_delay_ms = st.slider("Game Animation Delay (ms/step)", min_value=0, max_value=500, value=40, step=10)
            play_game_clicked = st.button("Play Full Game vs Chaser", type="primary")
            game_stats_placeholder = st.empty()

            if play_game_clicked:
                def _on_game_step(env, traj, ag) -> None:
                    g_live_fig = render_map_figure(
                        inst,
                        agent_pos=env.agent_pos,
                        rival_pos=env.rival_pos,
                        trajectory=traj,
                        delivered_mask=env.delivered_mask,
                        title=f"Live Rival Game (Step {env.step_count} | Cost {env.total_cost:.1f})",
                    )
                    game_map_placeholder.pyplot(g_live_fig)
                    plt.close(g_live_fig)
                    game_stats_placeholder.write(
                        f"**Step:** {env.step_count} | **Cost:** {env.total_cost:.1f} | "
                        f"**Delivered:** {bin(env.delivered_mask).count('1')}/{len(inst.delivery_stops)} | "
                        f"**Replans:** {ag.replan_counts.total_replans()}"
                    )
                    if game_delay_ms > 0:
                        time.sleep(game_delay_ms / 1000.0)

                st.session_state["game_result"] = run_ui_rival_game(
                    inst,
                    window_radius=window_radius,
                    use_tuned_weights=use_tuned,
                    on_step=_on_game_step,
                )

            g_res = st.session_state.get("game_result")
            if g_res:
                st.metric("Game Outcome", g_res["outcome"])
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Cost", f"{g_res['total_cost']:.1f}")
                m2.metric("Steps", g_res["total_steps"])
                m3.metric("Total Replans", g_res["replans"]["total_replans"])
                st.json(g_res["replans"])
        with g_col1:
            g_res = st.session_state.get("game_result")
            if g_res and not play_game_clicked:
                g_fig = render_map_figure(
                    inst,
                    agent_pos=g_res["trajectory"][-1],
                    rival_pos=g_res["rival_pos"],
                    trajectory=g_res["trajectory"],
                    delivered_mask=g_res["delivered_mask"],
                    title=f"Rival Game Replay ({g_res['outcome']})",
                )
                game_map_placeholder.pyplot(g_fig)
                plt.close(g_fig)
            elif not g_res:
                game_map_placeholder.info("Click 'Play Full Game vs Chaser' to watch a live adversarial episode.")

    with tab_llm:
        st.caption(f"Tab Stamp: {STAMP}")
        p_status = get_provider_status()
        st.info(
            f"**Provider:** `{p_status['provider']}` | **Model:** `{p_status['model']}` | "
            f"**API Key Configured:** `{p_status['key_configured']}` (secrets never displayed)"
        )
        l_col1, l_col2 = st.columns([1.0, 1.2])
        with l_col1:
            req_text = st.text_area(
                "Dispatcher Request",
                value="Plan an optimal route to complete all 5 deliveries using A* with our own heuristic.",
                height=100,
            )
            llm_mode = st.selectbox("Execution Mode", ["mock", "replay", "live"], index=0)
            use_val = st.checkbox("Enable Argument Validation & Self-Repair", value=True)
            if st.button("Dispatch Request", type="primary"):
                try:
                    st.session_state["llm_result"] = run_ui_dispatcher(
                        inst,
                        request_text=req_text,
                        llm_mode=llm_mode,
                        use_validation=use_val,
                    )
                except Exception as exc:
                    st.error(f"Dispatcher error: {exc}")
        with l_col2:
            l_res = st.session_state.get("llm_result")
            if l_res:
                st.metric("Dispatcher Status", l_res["status"])
                st.caption(
                    f"Tool Calls: {l_res['tool_calls_count']} | "
                    f"Validation Interventions: {l_res['validation_interventions']} | "
                    f"Latency: {l_res['total_latency_seconds']:.2f}s"
                )
                st.write("**Final Answer:**", l_res["final_answer"])
                st.subheader("Sense–Reason–Action Trace")
                st.json(l_res["steps"])

    ui_footer(st)


if __name__ == "__main__":
    main()
