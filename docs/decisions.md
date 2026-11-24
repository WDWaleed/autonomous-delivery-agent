# Project Architecture Decisions (D1 – D20)

**Project:** Autonomous Delivery Agent under Uncertainty and Competition  
**Course:** CS-370 Artificial Intelligence, Fall 2026

Every architectural or methodological decision is catalogued below. Any amendment requires rationale documentation and validation against the project rules.

---

| ID | Date | Status | Decision Summary & Contract |
| :--- | :--- | :--- | :--- |
| **D1** | 2026-11-24 | **Active** | Python $\ge 3.11$; isolated virtual environment `.venv`. Dependencies: `numpy`, `scipy` (statistics only), `pandas`, `matplotlib`, `streamlit`, `pytest`, `PyYAML`, `python-dotenv`, and provider-agnostic LLM SDK. LangChain not used by default; custom ReAct loop is standard. NumPy version pinned in `constraints.txt` prior to canonical instance generation in M0.4. |
| **D2** | 2026-11-24 | **Active** | Coordinates are `(row, col)`. Flat 1D index $p = r \times n + c$. Cardinal move order: North `(-1, 0)`, East `(0, 1)`, South `(1, 0)`, West `(0, -1)`. Action `WAIT` is included only when permitted/required by the game or an active twist. |
| **D3** | 2026-11-24 | **Active** | Grid integers `0..3` map to `road, grass, mud, ford`; costs correspond to the script's sorted values (road cheapest, ford dearest). Value `-1` denotes blocked obstacles. Step cost equals cell terrain cost entered plus surcharges/discounts. Transition costs must be finite and strictly positive ($> 0$). |
| **D4** | 2026-11-24 | **Active** | Parcels deliver automatically upon entering the stop cell unless precedence forbids it. Cell remains traversable. Goal reached when all required parcels are delivered. Return to depot is disabled by default (`return_to_depot: false`). |
| **D5** | 2026-11-24 | **Active** | Deterministic search state is a hashable tuple `(pos, mask, *extra)`. `mask` is the bitmask of delivered stops. `extra` contains state variables altering legal actions/transitions (fuel, toll budget, time phase) in twist-registry order. Path cost $g$ is managed by search nodes, not embedded in state. |
| **D6** | 2026-11-24 | **Active** | Slip dynamics are simulator-only. With probability $p_{\text{slip}}$, intended cardinal action is replaced uniformly by one of its two legal perpendicular moves. Planners and Minimax assume deterministic nominal transitions. Illegal intended moves rejected before slip sampling. |
| **D7** | 2026-11-24 | **Active** | Game outcomes: **Loss** = captured or parcel stolen. **Win** = all parcels delivered without capture/theft and delivery cost $C \le \kappa \cdot C^*$ (default $\kappa = 1.5$). **Draw** = delivery complete with $C > \kappa \cdot C^*$, or step cap $T_{\max} = 4 \cdot L^* + 20$ reached. Lexicographic tie-breaking for canonical optimal paths. |
| **D8** | 2026-11-24 | **Active** | Scenario generation for benchmarks: synthetic scenarios generated using deterministic scenario tokens `cms_ids + ["SCN-k"]` or deterministic local derivations from the canonical instance. Feasibility verified prior to benchmark execution. |
| **D9** | 2026-11-24 | **Active** | Statistical reporting standards: Means use Student's $t$ 95% CIs; proportions use Wilson score 95% CIs; paired comparisons use paired-difference CIs. Benchmark suite targets $\ge 30$ independent scenarios. |
| **D10** | 2026-11-24 | **Active** | User interface implemented with Streamlit, providing multi-tab interactive controls, algorithm comparisons, and persistent group identity stamping. |
| **D11** | 2026-11-24 | **Active** | LLM agent is a custom ReAct loop over provider-agnostic `LLMClient` with local record/replay caching for deterministic offline reproduction. |
| **D12** | 2026-11-24 | **Active** | Reduced game model operates on a Chebyshev window of radius $r=4$ around the agent. Nominal action set is $\{N, E, S, W, WAIT\}$ plus enabled twist actions. Rival searches to `adversary_search_depth`. |
| **D13** | 2026-11-24 | **Active** | Default limits per search: 2,000,000 node expansions and 60 seconds wall-clock time limit. Memory measured via `tracemalloc` and reported separately from stored node counts. |
| **D14** | 2026-11-24 | **Active** | **Source Discrepancy:** Assignment brief specifies 4–6 delivery stops; `oel_instance.py` generates 3–6 stops (`1 + rng.integers(3, 7)`). The generator script is preserved unmodified; our implementation accepts 3–6 stops and documents the variance honestly. |
| **D15** | 2026-11-24 | **Active** | All graph search edge costs are strictly positive ($c(s, a, s') > 0$). Zero-cost or negative edges are prohibited to ensure admissibility and correctness of UCS/A* and Dijkstra distance fields. |
| **D16** | 2026-11-24 | **Active** | Unified transition semantics: `docs/transition_model.md` acts as the single source of truth for planners, simulators, and game models. |
| **D17** | 2026-11-24 | **Active** | Terminal event precedence: capture/theft takes precedence over delivery completion; delivery completion takes precedence over step-cap draw. |
| **D18** | 2026-11-24 | **Active** | `peak_stored` counts distinct search records stored in memory (frontier + explored for graph search; stack/path for DFS/IDS), reported distinctly from physical heap bytes. |
| **D19** | 2026-11-24 | **Active** | Quick-mode outputs (`--quick`) are isolated and never overwrite full-mode experimental results required for the final submission report. |
| **D20** | 2026-11-24 | **Active** | Path-cost bound `max_cost` is enforced during search using accumulated $g$. Twist state variables (`max_fuel`, toll budget) are state components because they alter future reachable transitions. |
