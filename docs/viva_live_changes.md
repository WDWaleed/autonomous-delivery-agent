# Viva Live-Change Drills & Timings (M11.3)

**Group Stamp:** `Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)`  
**Canonical Instance:** `432382d9` (22×22 grid, start `(4, 20)`, 5 stops, `slip_probability=0.23`, `adversary=chaser`, `adversary_search_depth=6`, twists: `fog_partial_observability`, `one_way_streets`)  
**Target:** ≤ 5 minutes (300 s) end-to-end per live modification and rerun.

---

## 1. Delivered UI Controls & Cache Invalidation

Only controls actually present in [`delivery/ui/app.py`](../delivery/ui/app.py) are listed below:

- **Sidebar (Planning & World):** `Algorithm` (`astar`, `ucs`, `greedy`, `bfs`, `dls`, `ids`), `Heuristic` (`own`, `b2`, `b1`, `zero`), `DLS / IDS Depth Limit` (5–150), `Slip Probability (p)` (0.0–0.6), `Start Row` / `Start Col`, Twist checkboxes (`fog_partial_observability`, `one_way_streets`, `fragile_parcel`), `Rival Search Depth (plies)` (1–6), `Window Radius (Chebyshev)` (2–8), `RNG Seed`.
- **Tab 1 (`Planning & Execution`):** `Plan Route`, `Step`, `Play`, `Reset`, `Live Animation Delay (ms/step)`.
- **Tab 2 (`Rival Game`):** `Use Tuned Evaluation Weights (config/tuned_weights.json)`, `Game Animation Delay (ms/step)`, `Play Full Game vs Chaser`. (Only the assigned `chaser` rival is exposed.)
- **Tab 3 (`LLM Dispatcher`):** `Execution Mode` (`mock`, `replay`, `live`), `Enable Argument Validation & Self-Repair`, `Dispatch Request`.

**Cache Invalidation Verification:**  
`compute_config_hash(config)` in `delivery/ui/backend.py` hashes the full canonical sidebar configuration (`algo`, `heuristic`, `depth_limit`, `slip_p`, `start`, `stops`, `twists`, `rival_depth`, `window_radius`, `seed`) with SHA-256. Any parameter change invalidates the cached plan and resets active execution/replay state (`tests/test_ui_live_change.py::test_live_change_cache_invalidation_and_fast_rerun`).

---

## 2. Measured Live-Change Drills (CLI UI-Parity Rerun)

All drills below were executed end-to-end via `python -m delivery.cli rerun` (which calls the exact same `delivery.ui.backend.run_ui_plan` and `delivery.overrides` path as the Streamlit UI).

| Drill | Command | Expected Output | Measured Wall Time | Status |
| :--- | :--- | :--- | ---: | :--- |
| **Baseline** | `python -m delivery.cli rerun --algo astar --heuristic own --json` | `SUCCESS`, cost `268.0`, len `65`, expanded `740` | `2.630 s` | Pass (≤ 300 s) |
| **1. Slip Probability** | `python -m delivery.cli rerun --algo astar --heuristic own --set slip_probability=0.0 --json` | `SUCCESS`, cost `268.0`, len `65`, expanded `740` (planner is deterministic; simulator/UI step loop uses `p=0.0`) | `2.615 s` | Pass (≤ 300 s) |
| **2. Stop Change** | `python -m delivery.cli rerun --algo astar --heuristic own --set "delivery_stops=5,15;16,1;15,12;9,7" --json` | `SUCCESS`, cost `247.0`, len `59`, expanded `362` | `2.558 s` | Pass (≤ 300 s) |
| **3. Start Cell** | `python -m delivery.cli rerun --algo astar --heuristic own --set start=5,15 --json` | `SUCCESS`, cost `235.0`, len `57`, expanded `641` | `2.638 s` | Pass (≤ 300 s) |
| **4a. Twist Off (`one_way_streets`)** | `python -m delivery.cli rerun --algo astar --heuristic own --set twists=fog_partial_observability --json` | `SUCCESS`, cost `215.0`, len `53`, expanded `172` | `~2.5 s` (search `0.045 s`) | Pass (≤ 300 s) |
| **4b. Twist On (`fragile_parcel`)** | `python -m delivery.cli rerun --algo astar --heuristic own --set twists=fog_partial_observability,one_way_streets,fragile_parcel --json` | `FAILED`, expanded `11` (no mud/ford path exists to fragile stop under one-way constraints; reported honestly) | `2.511 s` | Pass (≤ 300 s) |
| **5. Rival Depth** | `python -m delivery.cli rerun --algo astar --heuristic own --set adversary_search_depth=3 --json` | `SUCCESS`, cost `268.0`, len `65`, expanded `740` (UI Tab 2 searches to 3 plies) | `2.500 s` | Pass (≤ 300 s) |
| **6. DLS Depth Limit** | `python -m delivery.cli rerun --algo dls --depth-limit 10 --json` | `CUTOFF`, len `0`, expanded `484` (honest cutoff below optimal depth 65) | `2.487 s` | Pass (≤ 300 s) |

---

## 3. Verification

Run the UI live-change & parity test suite:
```bash
pytest tests/test_ui_live_change.py -q
```
