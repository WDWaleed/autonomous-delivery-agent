# Streamlit UI Manual Verification & Screenshot Checklist (§7.7, M10.4)

**Group:** Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)  
**Launch Command:** `uv run streamlit run delivery/ui/app.py`

---

## 1. Functional Verification Checklist

- [ ] **Group Stamp & Identity:** Confirm `Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)` is visible in the top header, bottom footer, and on all three tabs (`1. Planning & Execution`, `2. Rival Game`, `3. LLM Dispatcher`).
- [ ] **All Search Algorithms:** Run `Plan Route` in Tab 1 for `astar` (`own`, `b2`, `b1`, `zero`), `ucs`, `greedy`, `bfs`, `dls`, and `ids`; confirm `dls` with small depth limit (`5`) reports `CUTOFF` honestly without faking success.
- [ ] **Step-by-Step Execution:** Click `Step`, `Play`, and `Reset` in Tab 1; confirm live map redraws and event log records slips, costs, and replans.
- [ ] **Live Slip Change:** Adjust `Slip Probability (p)` slider (e.g. `0.23` -> `0.0` -> `0.40`); confirm previous execution state resets and no stale metrics remain.
- [ ] **Start Coordinate Override:** Change `Start Row` / `Start Col` to another passable cell (e.g. `(4, 19)`); confirm route replans cleanly and matches `uv run python -m delivery.cli rerun --set start_r=4 start_c=19` (or `slip_probability=0.0`).
- [ ] **Invalid Override Error Handling:** Enter an out-of-bounds or obstacle start coordinate; confirm clear error banner is displayed without crashing.
- [ ] **Twist Toggles:** Toggle `one_way_streets`, `fog_partial_observability`, and `fragile_parcel`; confirm route cost/path updates immediately upon `Plan Route`.
- [ ] **Rival Game Panel (Tab 2):** Toggle `Use Tuned Evaluation Weights` and click `Play Full Game vs Chaser`; confirm W/D/L outcome, replan breakdown, and replay trajectory render cleanly.
- [ ] **Rival Depth Change:** Adjust `Rival Search Depth (plies)` (`2` -> `6`) and rerun Tab 2.
- [ ] **LLM Dispatcher Tab (Tab 3):**
  - Confirm provider/model/key-configured status displays cleanly **without** exposing any API key.
  - Test `mock` mode with `Enable Argument Validation` checked and unchecked.
  - Test `live` mode (optional, uses Groq quota) with a valid request and a contradictory request (e.g. forbidding start `(4, 20)`).

---

## 2. Report Screenshot Checklist

Save stamped screenshots to `report/figures/`:
- [ ] `report/figures/ui_planning_tab.png` — Tab 1 showing A\* (`own`) route on canonical map with stamp.
- [ ] `report/figures/ui_rival_tab.png` — Tab 2 showing completed game replay against `chaser` with W/D/L and replans.
- [ ] `report/figures/ui_dispatcher_tab.png` — Tab 3 showing Sense–Reason–Action trace and validation status.
