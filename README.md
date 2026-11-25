# CS-370 AI Open Ended Lab: Autonomous Delivery Agent

**Group Stamp:** `Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)`  
**Canonical Instance:** `432382d9` (22×22 grid, 5 delivery stops, `chaser` rival at depth 6, slip `p=0.23`, twists: `fog_partial_observability`, `one_way_streets`, approved 3rd twist: `fragile_parcel`)

Autonomous delivery agent operating in a 2D grid city under terrain costs, stochastic slip dynamics, dynamic city twists, adversarial rival pursuit (`chaser`), and an LLM-based natural-language dispatcher.

---

## 1. Environment Setup & Installation

### Prerequisites
- **Python:** $\ge 3.11$ (tested on Python `3.12.3`, Windows 11 x86_64)
- **Package manager:** `uv` (recommended) or standard `python` + `pip`

### Recommended Setup (`uv`)
```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt -c constraints.txt
```

### Standard `pip` Setup
**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

**Linux / macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

---

## 2. Quickstart (CLI & Tests)

Run the fast unit/integration test suite:
```bash
pytest -q
```

Run the full suite including clean-clone reproducibility (`@pytest.mark.slow`):
```bash
pytest -m slow -q
```

Inspect the canonical instance, certify feasibility, or run a live-parameter override:
```bash
python -m delivery.cli instance
python -m delivery.cli certify
python -m delivery.cli plan --algo astar --heuristic own
python -m delivery.cli rerun --algo astar --heuristic own --set slip_probability=0.0
```

---

## 3. One-Command Reproduction Pipeline

`run_all.py` executes all 13 dependency-ordered stages (`partA_benchmark`, `partA_reflex`, `partB_heuristics`, `partB_inadmissible`, `partB_ids_memory`, `partB_six_algorithms`, `partC_pruning`, `partC_slip`, `partC_games`, `partD`, `ablation`, `failure`, `numbers`) and writes `results/MANIFEST.json`.

```bash
# Quick validation mode (~3.7 min): 3-5 scenarios, 4 games, isolated under results/quick/
python run_all.py --quick

# Full official reproduction mode (writes results/tables/, results/figures/, results/numbers.json, report/generated_numbers.tex)
python run_all.py
```

Useful flags: `--only STAGE1,STAGE2`, `--fresh` (bypass cache), `--llm-mode {replay,live}` (default `replay` for offline determinism), `--workers N` (serial default `1`).

### Quick vs. Full Outputs (D19)
- **Quick mode (`--quick`)** outputs are strictly isolated under `results/quick/` (`results/quick/tables/`, `results/quick/figures/`, `results/quick/numbers.json`, `results/quick/MANIFEST.json`) and **never** overwrite official full-mode results or `report/generated_numbers.tex`.
- **Full mode** outputs live in `results/tables/`, `results/figures/`, `results/numbers.json`, and `report/generated_numbers.tex`, and are the sole source for the final IEEE report.

---

## 4. Interactive UI (Streamlit)

Launch the stamped 3-tab Streamlit application:
```bash
streamlit run delivery/ui/app.py
```

- **Tab 1 — Planning & Execution:** Run `astar`, `ucs`, `greedy`, `bfs`, `dls`, or `ids` with `own`/`b2`/`b1`/`zero` heuristics, toggle twists, modify start/slip/depth, and animate `Step`/`Play`/`Reset`.
- **Tab 2 — Rival Game:** Play full stochastic games against the `chaser` rival with default or tuned evaluation weights (`config/tuned_weights.json`), displaying W/D/L and replans.
- **Tab 3 — LLM Dispatcher:** Dispatch natural-language requests in `mock`, `replay`, or `live` mode with pre-execution argument validation and Sense–Reason–Act trace inspection.

See [`docs/viva_live_changes.md`](docs/viva_live_changes.md) for measured live-change drill timings (all ~2.5 s end-to-end).

---

## 5. Canonical Instance Regeneration & Source Discrepancy (D14)

Regenerate the canonical instance directly from the unmodified instructor script `oel_instance.py` using the group's real CMS IDs:
```bash
python oel_instance.py 513303 522356 514549
python -m delivery.cli instance --cms 513303 522356 514549 --save
```

**Brief vs. Script Discrepancy (D14):** The 4-page assignment brief states 4–6 delivery stops, whereas the unmodified `oel_instance.py` generates 3–6 stops (`1 + rng.integers(3, 7)` picks minus start). Our implementation supports the full 3–6 range without altering `oel_instance.py`; our generated canonical instance (`432382d9`) contains **5** delivery stops.

---

## 6. Repository Structure

- `oel_instance.py`: Unmodified instructor instance generator (SHA-256 pinned).
- `run_all.py`: Dependency-aware single-command reproduction pipeline.
- `delivery/`: Core package (`world/`, `twists/`, `search/`, `heuristics/`, `sim/`, `game/`, `agents/`, `llm/`, `experiments/`, `ui/`, `cli.py`, `stamp.py`).
- `config/`: Group members (`members.json`), tuned rival weights (`tuned_weights.json`), and LLM request suite.
- `instances/`: Saved canonical instance JSON.
- `tests/`: Unit, integration, contract, UI, and clean-clone reproducibility tests.
- `docs/`: Technical specs (`transition_model.md`, `decisions.md`, `source_audit.md`, `viva_live_changes.md`, `failure_analysis.md`).
- `results/`: Stamped official tables, figures, `numbers.json`, and `MANIFEST.json`.
- `report/`: IEEE format report source, `stamp.tex`, and `generated_numbers.tex`.

---

## 7. Known Limitations, Ethics & AI-Assistance Disclosure

- **Known Limitations:** Part B planners and Part C Minimax/Alpha-Beta assume deterministic transitions and do not model stochastic slip ($p=0.23$) during lookahead (handled via simulator slip-triggered replanning). The reduced game operates on a local Chebyshev window ($r=4$), so boundary exits are approximated. Live LLM calls remain subject to provider rate limits and nondeterminism (mitigated via `replay` cache and argument validation).
- **Ethics & Safeguards:** See `docs/failure_analysis.md` and the IEEE report ethics section for analysis of wrong-parcel/hallucinated-argument risks and implemented safeguards (argument validation, deterministic A\* fallback, resource caps, and stamped audit traces).
- **AI-Assistance Disclosure:** AI coding assistance was used during development for implementation scaffolding, test generation, and documentation; all group members have reviewed the codebase and can explain every module and line during the viva.
