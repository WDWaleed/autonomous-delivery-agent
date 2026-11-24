# Source Document Audit: CS-370 Open Ended Lab (OEL)

**Date of Audit:** 2026-11-24  
**Project:** Autonomous Delivery Agent under Uncertainty and Competition  
**Course:** CS-370 Artificial Intelligence, Fall 2026  
**Auditor:** Autonomous Development Agent (Antigravity)

---

## 1. Verified Source Files & Cryptographic Hashes

Both primary authoritative source documents are present directly in the working directory root and have been verified byte-for-byte.

| File | Relative Path | Size | Format / Pages | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| **Requirements Brief** | `CS-370 AI — Open Ended Lab (OEL) Autonomous Delivery Agent Fall 2026.pdf` | 471,290 bytes | PDF / 4 pages | `c9770bea37e661cbda9fb4646b17ba5efdfab0afbba889219a78b3c372527b2b` |
| **Instance Generator** | `oel_instance.py` | 2,579 bytes | Python script | `56a2dcdcff64ae5c7d9ebe138f70921360fa6b15904f3d562fd91c98f0c09d4b` |

*Policy: Neither source document will be modified during the project. The Python script hash is pinned by test fixtures in subsequent milestones.*

---

## 2. Comprehensive Brief Requirements Checklist

Requirements extracted from all 4 pages of the assignment PDF:

### 2.1 Scope, Rules & Hard Constraints

| ID | Description | Page in Brief | Status / Mapping |
| :--- | :--- | :--- | :--- |
| **R-SCOPE** | Touches Labs 1–8: Python/NumPy, LLM agents, PEAS agents, BFS/DFS/UCS, DLS/IDS, Greedy/A*, Minimax, Alpha-Beta. | Page 1 | M1, M3, M4, M6, M7, M8 |
| **R-GROUP** | Group size: 3 to 4 students (negative marking on group of 5 or above). | Page 1 | Config / M0.2 |
| **R-LIB-ALLOW** | Allowed libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `streamlit` / `pygame` / `tkinter`, `LangChain` or any LLM wrapper, hosted/local LLMs, visualization helpers (e.g. `networkx` drawing functions). | Page 1 | Tooling / M0.1 |
| **R-LIB-FORBID** | **Strictly prohibited libraries:** Library implementations of core algorithms (`networkx` shortest-path functions, `simpleai`, AIMA search module, `gym` solvers, ready-made minimax packages). BFS, UCS, IDS, Greedy, A*, Minimax, and Alpha-Beta **must be written from scratch** by the group. | Page 1 | Hard Rule / Enforced in `tests/test_no_forbidden_imports.py` |
| **R-PLAG** | Plagiarism rubric R3 applies to copying repositories or another group's work. | Page 1 | Code written bespoke |
| **R-AI-POLICY** | AI assistants are allowed, provided every member can explain every line in the viva. | Page 1 | Code walkthrough & viva docs (M13) |

### 2.2 Uniqueness Mechanisms (Section 3)

| ID | Description | Page in Brief | Status / Mapping |
| :--- | :--- | :--- | :--- |
| **U-INST** | Deterministic instance generated via `oel_instance.py` from group member CMS IDs: grid size (20–40), terrain cost table, obstacle density, start cell, delivery stops, rival type, slip probability (0.10–0.30), rival depth (3–6). Solvability guaranteed. | Page 1 | M0.4 |
| **U-TWISTS** | Two assigned twists from the script: `fuel_limit_with_charging_stations`, `one_way_streets`, `rush_hour_time_varying_costs`, `fog_partial_observability`, `toll_gates_budget`, `teleport_tunnels`, `wind_drift`, `rival_steals_parcels`. Must change state space, cost function, or transition model (not just UI). | Page 2 | M2.1–M2.4 |
| **U-PROPOSED** | Third twist proposed by group in one paragraph, approved by instructor and unique across groups. | Page 2 | M0.2 / M2.5 |
| **U-HEUR-EVAL** | Own custom heuristic $h$ for Part B and evaluation function for Part C; must differ from textbook standards and be justified. | Page 2 | M5.1, M7.3 |
| **U-STAMP** | Full names and CMS IDs printed on every output (plots, tables, console logs, UI screens, report figures) via `stamp()` helper. Unstamped outputs are not marked. | Page 2 | M0.3, M10, M12 / `tests/test_stamp_everywhere.py` |
| **U-LIVE** | Viva live change: adapt to a live change of one parameter (slip probability, stop location, twist flag) and rerun within ~5 minutes. | Page 2 | M10.4, M11.3, M13 |

### 2.3 Detailed Part Requirements (Section 4)

#### Part A: World Model and Agent Formalisation (Labs 1, 3)
- [x] Load instance into a NumPy grid and build cost map with vectorised operations.
- [x] Benchmark against pure-Python loop version and report speed-up.
- [x] Formulate PEAS description of the agent for the specific instance including both assigned twists.
- [x] Classify environment across standard dimensions (observability, determinism, episodic/sequential, static/dynamic, discrete/continuous, single/multi-agent) with twist justifications.
- [x] State agent type (goal-based vs utility-based) and analyze why a simple reflex agent fails.
- *Mapping:* **Milestones M1.1–M1.3, M3.1–M3.3**.

#### Part B: Route Planning (Labs 4, 5, 6)
- [x] Implement BFS, UCS, DLS, IDS, Greedy Best-First, and A* from scratch.
- [x] Formulate multi-stop delivery task as a single search problem: state includes position, set of parcels delivered, plus twist state (fuel, budget, time step, etc.).
- [x] Design custom heuristic $h$; prove mathematical admissibility; empirically test consistency on the instance.
- [x] Compare custom heuristic against at least two baselines (e.g. scaled Manhattan, distance-to-farthest-undelivered-stop) on: nodes expanded, runtime, peak memory, solution cost.
- [x] Demonstrate an inadmissible but faster variant returning a suboptimal plan and quantify the suboptimality gap.
- [x] Compare IDS behavior against A* under a stored-memory limit, detailing state cycle prevention on the current path.
- *Mapping:* **Milestones M4.1–M4.5, M5.1–M5.6**.

#### Part C: The Rival Agent (Labs 7, 8)
- [x] Model interaction on a reduced game (e.g. local window around agent or abstract junction graph).
- [x] Implement Minimax, then Alpha-Beta with custom move ordering. Report node counts for depths 2 through the rival's depth limit and calculate percentage saved by pruning.
- [x] Simulator slip dynamics: agent moves have stochastic slip ($p \in [0.10, 0.30]$), while minimax/alpha-beta assume determinism. Replan after every slip; report replan frequency, extra cost induced, and failure modes of deterministic assumption.
- [x] Design custom evaluation function ($\ge 4$ weighted features) with justification; tune weights experimentally (grid search).
- [x] Run $\ge 30$ games against assigned rival type; report win, draw, and loss rates with confidence intervals.
- *Mapping:* **Milestones M6.1–M6.3, M7.1–M7.6**.

#### Part D: LLM Dispatcher (Lab 2)
- [x] Build LLM agent with at least 3 tools: `plan_route` (Part B planner), `simulate` (world step), and one custom twist-specific tool.
- [x] Create 20 natural-language dispatcher requests of increasing difficulty (e.g., precedence, terrain avoidance, resource limits).
- [x] Benchmark LLM agent vs direct planner calls on success rate, solution cost, and tool call count. Log errors (wrong tool, hallucinations/invented arguments, ignored constraints).
- [x] Implement pre-execution validation layer checking arguments against instance; quantify error reduction.
- [x] Include Thought/Action/Observation traces for $\ge 2$ runs; map to Sense–Reason–Act cycle.
- *Mapping:* **Milestones M8.1–M8.6**.

#### Part E: Integration, UI and Evaluation
- [x] Interactive UI (Streamlit): shows grid, planned route, rival; algorithm selector; parameter controls; group member stamp on-screen.
- [x] Single reproduction command (`python run_all.py`) reproducing all tables and figures in the report.
- [x] Experiments: $\ge 30$ generated scenarios per configuration (seeded from CMS IDs), 95% confidence intervals, 1 ablation study, failure analysis.
- [x] Half-page ethics and reflection section: impact of rival/LLM wrong decisions and implemented safeguards.
- *Mapping:* **Milestones M9.1–M9.3, M10.1–M10.4, M11.1–M11.5**.

### 2.4 Deliverables, Viva & Grading Rubric (Sections 5–7)

- **Deliverables:**
  1. Source code repository with `README.md`, `requirements.txt` / constraints, and `run_all.py`.
  2. Final Report: **8 to 10 pages in IEEE format**.
  3. Working UI demo.
  4. Viva with live modification.
- **Viva Requirements:**
  - Every member answers questions on any part of the project.
  - Live parameter modification (e.g. slip probability, stop position, twist toggle) tested within ~5 minutes.
  - Specific oral quiz topics: admissibility proof, slip failure threshold ($p=0.5$), pruning efficiency, LLM failure case.
- **Rubric Weighting:**
  - Conceptual Understanding and Presentation: **20%**
  - Implementation and Originality: **30%**
  - Model Selection and Training: **20%**
  - System Deployment and UI: **15%**
  - Documentation and Reporting: **15%**
  - Individual viva adjustment factor (Rubric R4).

---

## 3. Instructor Script (`oel_instance.py`) Technical Audit

### 3.1 Public Interface and Entry Points
1. `reachable(grid, start, targets) -> bool`:
   - 4-connected BFS on cells with `grid[r][c] >= 0`. Validates that all target cells are reachable from start.
2. `stamp(members) -> str`:
   - Formats `[('Full Name', 'CMS ID'), ...]` into `" | ".join(f"{name} ({cms})" for name, cms in members)`.
3. `make_instance(cms_ids: list[str]) -> (dict, np.ndarray)`:
   - Primary deterministic instance factory.
4. CLI execution:
   - Invocation: `python oel_instance.py CMS_ID_1 CMS_ID_2 CMS_ID_3`
   - Writes `grid_<instance_id>.npy` and outputs JSON to stdout.

### 3.2 Seeding & Generation Pipeline
- **Key Derivation:** `key = "|".join(sorted(r.strip().upper() for r in cms_ids))`
- **Hashing:** `h = hashlib.sha256(key.encode()).hexdigest()`
- **Seed:** `seed = int(h[:16], 16)` initialized into `rng = np.random.default_rng(seed)`.
  *Note:* Depends on NumPy's `default_rng` generator algorithm. Must pin exact NumPy version in `constraints.txt`.
- **Generated Schema:**
  - `instance_id`: `h[:8]` (first 8 hex characters).
  - `grid_size`: integer in $[20, 40]$ via `rng.integers(20, 41)`.
  - `terrain_cost`: dictionary mapping `road`, `grass`, `mud`, `ford` to 4 distinct sorted integers from $[1, 9]$.
  - `obstacle_density`: float uniform $[0.15, 0.30]$ rounded to 3 decimal places.
  - `start`: `(row, col)` tuple on an unblocked cell.
  - `delivery_stops`: list of tuples picked from unblocked cells.
  - `adversary`: string chosen from `["chaser", "blocker", "interceptor"]`.
  - `slip_probability`: float uniform $[0.10, 0.30]$ rounded to 2 decimal places.
  - `adversary_search_depth`: integer in $[3, 6]$ via `rng.integers(3, 7)`.
  - `twists`: list of 2 unique strings chosen from `TWISTS`.
  - `grid`: 2D `np.ndarray` of shape `(n, n)`, values $0..3$ for terrains, $-1$ for blocked obstacles.

---

## 4. Discrepancy & Ambiguity Audit

During cross-examination between the assignment brief, `oel_instance.py`, and `implementation.md`, the following discrepancies and edge cases were identified and documented:

### Discrepancy D14: Delivery Stop Count (Brief vs Generator)
- **Brief Specification (Page 1, §3.1):** States delivery stops range from **4 to 6**.
- **Generator Implementation (`oel_instance.py` line 39):**
  ```python
  pick = free[rng.choice(len(free), 1 + int(rng.integers(3, 7)), replace=False)]
  start, stops = tuple(map(int, pick[0])), [tuple(map(int, p)) for p in pick[1:]]
  ```
  `rng.integers(3, 7)` produces values in $\{3, 4, 5, 6\}$. Therefore, the generator produces **3, 4, 5, or 6 stops**.
- **Resolution:** As mandated by project rules, the generator script must **not** be edited. The software will support between 3 and 6 stops dynamically and document this discrepancy in `docs/decisions.md` without fabricating compliance.

### Ambiguity A-RIVAL-DEPTH: Minimax Depth Range (Brief vs Script)
- **Brief Specification (Page 3, Part C):** "Report node counts for depths 2 to the rival's depth limit..."
- **Generator Specification:** `adversary_search_depth` is in $\{3, 4, 5, 6\}$.
- **Resolution:** Minimax and Alpha-Beta experiments will run for depths $d \in [2, \text{adversary\_search\_depth}]$ as required.

### Ambiguity A-TWIST-INTERPLAY:
- Both assigned twists must alter the transition model or state space simultaneously. Milestone M2.4 specifically constructs the combined state space and relaxed base graph.

---

## 5. Audit Conclusion & Milestone Preflight Status

- [x] Both source documents located, audited, and hashed.
- [x] Complete traceability established from brief requirements to project milestones M0 through M13.
- [x] Source discrepancy D14 catalogued.
- [x] Hard rule against forbidden solvers confirmed and scheduled for static AST test verification.
- **Milestone M0.0 Acceptance Criteria:** **SATISFIED.** Preflight is complete; ready for repository and tooling bootstrap in M0.1.
