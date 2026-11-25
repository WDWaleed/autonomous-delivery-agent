# implementation.md — CS-370 OEL: Autonomous Delivery Agent under Uncertainty and Competition

> **Audience:** an AI coding agent prompted with `Execute M4`, `Execute M4.2`, or `Continue` (the next unticked sub-milestone in §11).
>
> **Owner:** one developer. Work occurs on `main`; short-lived feature branches are permitted for risky experiments.
>
> **Source documents:** the four-page OEL requirements brief (PDF) and instructor script `oel_instance.py`; both are expected to be present in the project working directory before execution. M0.0 locates and audits both. The brief defines academic requirements; the script defines the actual generated-instance behaviour. Where they conflict, preserve both facts, implement the script's actual output, and document the discrepancy without fabricating compliance. The instructor is expected to test the working application; no custom scenario-token input or special instructor-side script invocation is assumed.
>
> **Language / stack:** Python 3.11+, NumPy, SciPy (statistics only), pandas, matplotlib, Streamlit, pytest, PyYAML, python-dotenv, and an LLM provider selected at M8.

---

## 0. Agent operating protocol (read this every session)

1. **Trigger phrases.**
   - `Execute M<n>` authorises every sub-milestone of that milestone, in order. Stop at each `🛑 ASK USER` checkpoint and wait for the required response before crossing it. A pending soft dependency may be worked around only where the milestone explicitly permits it.
   - `Execute M<n>.<m>` authorises exactly that sub-milestone and its acceptance checks. Do not start the next sub-milestone.
   - `Continue` authorises exactly the first unticked sub-milestone in §11 whose hard dependencies are complete. Do not continue to a second sub-milestone in the same turn.
   - If a requested item has an unresolved hard dependency, stop and explain the dependency. Do not silently skip it.
2. **Before starting:** inspect `git status`, `git log --oneline -5`, the target milestone, and its dependencies. The worktree must be clean. Run `pytest -q` and require it to pass. **Bootstrap exception:** before M0.1, the repository and test suite may not exist; M0.1 is the only milestone allowed to establish them. After M0.1, this gate applies. Run slow tests only when the milestone or release checklist explicitly requires `pytest -m slow`.
3. **Context loading and token efficiency:**
   - Read §0 (this operating protocol) at the beginning of every session.
   - Read the complete target sub-milestone, including its tasks, dependencies, acceptance criteria and commit instructions.
   - Read only the relevant portions of the fixed decisions (§3), engineering conventions (§6), architecture (§7), and other sections needed for the current task.
   - Do NOT automatically read all of §0–§7 for every task. This rule supersedes any general instruction to read those sections in full at the beginning of every session.
   - Consult Appendix A only for twists relevant to the current task. Consult other appendices only when necessary.
   - Use targeted searches (`rg`, `grep`, or equivalent) to locate relevant specifications, interfaces, functions and references instead of repeatedly reading entire files.
   - During M0.0, read the complete assignment PDF and `oel_instance.py`. Create `docs/source_audit.md` summarising the authoritative requirements, generator behaviour, instance format, assigned twists and relevant constraints. Refer to this audit for subsequent tasks instead of repeatedly rereading the original sources.
   - Treat `docs/decisions.md`, `docs/source_audit.md`, and other project documentation as working references once their contents have been verified.
   - Never reread the entire `implementation.md` unless explicitly requested or performing a document-wide consistency audit.
   - For changes affecting multiple components, search for all affected definitions, implementations, tests and documentation before making changes. Inspect every relevant specification and update all affected references.
   - Do not repeatedly load completed milestone descriptions unless the current task depends on them.
   - Prefer concise summaries of previously established decisions over repeating large amounts of existing documentation in session notes.
4. **Do only what was requested.** `Execute M<n>` is the only trigger that authorises multiple sub-milestones. Do not infer permission to proceed from a successful commit.
5. **Tests ship with code.** Write tests in the same sub-milestone as the feature. Do not commit with failing tests. Do not weaken or delete a test merely to make an implementation pass; fix the implementation or document a justified design change.
6. **Commits.** Make a commit at every `COMMIT:` marker using the exact message given (a body may be added). Split a commit further if the diff is unusually large. Tick the sub-milestone in §11 in the same commit. A commit is not evidence of milestone completion unless all acceptance criteria have been verified.
7. **🛑 ASK USER.** Stop and ask the user (use a question tool if available, otherwise plain text). Never invent user-supplied data, full names, CMS IDs, API keys, third-twist approval, game-rule approval, or manual-test results. Do not mark an approval checkpoint complete until the user has supplied the required confirmation.
8. **Acceptance.** Run every command in the acceptance list. In the final reply, give a 2–3 sentence summary of what was built, what the user must test manually (if anything), and the next authorised milestone. Do not paste long logs.
9. **Deviations.** Record changes to algorithmic assumptions, state representation, transition semantics, experimental methodology, assignment requirements, reproducibility, or external interfaces in `docs/decisions.md` (date, decision, reason, affected tests/milestones). Minor implementation details that do not alter a documented contract do not require a decision record. Tell the user about every material deviation.
10. **Hard rules (never break):**
    - Never edit `oel_instance.py`. A test pins its SHA-256.
    - Do not use library implementations of BFS, UCS, DLS, IDS, Greedy, A\*, Minimax, or Alpha-Beta. In particular, do not use `networkx` path/search functions, `simpleai`, AIMA, gym solvers, minimax packages, or `scipy.sparse.csgraph`. Own Dijkstra for distance fields is allowed. `tests/test_no_forbidden_imports.py` enforces this.
    - Every plot, table, console log, UI screen and report figure carries the stamp (§6.2). An unstamped output is a bug.
    - Never commit secrets. `.env` is git-ignored; `.env.example` is committed. Never print or include an API key in a log, trace, exception, test fixture or commit.
    - Never hard-code instance-specific values. They derive from the instance/twist files and `config/`.
    - Do not assume a Unix shell. Use `pathlib`, portable Python subprocess calls, and spawn-safe multiprocessing if it is eventually justified. The user is likely on Windows with VS Code.
    - Do not fabricate experimental results, favourable seeds, approvals, or manual checks. Report negative, inconclusive and resource-limited outcomes honestly.
11. **Dependency semantics.** Milestones may declare **hard dependencies** (must be complete before work starts) and **soft dependencies** (work may proceed, but final acceptance or documentation remains blocked). A pending approval must not block unrelated work, but it must never be represented as completed.
12. **Understanding checkpoints.** At the end of each major algorithmic milestone, explain the algorithm's invariants, important implementation decisions, why its tests are meaningful, and known limitations. Viva preparation in M13 consolidates this knowledge; it does not replace learning during implementation.

---

## 1. Things the agent must ask the user (index)

| ID  | Ask                                                                                                                         | When                                  | Why                                                              |
| --- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------- |
| A0  | Submission deadline and viva date                                                                                           | M0.2                                  | Set priorities and decide whether optional work is realistic     |
| A1  | Full names and CMS IDs of every group member (3–4); confirm the exact member-ID set used for the group's canonical instance | M0.2                                  | Stamp and instance generation depend on them                     |
| A2  | Choose a third-twist proposal and obtain instructor approval                                                                | M0.2 (start early), implement at M2.5 | Approval may take time; implementation must not precede approval |
| A3  | Confirm names/IDs render correctly, the intended member-ID set is used, and assigned twists/instance summary look right     | M0.3, M0.4                            | A typo or wrong ID set changes the instance and stamp            |
| A4  | LLM backend choice (hosted API vs local model), who creates the key, and consent to spend quota                             | M8.1 and M8.6                         | Part D needs a usable provider and explicit quota consent        |
| A5  | Choose the group's own heuristic design                                                                                     | M5.1                                  | It must be the group's own design and defensible proof           |
| A6  | Confirm game rules: win/draw/loss, κ, capture, step-cap and event precedence                                                | M6.1                                  | Defines Part C results                                           |
| A7  | Manual checks: figures, routes, replays, UI and full run                                                                    | M1, M2, M5, M7, M10, M11, M12         | These checks require the user's screen, judgement or machine     |
| A8  | Report toolchain: LaTeX (IEEEtran) vs Word                                                                                  | M12.1                                 | Determines report source/build workflow                          |
| A9  | Whether to implement the optional remaining twists as plug-ins                                                              | M11.4                                 | Stretch work only; not on the critical path                      |

---

## 2. Requirement traceability (brief → milestone)

| Brief requirement                                                                                                                     | Milestone                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Load instance into NumPy grid, vectorised cost map, benchmark versus pure Python                                                      | M1.1, M1.2                                                         |
| PEAS (including twists), environment classification, agent type, why reflex fails                                                     | M3.2, M3.3                                                         |
| Two assigned twists change state/cost/transition, not only UI                                                                         | M2.1–M2.4                                                          |
| Third approved, unique twist                                                                                                          | M2.5                                                               |
| BFS, UCS, DLS, IDS, Greedy, A\* from scratch                                                                                          | M4.3, M4.4                                                         |
| Single search problem (position + delivered set + all transition-relevant twist state)                                                | M4.1                                                               |
| Own heuristic, admissibility proof, empirical consistency, at least two baselines on nodes/runtime/memory/cost                        | M5.1–M5.3                                                          |
| Inadmissible-but-faster variant returns a worse plan; report gap                                                                      | M5.4                                                               |
| IDS under a stored-node budget versus A\*, path-cycle avoidance                                                                       | M5.5                                                               |
| Reduced game, Minimax, Alpha-Beta, own move ordering, node counts at depths 2…rival limit, percentage saved                           | M6.1–M6.3                                                          |
| Assigned rival strategy and required adversarial behaviour                                                                            | M7.1–M7.2 (additional rival types only if the brief requires them) |
| Slip in simulator, deterministic planners, replanning statistics, deterministic-model mismatch                                        | M7.4                                                               |
| Own evaluation function (at least four features), justified and experimentally tuned                                                  | M7.3, M7.5                                                         |
| At least 30 games versus rival type with W/D/L and confidence intervals                                                               | M7.6                                                               |
| LLM agent, at least three tools including a twist tool, 20 requests, comparison, failure log, validation, two Sense–Reason–Act traces | M8                                                                 |
| UI with grid, route, rival, algorithm, parameters, names and CMS IDs                                                                  | M10                                                                |
| One command reproduces all tables/figures                                                                                             | M0.1 stub, M9, M11                                                 |
| At least 30 scenarios per full-mode configuration, mean ± 95% CI, ablation, failure analysis                                          | M2.6, M5, M7, M9                                                   |
| README, requirements, `run_all.py`, IEEE report 8–10 pages, UI demo                                                                   | M11, M12                                                           |
| Half-page ethics/reflection                                                                                                           | M12.4                                                              |
| Viva: explain code, live parameter change within about five minutes, instance regeneration                                            | M11.3, M13                                                         |

**Source discrepancy rule:** the four-page brief states 4–6 stops, while the unmodified instructor script can generate 3–6. The application and tests must support every output the script can produce. Do not add, delete or fabricate a stop to conceal the discrepancy. Record the actual stop count in the UI, manifest and report, and state the brief/script discrepancy plainly if the generated group instance has three stops. This discrepancy is not a reason to block implementation or require the instructor to supply custom inputs.

---

## 3. Fixed decisions (changes require user approval and a decision record)

| ID  | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Python ≥ 3.11; virtual environment `.venv`. Dependencies: NumPy, SciPy (statistics only), pandas, matplotlib, Streamlit, pytest, PyYAML, python-dotenv, and the selected LLM SDK. LangChain is not used by default; a small own ReAct loop is the default. The NumPy version must be selected and pinned in `constraints.txt` **before M0.4 generates the canonical instance**, not postponed until M11. Install with `pip install -r requirements.txt -c constraints.txt`. The selected version must be compatible with the instructor script/environment as far as can be established.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| D2  | Coordinates are `(row, col)`. Flat index `p = r*n + c`. Deterministic move order: N(-1,0), E(0,1), S(1,0), W(0,-1). `WAIT` is added only where the game or an enabled twist requires it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| D3  | Grid values `0..3` map to `road, grass, mud, ford`; costs are the script's four sorted values, with road cheapest and ford dearest. `-1` means blocked. Step cost is the terrain cost of the cell entered, plus applicable objective-cost surcharges/discounts; resource-only toll fees are not added to objective cost unless the twist definition explicitly says so. Any `c_min` used in a heuristic is the minimum legal transition cost across the active deterministic transition model (including tunnel actions and deterministic discounts), not automatically the minimum base terrain cost. Every transition cost used by UCS/A\*/Dijkstra must be finite and **strictly positive**. A discount may reduce a cost only if the result remains strictly positive. Negative and zero-cost transitions are prohibited unless the search algorithms are deliberately redesigned and re-proved.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| D4  | A parcel is delivered automatically on entering its stop cell, unless a precedence constraint forbids delivery yet; the cell remains traversable. If the initial position coincides with a stop, apply the same delivery rule during initial-state construction. Goal means all required parcels delivered. Return to depot is off by default (`return_to_depot: false`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| D5  | The deterministic delivery-search state is a hashable tuple `(pos, mask, *extra)`. `mask` is the delivered-stop bitmask. `extra` contains every state variable that can change future legal actions, transition costs, goal status or successor states, in fixed twist-registry order (fuel, remaining budget, time phase, etc.). Fog knowledge is held in the planning problem's immutable belief-map context for one planning episode; it changes between replanning episodes, not during a fixed-map search. The search node's accumulated path cost `g` is kept by the search algorithm, not duplicated in `State`, except where a twist genuinely makes future dynamics depend on accumulated cost.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| D6  | Slip is simulator-only. With probability `slip_probability`, a valid intended cardinal move is replaced by one of its two perpendicular legal moves, selected uniformly from the legal candidates. Planners and Minimax/Alpha-Beta use the nominal deterministic transition model and do not model stochastic slip. If neither perpendicular move is legal, the agent stays and pays an explicit `fallback_stay_cost`, initially the cheapest terrain cost; it is recorded as a slip attempt with no displacement. Illegal intended actions are rejected before slip sampling. Slip outcomes respect blocked cells, one-way edges and other applicable movement restrictions. Fuel/time consequences are applied according to the actual displacement and the explicit WAIT/fallback rules.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| D7  | Proposed game outcome, to be confirmed at A6: **Loss** = captured (same cell; edge-swap helper also supported for any joint-transition/replay model) or parcel stolen. **Win** = all parcels delivered, no capture/theft, and agent delivery cost `C ≤ κ·C*`; default κ = 1.5. **Draw** = no loss, but delivery completes with `C > κ·C*`, or the agent-step cap `T_max = 4 × L* + 20` is reached, where `L*` is the action count of a canonical optimal no-rival plan. Canonical plan ordering is lexicographic: minimise `(total_cost, action_count, action-rank sequence)`, using the fixed cardinal action order N,E,S,W, then WAIT, then enabled twist actions in registry order. This removes ambiguity when several plans have equal optimal cost but different lengths. `T_max` counts courier/agent actions, not rival plies. If `C*=0`, the threshold is 0 and no cost-ratio division is performed (report the ratio as not applicable, or 1 only when both costs are zero). `C*` is the optimal no-rival, no-slip cost computed with the actual full map (an omniscient benchmark), the same instance, enabled deterministic twists, constraints and depot-return setting. Capture/theft has precedence over delivery completion; delivery completion has precedence over step-cap draw. Evaluate terminal conditions after each atomic action. |
| D8  | Scenario `k` is generated locally by the experiment runner using the unmodified instructor generator with a deterministic scenario token, initially `cms_ids + [f"SCN-{k:03d}"]`, if M0.4 confirms the function accepts it. Tokens are internal experiment identifiers only: they are never written to `config/members.json`, the stamp, submission identity or instructor-facing inputs. The instructor is not expected to enter or pass these tokens. If the script rejects arbitrary tokens, inspect its existing public generation/seed interface and use that without editing the script; if no seed interface exists, implement a documented deterministic scenario-derivation wrapper from the canonical instance, validate every derived scenario and label the derivation honestly. Do not stop to ask the instructor for custom input and do not silently treat derived variants as independently generated maps. Reapply group twists/rival/slip unless swept by the experiment. Derive twist RNG seeds from a stable hash, e.g. `int(sha256(f"{scenario_key}:{twist_name}".encode()).hexdigest()[:16], 16)`; do not assume `instance_id` is hexadecimal. Infeasible scenarios are re-seeded `SCN-k-r1`, `-r2` … (maximum five retries); if target N cannot be generated, report the shortfall rather than silently lowering N.                 |
| D9  | Means use t-based 95% CIs; proportions use Wilson 95% CIs; paired comparisons use paired-difference CIs; skewed/median statistics use bootstrap CIs. Full-mode scenario experiments target at least 30 independently seeded scenarios per configuration. If the D8 fallback creates derived variants from one canonical parent rather than independent generator instances, label the design as a derived-scenario sensitivity suite, use parent-cluster-aware uncertainty, and do not describe those variants as 30 independent generator samples. Repeated observations nested within the same request/game/scenario are not treated as independent samples; use paired or cluster-aware analysis. Sample-size/precision rules are declared before inspecting outcomes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| D10 | UI is Streamlit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| D11 | LLM agent is a small own ReAct loop over a provider-agnostic `LLMClient`, temperature 0 by default, with record/replay cache so `run_all.py` works offline. Temperature 0 is not claimed to guarantee identical live outputs. Traces store concise decision/rationale summaries, tool calls and observations; they do not depend on a provider exposing private chain-of-thought.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| D12 | Reduced game = square window of Chebyshev radius `r` (default 4) around the current agent position. The base game action set is N,E,S,W,WAIT; enabled twists may add actions such as deliberate TELEPORT. `legal_actions(state, player)` returns only actions legal for that player and state after terrain, direction, resource, turn and twist checks. The agent moves first; depth is counted in plies. Game state includes both positions, delivery mask, turn, all transition-relevant twist state, accumulated agent delivery cost, agent-action count for `T_max`, and global tick/time phase where needed. The rival searches with its own evaluation to `adversary_search_depth`. Window-boundary moves whose destination lies outside the extracted square are unavailable in that reduced search; this approximation is documented and evaluated.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| D13 | Default limits per search: 2,000,000 expanded nodes and 60 seconds. A separate stored-record budget is optional. A stored-record budget is **not** described as a physical RAM limit. Actual memory is measured separately with `tracemalloc`. A limit hit is recorded as `NODELIMIT`, `TIMEOUT` or `MEMORY`, never hidden.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| D14 | The brief says 4–6 stops; the unmodified script can yield 3–6 (`1 + integers(3,7)` picks minus the start). Code must handle 3–6 because that is the script's actual output range. Do not modify the script or fabricate a stop. If the generated group instance has three stops, state the count and source discrepancy accurately in the report; this is a documented brief/script inconsistency, not an implementation blocker or an instructor-side custom-input requirement.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| D15 | All search edge costs are finite and strictly positive. This supports the assumptions used by UCS, Dijkstra and A\* and avoids zero-cost-cycle complications. Cost-changing twists must preserve this invariant.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| D16 | A single transition-semantics specification in `docs/transition_model.md` governs planner, simulator and game behaviour. Any component-specific difference (especially stochastic slip) must be stated explicitly. The implementation and cross-component tests enforce the same deterministic rules.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| D17 | Terminal-event precedence is capture/theft first, then completed delivery and κ threshold, then agent-step-cap draw. A6 may amend this before the game model is frozen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| D18 | `peak_stored` is a common **stored-record count**, not bytes. Count frontier/explored records for graph-search algorithms and stack/path/required bookkeeping records for DFS/IDS. Report actual peak bytes separately. Publish the exact counting rule in the metrics documentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| D19 | Quick-mode outputs are segregated from full-mode outputs and can never overwrite or satisfy report/release requirements for full-mode results.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| D20 | `max_cost` is a path-cost bound enforced by the search engine using each node's accumulated `g`; it is not a twist-state variable unless a twist's future transitions themselves depend on remaining cost budget. `max_fuel` and toll budget are state variables when they alter future legal transitions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

---

## 4. Repository layout

```text
oel-delivery-agent/
├─ oel_instance.py              # instructor script, UNMODIFIED (SHA-256 pinned)
├─ implementation.md  README.md  requirements.txt  constraints.txt  pyproject.toml  .env.example  .gitignore
├─ run_all.py                   # single reproduction command
├─ config/    members.json  run.yaml  tuned_weights.json  third_twist.yaml
├─ instances/<instance_id>/     instance.json  grid.npy  twists.json  meta.json
├─ data/      requests.yaml     # 20 dispatcher requests + ground-truth specs
├─ delivery/
│  ├─ stamp.py  instance.py  overrides.py  cli.py
│  ├─ world/        grid.py  reference_loops.py  distance.py
│  ├─ twists/       base.py  protocols.py  registry.py  <one module per twist>
│  ├─ sim/          env.py  runner.py
│  ├─ agents/       base.py  reflex.py  hybrid.py
│  ├─ search/       problem.py  core.py  estimate.py  bfs.py  ucs.py  dls.py  ids.py  greedy.py  astar.py  compact.py
│  ├─ heuristics/   base.py  baselines.py  own.py  variants.py
│  ├─ game/         window.py  minimax.py  alphabeta.py  ordering.py  evaluation.py  rivals.py  tuning.py
│  ├─ llm/          client.py  tools.py  react.py  validation.py  requests.py  checker.py
│  ├─ experiments/  scenarios.py  stats.py  cache.py  parallel.py  failure_analysis.py  numbers.py  stages/
│  ├─ viz/          render.py  figures.py
│  └─ ui/           app.py
├─ tests/
├─ results/   figures/  tables/  raw/  logs/  traces/  numbers.json  MANIFEST.json
├─ docs/      source_audit.md  decisions.md  metrics.md  transition_model.md  partA_peas.md  heuristic_design.md  heuristic_proof.md
│             partC_game_model.md  eval_design.md  sra_mapping.md  failure_analysis.md
│             ui_manual_test.md  viva_live_changes.md  code_walkthrough.md  viva_qa.md
│             third_twist.md
└─ report/    main.tex  references.bib  stamp.tex  generated_numbers.tex  check_numbers.py  figures/
```

**Committed:** source, configs, canonical instance files, `data/requests.yaml`, final `results/figures`, `results/tables`, `results/numbers.json`, recorded `results/llm_cache.jsonl` (without secrets), `config/tuned_weights.json`, docs, report source and final report PDF.

**Git-ignored:** `.venv/`, `.env`, `__pycache__/`, `results/raw/`, `results/cache/`, `results/logs/` (unless a final sanitized log is deliberately committed), LaTeX build files, `docs/viva_weak_spots.md`, temporary quick-mode output.

---

## 5. Git workflow (solo)

- Work on `main`. Linear history: no merge commits, no pull requests.
- Optional feature branches only for risky or experimental work: `exp/eval-tuning` (M7.5), `exp/llm-dispatcher` (M8), `exp/extra-twists` (M11.4). Integrate with `git rebase main`, then `git checkout main && git merge --ff-only <branch>`, then delete the branch. `main` must pass `pytest -q`.
- Conventional Commits: `type(scope): subject`, imperative, ≤72 characters. Types: `feat fix test docs chore refactor perf build`. Scopes: `world twists sim agents search heuristic game llm ui experiments stamp config bench report plan release`.
- Annotated tags: `v0.1.0` after M3 (Part A), `v0.2.0` after M5 (Part B), `v0.3.0` after M7 (Part C), `v0.4.0` after M8 (Part D), `v0.5.0` after M10 (UI), `v0.9.0-rc.1` after M11, `v1.0.0` at M13.4. Command: `git tag -a vX.Y.Z -m "<message>"`.
- Commit after each green sub-milestone, never mid-way through failing tests. A milestone tag is created only after the milestone's acceptance criteria and required approval checkpoints are satisfied.

---

## 6. Engineering conventions

### 6.1 Determinism and provenance

- No use of global `np.random` or `random`. Every stochastic component receives an explicit `np.random.Generator` built from a `SeedSequence` derived from the instance seed (`int(sha256(sorted uppercase CMS IDs)[:16], 16)`, matching the script) plus named child streams.
- Select and pin NumPy before canonical instance generation at M0.4. Record `numpy.__version__`, Python version, script SHA-256 and sorted member IDs in `instances/<id>/meta.json`.
- Verify the script's random-generation behaviour in the pinned environment. The golden test checks the committed instance under that environment; it does not by itself prove instructor-machine compatibility.
- Scenario seeds, twist seeds and experiment seeds are explicit and recorded. Random move ordering uses its own named generator.
- Timing is the only routinely non-deterministic result category. Keep timings in a separate `timing` block and compare canonical results excluding runtime, timestamps and machine-specific fields.
- Every experiment manifest records Git commit, instance/config hashes, seeds, package versions, run mode, algorithm settings, hardware/OS and output hashes. LLM stages additionally record provider/model/prompt/tool-schema identifiers where available.

### 6.2 The stamp

`delivery/stamp.py` loads `config/members.json` and builds `STAMP = oel_instance.stamp(MEMBERS)` (same format as instructor helper).

Required behaviour:

- **Console/logs:** print the stamp at process start, at start and end of every stage, and before every printed table. Every persisted log begins and ends with the stamp. Never include secrets.
- **Figures:** save only through `save_figure(fig, name)`, which draws the stamp in a footer and writes PNG plus PDF for the report, with PNG metadata containing the stamp.
- **Tables:** save only through `save_table(df, name)`, which writes CSV (first line `# <stamp>`), Markdown (stamp as caption) and LaTeX (stamp inside `\caption`).
- **UI:** call `ui_header()` and `ui_footer()` on every page/tab/state screen. The full stamp must remain visible in the normal screen layout.
- **Report:** `report/stamp.tex` (generated) defines `\stampline`; every report figure/table caption includes it.
- **Tests:** `tests/test_stamp_everywhere.py` scans final `results/` CSV/MD/TeX/log files for the stamp and checks PNG metadata. It must distinguish intentionally excluded raw/cache files from final deliverables.

### 6.3 Testing

- `pytest -q` is the normal gate. Configure `addopts = "-m 'not slow'"` so slow tests are excluded by default. Mark slow tests `@pytest.mark.slow`; run them explicitly with `pytest -m slow` at M11/M13.
- Property tests use fixed seeds (e.g. A\* cost equals UCS cost on 50 random small scenarios; Alpha-Beta value equals Minimax value on 200 random positions).
- Oracles are **own** brute-force code (e.g. Bellman-Ford, exhaustive state-space Dijkstra on tiny grids), not prohibited library solvers.
- Add cross-component invariants: deterministic planner paths replay legally in the simulator; planner and simulator costs agree; delivered parcels are visited; disabled twists reproduce base dynamics; A\*(h=0) matches UCS; Alpha-Beta matches Minimax; cached non-timing outputs reproduce; UI and CLI overrides resolve to identical configurations.
- Manual tests are listed per milestone as `🛑`; the agent gives exact commands and expected observations.

### 6.4 Metrics and resource accounting

Metrics used where applicable:

- `expanded`: nodes popped and expanded.
- `generated`: successor records generated.
- `peak_stored`: maximum stored search records under the common convention in D18.
- `peak_frontier`, `peak_explored`: where applicable.
- `path_length`: number of actions in the returned plan.
- `cost`: accumulated transition cost.
- `runtime_s`: median of three `perf_counter` runs per scenario/configuration, with no profiler running.
- `peak_mem_bytes`: separate pass with `tracemalloc`, because tracing distorts runtime.
- `reopened`, `stale_pops`, `max_depth_reached`: where applicable.
- `status ∈ {SOLVED, FAILED, CUTOFF, MEMORY, TIMEOUT, NODELIMIT, INVALID}`. `CUTOFF` is distinct from `FAILED`; limits are never reported as infeasibility.

The node/record accounting rule is documented in `docs/metrics.md` or the relevant experiment documentation. `peak_stored` is not presented as bytes or physical RAM.

### 6.5 Performance and parallelism

- Use flat integer positions, precomputed neighbour/cost tables (NumPy converted to Python lists for hot loops), `__slots__` where profiling justifies them, and `heapq` with a monotonically increasing tie-breaker counter.
- Begin experiment execution serially. Profile representative workloads before adding `ProcessPoolExecutor`. Add spawn-safe top-level workers only if the measured runtime improvement justifies the complexity.
- If multiprocessing is enabled, workers must be top-level functions, receive serialisable inputs, use independent explicit RNG streams, and write through the keyed result cache. `workers=1` always uses serial execution.
- Targets: `run_all.py --quick` under five minutes; full run target approximately 90 minutes on the user's laptop. These are targets, not acceptance conditions that justify truncating or falsifying experiments. Measure and record actual duration in `results/MANIFEST.json`.

---

## 7. Architecture and shared contracts

```text
oel_instance.py ──► Instance ──► GridWorld ──► TwistStack ─┬─► DeliveryProblem  (deterministic planners, Part B)
 (unmodified)       (instance.py) (world/)     (twists/)    ├─► DeliveryEnv      (simulator: slip, fog, rival; Parts A/C)
                                                            └─► WindowGame       (Minimax / Alpha-Beta; Part C)
 Planners + heuristics ─► HybridAgent (A* route + tactical Alpha-Beta) ─► Runner (Sense–Reason–Act traces)
 Planner tools + Env tools + twist tool ─► LLM ReAct loop ─► validation layer ─► same planners/env (Part D)
 Experiments (scenarios, stats, cache) ─► results/ ─► numbers.json ─► report/generated_numbers.tex
 UI (Streamlit) and CLI both go through delivery/overrides.py
```

### 7.1 Search contracts

```python
State = tuple  # (pos: int, delivered_mask: int, *transition_relevant_extra)

class Problem(Protocol):
    def initial(self) -> State: ...
    def is_goal(self, state: State) -> bool: ...
    def successors(self, state: State) -> list[tuple[Action, State, float]]: ...

class Heuristic(Protocol):
    name: str
    admissible: bool
    def prepare(self, problem: Problem) -> None: ...
    def __call__(self, state: State) -> float: ...

@dataclass
class SearchResult:
    status: SearchStatus
    plan: list[Action] | None
    states: list[State] | None
    cost: float | None
    path_length: int | None
    expanded: int
    generated: int
    peak_stored: int
    peak_frontier: int | None
    peak_explored: int | None
    runtime_s: float
    peak_mem_bytes: int | None
    algorithm: str
    heuristic: str | None
    reopened: int
    stale_pops: int
    max_depth_reached: int
    termination_reason: str | None
```

- `successors` returns only legal deterministic successors and strictly positive finite step costs.
- Heuristics used by A\* are non-negative, return zero at goal states, and document whether they are proven admissible for the current twist configuration.
- A static position-based relaxed graph for state-dependent twists is a deliberate over-approximation: include the union of every transition that can be legal in any reachable relevant twist state, with the minimum strictly positive cost for each directed edge across those states. This can make the graph more connected/cheaper, never more restrictive/expensive. Preserve direction. If no safe finite position-only relaxation exists, use an expanded relaxed state graph or mark that heuristic unavailable; never force a position-only graph. Resource restrictions (fuel/toll) may be ignored only when doing so adds possibilities and cannot increase edge costs.
- Accumulated path cost `g` belongs to search nodes. The engine enforces an optional `max_cost` by discarding a successor when `g + step_cost > max_cost` (within a documented numeric tolerance). This does not require a duplicate `max_cost` state dimension.
- Fuel, toll budget, time phase and any other variable that changes future transitions/costs belong in `State`.
- Duplicate-state handling must retain the best known `g` for cost-based search. A\* reopens a state when a better `g` is found, including after it was previously closed.
- If `max_cost` is active, algorithms whose priority is not accumulated cost (BFS, Greedy, DLS and IDS) retain non-dominated labels `(depth, g)` per state. A label is dominated only when another label reaches the same state with both no greater depth and no greater `g`. Positive edge costs and a finite `max_cost` bound the number of feasible path labels. UCS/A\* may use best-`g` handling because they optimise cost.
- Search algorithms must handle the initial state already being a goal.

### 7.2 Twist capability contracts

Do not force every twist to implement every hook. `Twist` supplies identity, generation metadata and declared capabilities. Optional protocols/capabilities are implemented only where required:

- `StateExtension`: state component names, initial values and update rules.
- `TransitionModifier`: action filtering and deterministic successor modification.
- `CostModifier`: deterministic cost changes while preserving strictly positive costs.
- `StochasticOutcome`: simulator-only stochastic outcomes (such as slip or wind push).
- `PerceptionModifier`: agent percept/belief updates (such as fog).
- `RelaxationProvider`: relaxed graph/edge contribution for admissible heuristics.
- `Renderer`: visual layer.
- `LLMToolProvider`: twist-specific tool specification and deterministic implementation.

`TwistStack` composes capabilities in fixed registry order. It validates conflicts and state layout, and exposes the deterministic transition model, simulator-only effects, belief/percept updates, relaxed graph and rendering layers separately. It must not call a missing capability as though every twist implemented it.

### 7.3 Game-state contract

Use a frozen, hashable `GameState` dataclass (or a fully documented tuple) containing:

```text
agent_position
rival_position
delivered_mask
turn
agent_twist_state
rival_twist_state (only if any rival transition depends on it)
accumulated_agent_cost
elapsed_agent_steps
elapsed_global_ticks (or a sufficient time phase plus step counter)
```

Include every variable that can affect future legal actions, costs, terminal conditions or evaluations. A parcel theft is immediately terminal, so a stolen-parcel mask is unnecessary unless the rules later allow play to continue after theft. In the game, `turn` and ply depth are distinct from `elapsed_agent_steps`; D7's step cap counts courier/agent actions, not rival plies. Time-dependent twists use a documented global-tick convention; each player action advances one global tick, while only courier moves add agent delivery cost/fuel as specified by the twist.

### 7.4 Constraints and overrides

`Constraints` includes `forbidden_terrains`, `precedence` (list of `(a,b)` meaning deliver `a` before `b`), `max_cost`, `max_fuel` (capacity/initial resource bound when fuel twist is active), and `first_stop`. Path-cost limits are enforced using search-node `g`; fuel and toll budgets are state variables because they change future legal actions.

`delivery/overrides.py` is the single place where overrides are parsed and validated. CLI `--set key=value`, UI and `run_all.py --set` all use it. After any world-changing override, revalidate that start/stops are free, required cells are reachable under enabled deterministic transitions, twist configurations are internally valid, and resource constraints admit a certified route where certification is requested.

### 7.5 Transition specification

M0.4/M2 create `docs/transition_model.md` before simulator integration. It specifies, in order, action validation, stochastic resolution (simulator only), destination legality, base/twist cost, resource/time updates, automatic delivery, rival interaction, terminal-event precedence, and percept construction. The same deterministic semantics are used by `DeliveryProblem`, `DeliveryEnv` when slip is disabled, and `WindowGame`, except for expressly documented model differences (slip, hidden information, rival turns). Cross-component tests enforce parity.

---

## 8. Milestones

Format: **Goal · Dependencies · Tasks · Acceptance · 🛑 (if any) · COMMIT**. Sizes are estimates only: S ≈ one session, M ≈ one–two sessions, L ≈ several sessions.

### M0 — Bootstrap and inputs (S–M)

#### M0.0 Source-document preflight

- **Dependencies:** none. This is the first action when the coding agent is pointed at the working directory.
- **Tasks:**
  - [x] Locate the four-page OEL requirements PDF and `oel_instance.py` in the provided working directory; create `docs/` if it does not yet exist so the audit can be written before Git initialisation. Do not ask the user to re-upload files that are already present. If either is missing, stop and identify the missing filename/path rather than guessing requirements. The expected files are already supplied in the working directory; do not ask the user to upload them again unless they are genuinely absent from the active workspace.
  - [x] Read all four pages of the brief, including tables, footnotes and figures. Create `docs/source_audit.md` with a concise checklist of every explicit deliverable, constraint, marking requirement, page-count rule, live-demo/viva requirement and any wording that is ambiguous. Record the PDF filename, page count and the page supporting each requirement.
  - [x] Read the complete `oel_instance.py` source. Record its public entry points, expected member-ID format, seeding/hash behaviour, generated fields, stop-count range, assigned-twist representation, dependencies and any validation assumptions in `docs/source_audit.md`. Do not edit the script.
  - [x] Compare the source documents against §2 and the rest of this plan. Add any omitted requirement to the traceability table and the appropriate milestone. If the four-page brief contradicts this plan, the brief's explicit academic requirement must be represented; if the generator contradicts it, preserve the generator's actual behaviour and record the discrepancy without fabricating compliance.
  - [x] Confirm that the source PDF and script are available to subsequent milestones from stable project-relative paths. Record hashes for both in `docs/source_audit.md`; M0.4 separately pins the script hash in tests. Do not modify the four-page PDF. Keep it in the working directory or a documented `source_docs/` path; whether it is committed is governed by course/submission rules, not by the coding agent's assumption.
- **Acceptance:** all four PDF pages and the full script have been inspected; `docs/source_audit.md` exists; every explicit brief requirement maps to a milestone or is flagged for resolution; no source file was modified.
- **Commit handling:** M0.0 occurs before `git init`, so it has no separate commit. Include `docs/source_audit.md` and tick M0.0 in the first M0.1 repository commit.

#### M0.1 Repository and tooling

- **Hard dependencies:** M0.0 source-document preflight (the source files remain outside the code layout until their location/tracking policy is recorded). The repository and test suite may not yet exist, so the bootstrap exception in §0 applies.
- **Tasks:**
  - [x] `git init -b main`; create §4 layout (packages with `__init__.py`), `.gitignore`, `pyproject.toml` (`pythonpath=["."]`, `slow` marker, default `addopts="-m 'not slow'"`), initial `requirements.txt` plus a `constraints.txt` file used to pin NumPy before M0.4, `.env.example`, README skeleton (venv creation/activation for Windows and Unix, install command `pip install -r requirements.txt -c constraints.txt`).
  - [x] `run_all.py` stub with `--quick --only STAGE[,STAGE] --fresh --workers N --llm-mode {replay,live} --set key=value`; stage registry under `delivery/experiments/stages/`; stamped banner; `results/MANIFEST.json` with Python/NumPy versions, seeds, Git commit, run mode and per-stage timing/status.
  - [x] Create `docs/decisions.md` seeded with D1–D20 and the D14 discrepancy; retain `docs/source_audit.md` created in M0.0.
  - [x] Create `docs/transition_model.md` skeleton and `docs/metrics.md` skeleton. The `constraints.txt` file created above starts empty; M0.4 fills it with the exact NumPy pin, while M11.2 pins the remaining top-level dependencies in `requirements.txt`.
  - [x] Add `tests/test_smoke.py` and `tests/test_no_forbidden_imports.py` (static scan of `delivery/` for forbidden imports/calls).
  - [x] `.vscode/settings.json` pointing at `.venv` and enabling pytest (optional).
- **Acceptance:** `pytest -q` passes; `python run_all.py --help` works; the source PDF/script remain available at documented relative paths; `git status` is clean after the initial commit. Tick M0.0 and M0.1 in this same initial commit because M0.0's audit was performed before Git existed.
- **COMMIT:** `chore: initialise repository, tooling and run_all stub`

#### M0.2 🛑 Collect inputs (A0, A1, A2)

- **Tasks:**
  - [x] Ask A0 (deadline and viva date).
  - [x] Ask A1: full names and CMS IDs of all members (3–4), and confirm the exact member-ID set used to generate the group's canonical instance with `oel_instance.py`. Write `config/members.json` and `config/run.yaml` (initial defaults: `n_scenarios: 30`, `n_games: 50`, `window_radius: 4`, `kappa: 1.5`, `return_to_depot: false`, limits from D13).
  - [x] Tell the user about D14 (script gives 3–6 stops while brief says 4–6). This is documented as a source discrepancy; it does not require the instructor to enter custom data or block implementation. Do not claim a three-stop generated instance satisfies the brief's stated four-stop minimum.
  - [x] Ask A2. Explain that the third twist needs a one-paragraph proposal, instructor approval and uniqueness. Offer Appendix B candidates; recommend Fragile Parcel as the lowest-risk option, subject to assigned-twist compatibility.
  - [x] Explain that M2.5 implementation waits for instructor approval, while unrelated framework/search work may proceed.
  - [x] Tell the user an LLM key or local model is needed at M8; the key stays in their local `.env` and must never be exposed in logs or commits.
- **Acceptance:** member file and run config are validated; unresolved approval items are explicitly tracked.
- **COMMIT:** `chore(config): add group members and run configuration`

#### M0.3 Stamp infrastructure

- **Hard dependencies:** M0.2.
- **Tasks:**
  - [x] `delivery/stamp.py`: `STAMP`, `banner()`, stamped logger, `save_figure`, `save_table`, `stamp_text_file`, UI header/footer stubs, report `stamp.tex` generator.
  - [x] Tests: stamp equals `oel_instance.stamp(MEMBERS)`; figure footer and PNG metadata; CSV first line; Markdown caption; LaTeX caption.
- **Acceptance:** `pytest -q`; `python -m delivery.stamp --demo` prints banner and writes `results/figures/stamp_demo.png`.
- **🛑 A3:** show the user the banner and demo figure; user confirms all names and IDs are spelled correctly.
- **COMMIT:** `feat(stamp): add stamped console, figure and table output helpers`

#### M0.4 Instance generation and loading

- **Hard dependencies:** M0.3.
- **Tasks:**
  - [x] Keep the provided `oel_instance.py` at the repository root unmodified. If it is already there, verify it rather than overwrite it; if it is elsewhere in the working directory, copy its exact bytes once. Calculate and pin SHA-256 in `docs/decisions.md` and a test.
  - [x] Inspect its ID parsing, random seeding and dependency behaviour. Select the exact NumPy version and write `numpy==<exact-version>` to `constraints.txt` now, before creating canonical golden data; verify installation uses `-c constraints.txt`. Verify whether an internal synthetic scenario token is accepted for D8; never pass it as a real member to stamp generation. These tokens are used only by the project's experiment runner; the instructor is not expected to enter or provide them.
  - [x] `delivery/instance.py`: `Instance` dataclass (`id`, `n`, grid `int8` with −1 blocked, terrain-cost mapping, start, stops, rival, slip probability, rival search depth, twists, seed); `load_instance(cms_ids, overrides=None)` calls `make_instance`; `save_instance` writes `instances/<id>/{instance.json,grid.npy,meta.json}` with Python/NumPy versions, script hash and sorted IDs.
  - [x] Tests: golden regeneration equals committed files; ID order/case/whitespace invariance; invariants (n 20–40, 3–6 stops, start/stops free, four-connected base reachability where the script promises it, slip 0.10–0.30, rival depth 3–6, two distinct valid twists, four distinct costs).
  - [x] CLI `python -m delivery.cli instance` prints stamped summary (size, costs, start, stops, rival, slip, depth, assigned twists).
  - [x] Document whether scenario-token generation in D8 is supported. If arbitrary tokens are rejected, use an existing supported seed/ID interface exposed by the script. If none exists, implement and document a deterministic local scenario-derivation wrapper from the canonical instance, with explicit seeded transformations and feasibility validation; label such cases as derived scenarios rather than direct generator outputs. Never edit the instructor script, invent member identities, or request custom instructor-side input.
- **Acceptance:** `pytest -q`; CLI output matches `python oel_instance.py <real group IDs>` JSON under the exact NumPy version recorded in `constraints.txt`; script hash matches; canonical instance is committed; internal scenario-token or documented derived-scenario behaviour is tested.
- **🛑 A3:** show the user assigned twists, member IDs and stop count; confirm the intended group member-ID set. Explain the brief/script stop-count discrepancy and show the actual generated count; no instructor-side custom input is required.
- **COMMIT:** `feat(instance): vendor instructor generator and load instance deterministically`
- **COMMIT:** `chore(instance): add generated instance files`

### M1 — World model and vectorised cost map (Part A, technical) (M)

#### M1.1 Grid world

- **Hard dependencies:** M0.4.
- **Tasks:**
  - [x] `delivery/world/grid.py` `GridWorld`: grid, passable bool mask, entering-cost map (`int32`, blocked sentinel), terrain lookup table, flat neighbour table `[n²,4]` (−1 off-grid/blocked), and move-cost table `[n²,4]`, built with vectorised NumPy rather than per-cell Python loops.
  - [x] Helpers `idx(r,c)`, `rc(p)`.
  - [x] Tests: lookup correctness, blocked impassable, neighbour symmetry for ordinary two-way edges, border handling, start/stops passable. Directional twists are applied later and must not be incorrectly asserted symmetric.
- **Acceptance:** `pytest -q`; grid arrays agree with a small hand-constructed reference.
- **COMMIT:** `feat(world): build NumPy grid, cost map and neighbour tables (vectorised)`

#### M1.2 Loop reference and benchmark

- **Tasks:**
  - [x] `world/reference_loops.py`: pure-Python cost-map, passability-mask and move-cost reference; tests assert exact equality with vectorised output.
  - [x] Benchmark stage `partA_benchmark`: sizes `[20, 30, instance n, 200, 1000]`; at least 30 repeats for ordinary sizes and fewer documented repeats for 1000×1000 if runtime is excessive. Include warm-ups; measure median and mean, variability, speed-up, absolute time saved and paired confidence intervals where appropriate.
  - [x] Stamped table `partA_speedup` and log-scale figure.
- **Acceptance:** `python run_all.py --only partA_benchmark --quick`; no hard requirement that speed-up exceed 1 at every size. The report explains NumPy overhead on small arrays and observed crossover behaviour.
- **COMMIT:** `feat(bench): vectorised vs loop cost-map benchmark`

#### M1.3 Own distance fields and renderer

- **Tasks:**
  - [x] `world/distance.py`: own `heapq` Dijkstra `distance_field(world, source, reverse=False, edge_model=None)`. Directed reverse-Dijkstra must be correct for distances from arbitrary positions to a fixed target. `edge_model` permits one-way/tunnel and other deterministic relaxed edges.
  - [x] Tests against own Bellman–Ford oracle on random small directed and undirected graphs.
  - [x] `viz/render.py` `render_world(world, route=None, agent=None, rival=None, layers=())`, colour-blind-safe terrain palette, start/stops/blocked marked and optional route polyline. Save only through `save_figure`.
- **Acceptance:** `pytest -q`; Dijkstra and oracle agree on fixtures.
- **🛑 A7:** open `results/figures/world_overview.png` and confirm the map looks right.
- **COMMIT:** `feat(world): own Dijkstra distance fields and world renderer`

### M2 — Twist framework (M–L)

#### M2.1 Plugin framework

- **Hard dependencies:** M1.1 and M1.3.
- **Tasks:**
  - [x] `twists/base.py`, `twists/protocols.py`, `twists/registry.py`: minimal Twist identity/generation base plus optional capability protocols from §7.2. Reserve all eight twist names and fixed registry order.
  - [x] `TwistStack` composes enabled twists, validates conflicts, builds extra-state layout, action filters, deterministic cost/transition hooks, simulator-only stochastic hooks, percept hooks and relaxed edges.
  - [x] Persist generated twist data in `instances/<id>/twists.json`.
  - [x] `twists_enabled` flags in config/overrides.
  - [x] Tests: empty stack reproduces base dynamics; toggling a stateful twist changes state layout consistently; a twist without a capability does not require a dummy hook; registry ordering is deterministic.
- **Acceptance:** `pytest -q`; transition-spec skeleton updated.
- **COMMIT:** `feat(twists): add twist plugin framework and registry`

#### M2.2 Assigned twist A

- **Hard dependencies:** M2.1.
- **Tasks:** implement the first assigned twist according to Appendix A and the actual instructor assignment: deterministic generation, feasibility-safe parameters, state/cost/transition effects, relaxed-graph contribution, simulator/percept effect where applicable, render layer, LLM tool spec, and focused tests. Render a stamped figure.
- **Acceptance:** tests demonstrate a real state/cost/transition/perception change as appropriate; relaxed model satisfies its contract.
- **COMMIT:** `feat(twists): implement fog_partial_observability`

#### M2.3 Assigned twist B

- **Hard dependencies:** M2.1.
- **Tasks:** same requirements as M2.2 for the second assigned twist.
- **Acceptance:** tests demonstrate the actual twist effect and valid relaxed model.
- **COMMIT:** `feat(twists): implement one_way_streets`

#### M2.4 Twist interplay and relaxed graph

- **Hard dependencies:** M2.2 and M2.3.
- **Tasks:**
  - [x] Test both twists together: state layout, capability composition order, action filtering, transition legality, cost accumulation and simulator effects.
  - [x] `TwistStack.relaxed_graph()` returns a directed edge model that includes every deterministic transition available in the actual planning problem at no greater cost, while relaxing restrictive constraints. Simulator-only slip/wind randomness is excluded from deterministic planner semantics and documented as such.
  - [x] For directed pair distances used by the MST, define the undirected lower-bound edge weight `w(u,v) = min(d(u,v), d(v,u))`, where each directed distance is computed in the relaxed graph. Any feasible directed traversal from one required waypoint to the next costs at least this symmetric lower-bound edge weight; the MST built from these weights is therefore a lower bound on a route connecting the required waypoints.
  - [x] Property tests: every actual deterministic edge has a no-more-expensive relaxed counterpart; random legal walks cost at least the relaxed distance; exact small-graph checks. Random-walk tests are sanity checks, not the admissibility proof.
- **Acceptance:** `pytest -q`; proof obligations documented in `docs/transition_model.md` and referenced by `docs/heuristic_proof.md`.
- **COMMIT:** `test(twists): twist interplay and relaxed-graph lower-bound tests`

#### M2.5 🛑 Third twist (A2)

- **Soft dependency:** other M2 work may proceed while approval is pending; implementation and final Part A acceptance may not.
- **Tasks:**
  - [x] If no approved proposal exists, draft `docs/third_twist.md` (one paragraph: rule, state/cost/transition effect, unique name, conflicts and heuristic relaxation) from the user's selected candidate. Ask the user to submit it and stop until approval is confirmed.
  - [x] After written/explicit approval, implement it as a plugin with tests and `config/third_twist.yaml`.
  - [x] It must change state space, cost or transition, not only UI, and must not conflict with assigned twists.
  - [x] Update PEAS and transition docs after the approved rule is fixed.
- **Acceptance:** approval recorded; plugin tests pass; actual deterministic transition or resource effect is demonstrated.
- **COMMIT:** `feat(twists): implement fragile_parcel`

#### M2.6 Overrides, scenario generator, statistics and cache

- **Hard dependencies:** M2.1; assigned-twist generator hooks from M2.2/M2.3. Third-twist support is a soft dependency until approved; configurations without it remain clearly labelled provisional.
- **Tasks:**
  - [x] `delivery/overrides.py` with parsing/validation and tests.
  - [x] `experiments/scenarios.py`: `scenario(k, base, overrides)` per D8. Use direct script-generated scenarios when internal tokens are accepted; otherwise use the documented deterministic local derivation from M0.4. The runner, not the instructor, supplies scenario keys. Record whether each scenario is direct or derived. Initial feasibility certificate is base reachability; full planner certificate is wired at M4.5.
  - [x] `experiments/stats.py`: `mean_ci`, `wilson_ci`, `paired_diff_ci`, `bootstrap_ci`; tests with known values (t critical ≈2.045 for n=30; Wilson edge cases at 0/30 and 30/30; paired-difference behaviour). Bootstrap supports scenario/request-level clustering where required.
  - [x] `experiments/cache.py`: result cache keyed by canonical hash of config, seed, source-file hash and run mode; `--fresh` bypasses. Never cache secrets.
  - [x] `experiments/parallel.py`: serial-first interface; spawn-safe pool helper may be implemented but remains unused until profiling justifies it. `workers=1` is serial.
- **Acceptance:** `pytest -q`; 30 feasible independently seeded generator scenarios, or a clearly labelled derived-scenario suite when generator seeding is unavailable, are produced within practical time; shortfalls and dependence structure are reported; repeated runs match non-timing outputs.
- **🛑 A7:** show assigned-twist layer figures and ask user to confirm that the rules look as expected. Third twist shown separately if approval is pending.
- **COMMIT:** `feat(experiments): seeded scenario generator, CI statistics and result cache`

### M3 — Simulator and agent formalisation (completes Part A) (M)

#### M3.1 Stochastic environment

- **Hard dependencies:** M2.1–M2.4 and `docs/transition_model.md`.
- **Tasks:**
  - [x] `sim/env.py` `DeliveryEnv(world, twists, slip_p, rng)` with `reset()`, `step(action) → (percept, cost, events)`, exposing true state to planners only in fully observable deterministic configurations. Implement deterministic mode (`slip_p=0`), fog percepts/belief updates if active, WAIT where required and rival hook stub.
  - [x] Follow the canonical transition order. Separate intended action, sampled action and realised destination in events.
  - [x] Events log: slip attempt, realised slip/displacement, fallback stay, delivery, illegal move, cost, resource updates and twist events.
  - [x] Tests: controlled slip-selection frequency over ≥20,000 eligible opportunities; perpendicular rule and fallback separately; blocked/one-way respected; auto-delivery; cost accounting equals transition-cost sum; zero-slip simulator transitions match `DeliveryProblem` successors on shared states.
- **Acceptance:** `pytest -q`; simulator/planner parity tests pass for deterministic configurations.
- **COMMIT:** `feat(sim): stochastic delivery environment with slip and twist hooks`

#### M3.2 Sense–Reason–Act runner and reflex baseline

- **Hard dependencies:** M3.1.
- **Tasks:**
  - [x] `sim/runner.py`: `run(agent, env)` logs percept → concise decision summary → action/observation per step; reused for Part D traces. Do not require private chain-of-thought from an LLM.
  - [x] `agents/reflex.py`: stateless rule that steps to the neighbour that most reduces Manhattan distance to nearest undelivered stop, avoids blocked/illegal cells, and has no memory. Define deterministic tie-breaking and behaviour when every move fails to reduce distance.
  - [x] Stage `partA_reflex`: 30 scenarios; report success rate within step cap and mean cost of successful runs (mean ± CI; Wilson CI for success).
  - [x] Qualitative stamped failure pictures (walls, loops, local minima) with event explanations.
- **Acceptance:** `pytest -q`; reflex outputs are deterministic for fixed state and tie-breaking.
- **COMMIT:** `feat(agents): sense-reason-act runner and simple reflex baseline`

#### M3.3 PEAS, environment classification and agent type

- **Hard dependencies:** M3.2. **Soft dependency:** third-twist approval/implementation; final Part A completion is blocked until the approved twist is represented.
- **Tasks:** [x] `docs/partA_peas.md` contains PEAS for the actual assigned and approved twists; seven-row environment classification (observable, deterministic/stochastic, episodic/sequential, static/dynamic, discrete/continuous, single-/multi-agent, known/unknown), with one-paragraph justifications tied to actual twists, slip and rival; choice of utility-based (or goal-based with explicit justification) agent; why simple reflex fails, with numbers from `partA_reflex` rather than hard-coded values. If approval remains pending, label the draft incomplete and do not tick final acceptance.
- **🛑 A7:** user reads the document (must be able to defend it at viva) and requests edits.
- **Acceptance:** document matches implemented environment and instructor-approved twist set; all reported metrics are generated.
- **COMMIT:** `docs(partA): PEAS description, environment classification and agent type`
- **TAG:** `git tag -a v0.1.0 -m "Part A complete"`

### M4 — Search engine and algorithms (Part B, 1/2) (L)

#### M4.1 Single search problem

- **Hard dependencies:** M3.1 and the deterministic transition contract. Third twist is a hard dependency only if it is approved/required for the instance's assigned configuration; do not claim all-twists feasibility before it exists.
- **Tasks:** [x] `search/problem.py` `DeliveryProblem(world, twist_stack, constraints, return_to_depot)` with D5 state, initial auto-delivery, precedence constraints (cell remains traversable), twist-extended states, `describe(state)`. `max_cost` is enforced by search-node path cost, not omitted from constraints. Add hand-made 3×3/5×5 tests with known optimal cost/length, directed/blocked cases, constraints, resources and goal test.
- **Acceptance:** deterministic transitions match `DeliveryEnv` with slip disabled; tests pass.
- **COMMIT:** `feat(search): single-problem formulation with delivered-set state and twist-extended states`

#### M4.2 Search infrastructure

- **Hard dependencies:** M4.1.
- **Tasks:**
  - [x] `search/core.py`: `Node` (`__slots__` where useful), `SearchResult`, `Limits` (expanded nodes, time, stored-record budget, optional path-cost bound), `Metrics`, path reconstruction, status enum and limit exceptions. Enforce D15 positive finite edge costs. Count stored records consistently under D18. Record why a search stopped.
  - [x] `search/estimate.py`: estimate a conservative combinatorial upper bound on state count from passable positions, delivered-mask count and the cardinalities of active state extensions (fuel, budget, time phase, etc.). Formula for a simple delivery problem: `N_passable × 2^k × product(extension_cardinalities)`. Clearly label this as a loose upper bound, not an exact reachable-state count; use it to warn about expensive searches and guide resource budgets.
  - [x] Add a `canonical_optimal_plan` helper used when computing `C*` and `L*`. It must minimise `(total_cost, action_count, action-rank sequence)` exactly, not rely on incidental heap insertion order. Use a lexicographic best-label/priority rule with the positive-cost invariant, or a documented two-pass method. Test equal-cost routes with different lengths and equal-cost/equal-length routes with different action sequences.
- **Acceptance:** tests that each limit triggers its own status; estimator matches hand-calculated upper bounds on tiny fixtures; canonical-plan tests select minimum cost, then minimum action count, then fixed action order; no limit result is misreported as `FAILED` or `INFEASIBLE`.
- **COMMIT:** `feat(search): node, result, metrics and limit infrastructure`

#### M4.3 Uninformed algorithms

- **Hard dependencies:** M4.2.
- **Tasks:**
  - [x] `bfs.py`: graph search, optimal in number of actions, not necessarily cost.
  - [x] `ucs.py`: optimal in accumulated cost with strictly positive costs, lazy-deletion heap and best-g duplicate handling.
  - [x] `dls.py`: iterative explicit-stack DFS; return `CUTOFF` versus `FAILED` distinctly; on-path set prevents cycles.
  - [x] `ids.py`: repeated DLS with path-cycle avoidance, optional stored-record budget and optional `max_depth`.
  - [x] Tests: BFS minimum depth; UCS cost against own exhaustive oracle; DLS cutoff/failure; IDS terminates on cyclic maps and has BFS-minimum depth when it finds a solution; count IDS stack/path bookkeeping under D18. Do not assert a universal physical-memory advantage.
- **Acceptance:** `pytest -q`; understanding checkpoint explains optimality objective and completeness assumptions for each algorithm.
- **COMMIT:** `feat(search): BFS, UCS, DLS and IDS from scratch`

#### M4.4 Informed algorithms and baseline heuristics

- **Hard dependencies:** M1.3, M2.4, M4.2.
- **Tasks:**
  - [x] `heuristics/base.py` (`Heuristic` protocol, `ZeroHeuristic`).
  - [x] `heuristics/baselines.py`:
    - **B1:** `c_min × Manhattan distance` to nearest undelivered stop only when all planned movement edges are ordinary cardinal steps and every transition cost is at least `c_min`. One-way restrictions and blocked cells cannot invalidate Manhattan as a lower bound, but tunnels/additional planning movement can. In a tunnel/extra-edge configuration, use the minimum edge cost times minimum action-count distance in the relaxed graph, or mark B1 unavailable for that configuration; never call the unadjusted Manhattan version admissible there.
    - **B2:** `max_{u∈U} d_R(pos,u)`, where `d_R` is directed shortest-path distance in the relaxed graph. Precompute reverse distance fields from each target as needed. A route visiting every remaining stop must reach the farthest one.
  - [x] `greedy.py`; `astar.py` with closed-node reopening when a better `g` is found, preserving optimality with admissible but inconsistent heuristics under the documented positive-cost assumptions.
  - [x] Tests: A\* admissible-heuristic cost equals UCS on 50 random small scenarios; Greedy returns valid plans but has no optimality guarantee; fixed tie-breaking reproduces results.
- **Acceptance:** `pytest -q`; B1/B2 admissibility conditions and unavailable cases are explicit in output/docs.
- **COMMIT:** `feat(search): greedy best-first and A* with re-opening, baseline heuristics`

#### M4.5 CLI and feasibility certificate

- **Hard dependencies:** M4.4 and approved twist implementation for the final configuration.
- **Tasks:**
  - [x] `python -m delivery.cli plan --algo {bfs,ucs,dls,ids,greedy,astar} --heuristic {zero,b1,b2,own} [--set k=v]` prints stamped summary and writes route figure.
  - [x] Certify group instance solvability with all required twists enabled. Use A\* with a verified admissible heuristic (B2 or own) or UCS. If a complete search exhausts the reachable state space, record `INFEASIBLE`; if a resource limit interrupts it, record `UNKNOWN`, not infeasible. A found valid path is `FEASIBLE`.
  - [x] If twist generation makes an instance infeasible, adjust only documented twist-generator parameters, regenerate `twists.json`, and record the decision; never edit base instance/script.
  - [x] Wire full planner certificate into `scenarios.py` and D8 retry policy.
- **Acceptance:** `pytest -q`; CLI output and route figure valid; certificate has one of the three honest statuses.
- **🛑 A7:** user runs one plan command and inspects route figure.
- **COMMIT:** `feat(cli): plan command and instance feasibility certificate`

### M5 — Heuristics and Part B evaluation (Part B, 2/2) (L)

#### M5.1 🛑 Heuristic design (A5)

- **Hard dependencies:** M4.4 and M2.4.
- **Tasks:** write `docs/heuristic_design.md` with two or three candidates, each specifying formula, proof outline, twist-awareness, expected strength and computational cost. Candidate designs:
  - **Candidate A — relaxed MST:** `h(s)=MST(U∪{p})`, where `U` is undelivered stops and `p` current position. For directed relaxed graphs, use symmetric lower-bound pair weight `w(u,v)=min(d_R(u,v), d_R(v,u))`; calculate the MST on the complete undirected graph of required waypoints. Every feasible route induces a connected spanning walk over these points, and each selected edge weight is no greater than the cost of the corresponding directed route segment; therefore the MST is a lower bound. `h=0` when `U` is empty.
  - **Candidate B — combined lower bound:** `h(s)=max(MST(U∪{p}), max_{u∈U} d_R(p,u))`. Both components are independently admissible, so their maximum is admissible. This may strengthen the MST bound in asymmetric directed configurations.
  - **Optional Candidate C — another resource-aware bound:** propose only if every added term has an independent proof. Do not add a fuel/refuel detour to another lower bound unless a proof shows that the costs are not double-counted. Combining independently admissible bounds with `max` is safe; adding them is not automatically safe.
  - Each twist component must be a relaxation (ignoring a constraint never raises `h`). Fuel/toll restrictions may be ignored for admissibility, at the cost of a weaker heuristic. Time-varying costs use a valid lower-bound multiplier. Tunnels must be represented in relaxed distances.
- **🛑 A5:** user chooses one candidate or requests an edit. Do not implement the own heuristic before choice.
- **Tasks after choice:** `heuristics/own.py` with `prepare()` for required distance fields/subset MSTs; tests on small hand-verifiable cases and every enabled twist.
- **Acceptance:** user choice recorded; formula and assumptions are mathematically coherent; tests pass.
- **COMMIT:** `docs(heuristic): candidate designs` then `feat(heuristic): own admissible heuristic`

#### M5.2 Admissibility proof and consistency checks

- **Hard dependencies:** M5.1.
- **Tasks:**
  - [x] `docs/heuristic_proof.md`: definitions, lemmas for each component, proof of admissibility, proof of each twist relaxation, directed-distance/MST construction, and discussion of consistency.
  - [x] Exhaustively enumerate compact scenarios and compute exact remaining costs by backward Dijkstra over the full reachable state space; assert `h(s) ≤ h*(s)` for all tested states. Clearly handle states with no path to a goal (`h*=∞`).
  - [x] Empirical consistency on 30 scenarios: check `h(s) ≤ c(s, s') + h(s')` over every generated edge visited/generated by the test traversal; report violation count and maximum violation. Consistency is an empirical check unless separately proved.
  - [x] If heuristic is admissible but inconsistent, verify A\* reopening preserves optimality.
- **Acceptance:** exhaustive checks and tests pass; proof reviewed for all active twists.
- **COMMIT:** `test(heuristic): exhaustive admissibility and empirical consistency checks`
- **COMMIT:** `docs(heuristic): admissibility proof`

#### M5.3 Heuristic comparison experiment

- **Hard dependencies:** M5.2.
- **Tasks:** stage `partB_heuristics`: UCS (`h=0`), A*+B1 where valid, A*+B2, A\*+own, Greedy+own on at least 30 paired full-mode scenarios. Report nodes, generated, peak stored records, peak bytes, runtime, path length, cost and DNF/limit counts; mean ± 95% CI; paired own-vs-B2 comparisons on expanded nodes/runtime.
- **Acceptance:** no configuration silently drops DNF; each comparison uses identical scenario seeds.
- **COMMIT:** `feat(experiments): heuristic comparison on 30 scenarios`

#### M5.4 Inadmissible faster variant

- **Hard dependencies:** M5.3.
- **Tasks:**
  - [x] `heuristics/variants.py`: weighted `w·h_own`, `w ∈ {1.2,1.5,2,3}`, and optionally one structurally inadmissible variant.
  - [x] Add a deterministic hand-built `Problem` fixture that proves weighted A* can be faster (fewer expanded nodes) and return a worse plan. Required fixture: `S→A` cost 5, `A→G` cost 10, `S→B` cost 1, `B→G` cost 8; use admissible estimates `h(A)=0`, `h(B)=8`, `h(G)=0`, `h(S)=0`. With `w=2`, weighted A* returns cost 15 after expanding fewer non-goal nodes; ordinary A\*/UCS returns the optimal cost 9. Encode this as a regression test and explain why the weighted priority changes the result.
  - [x] Stage `partB_inadmissible`: also evaluate weighted variants on actual generated delivery scenarios; report whether any produce a faster suboptimal delivery route, cost gap percentage and node/runtime savings. Keep the guaranteed hand-built demonstration separate from observed delivery-scenario results; never imply that the latter occurred if it did not.
  - [x] Render two actual delivery routes side by side if a generated showcase exists. Otherwise render the hand-built fixture's two paths and explicitly label it as a constructed demonstration.
- **Acceptance:** the deterministic fixture test must demonstrate fewer expansions and a strictly worse cost for weighted A*; generated-scenario outcomes are reported honestly, including zero occurrences. The constructed fixture is the guaranteed algorithm-level demonstration; do not claim it is a generated-instance result. Never label weighted A* as admissible for `w>1` without a separate proof.
- **COMMIT:** `feat(experiments): inadmissible heuristic variant gap analysis`

#### M5.5 IDS versus A\* under a stored-node budget

- **Hard dependencies:** M4.3 and M4.4.
- **Tasks:**
  - [x] `search/compact.py`: deterministic compact scenarios (crop/window, retain nearest 2–3 stops) with solution depth ≤ `D_max` (default 14–16).
  - [x] Stage `partB_ids_memory`: sweep stored-record budgets (1k, 5k, 20k, 100k). Seek documented scenarios where A\*/UCS/BFS exceed the record budget while IDS completes within it. Do not guarantee this outcome for every map.
  - [x] Measure extra expansions/runtime and actual peak bytes separately.
  - [x] Ablation: IDS with versus without on-path cycle check; without it should demonstrate repeated-cycle blow-up on a cyclic fixture or reach a resource limit.
  - [x] Curves of peak stored records/bytes and expansions versus depth.
- **Acceptance:** the stored-record rule is identical across algorithms; conclusions do not claim a physical RAM limit from record counts.
- **COMMIT:** `feat(experiments): compare IDS and A* under stored-node budgets`

#### M5.6 Six-algorithm table

- **Hard dependencies:** M5.3 and M5.5.
- **Tasks:** stage `partB_algos`: BFS, UCS, DLS (limit below and at/above optimal depth), IDS (including compact suite), Greedy and A\* on at least 30 full-mode scenarios where feasible, with DNF/limit reporting. Report path length and cost separately and discuss cost-optimality (BFS/IDS minimize moves, not cost).
- **🛑 A7:** user reviews Part B figures and tables.
- **Acceptance:** stamped tables/figures and a coherent interpretation of each algorithm's objective/guarantee.
- **COMMIT:** `feat(experiments): full algorithm comparison table`
- **TAG:** `git tag -a v0.2.0 -m "Part B complete"`

### M6 — Reduced game, Minimax and Alpha-Beta (Part C, 1/2) (L)

#### M6.1 Game model

- **Hard dependencies:** M5.
- **Tasks:**
  - [x] `game/window.py` `WindowGame` per D12, using complete `GameState` in §7.3. Base game actions are N/E/S/W/WAIT; add twist-specific actions when enabled (e.g. deliberate TELEPORT for tunnel endpoints). Implement `legal_actions(state, player)` so it returns only actions legal under terrain, one-way restrictions, resources, turn and active twist rules. Extract square Chebyshev window and document boundary restriction.
  - [x] Terminal tests: capture, theft, completed delivery with κ threshold, excessive-cost draw, step-cap draw. Implement same-cell capture and a reusable edge-swap capture helper; document that a pure alternating one-action-at-a-time model ordinarily resolves same-cell contact immediately, while edge-swap detection is retained for any paired/joint transition and replay semantics required by the brief.
  - [x] Define `C*` and canonical action count `L*` on the same deterministic instance/twist/constraint configuration without rival/slip. Canonical route selection minimises `(total_cost, action_count, action-rank sequence)`; use N,E,S,W, then WAIT, then enabled twist actions in registry order. Use the resulting `L*` in `T_max = 4 × L* + 20`. If the optimal cost cannot be certified, mark the game scenario uncertified and do not invent a threshold.
  - [x] Define terminal utility: `+M` for win, `-M` for loss, `0` for draw, and bounded `f(s)` for non-terminal states, with `M` larger than the maximum absolute non-terminal evaluation and any terminal tie-break. Winning terminal states may use a small cost tie-break that cannot cross into draw/loss utility.
  - [x] Implement the assigned rival's legal-move generator. Additional rival-type generators are optional and are added only if the four-page brief requires multiple types or the user authorises them.
  - [x] `docs/partC_game_model.md`: players, states, actions, utilities, depth in plies, window approximation, strategic A\* next-stop target, D7 event precedence and what time/step counters mean.
  - [x] Tests: legal base and twist-specific moves (including TELEPORT when enabled), blocked/one-way, state completeness, same-cell/swap helper, terminal precedence, κ boundary, canonical C*/L* tie-breaking, time and step-cap accounting.
- **🛑 A6:** present proposed D7 rules, κ, capture, event precedence, `T_max` definition and utilities; get confirmation/changes before freezing the model.
- **Acceptance:** rules confirmed; tests and written formalisation agree.
- **COMMIT:** `feat(game): reduced window game model` and `docs(partC): game formalisation`

#### M6.2 Minimax and Alpha-Beta

- **Hard dependencies:** M6.1.
- **Tasks:** `game/minimax.py` and `game/alphabeta.py` from scratch, depth-limited, injected evaluation/move-ordering callables, `SearchStats` (nodes expanded, evaluations, cutoffs, depth reached), principal variation. Placeholder evaluation uses route progress and rival distance but must respect terminal utility bounds. Tests: Alpha-Beta value equals Minimax value on 200 seeded positions at depths 1–5 and several orderings; textbook tree with known pruned leaves; deterministic tie handling.
- **Acceptance:** `pytest -q`; understanding checkpoint explains alpha/beta bounds and why ordering changes node count but not value.
- **COMMIT:** `feat(game): minimax and alpha-beta from scratch`

#### M6.3 Own move ordering and pruning experiment

- **Hard dependencies:** M6.2.
- **Tasks:**
  - [x] `game/ordering.py`: `natural`, `random(seeded)` and own orderings. Agent ordering: target progress by relaxed distance, then distance from rival/threat, then principal variation from previous depth. Rival ordering is type-specific.
  - [x] Stage `partC_pruning`: at least 200 positions sampled from simulated games over the 30 scenarios; node counts for depths 2…`adversary_search_depth` for Minimax and Alpha-Beta under each ordering; percentage saved with mean ± CI.
  - [x] Written analysis of why own ordering helps or hurts in this game. Random ordering must use recorded seeds.
- **Acceptance:** Alpha-Beta/Minimax values agree for each compared position; no ordering-dependent value changes.
- **COMMIT:** `feat(game): own move ordering and pruning experiment`

### M7 — Rival, evaluation, slip replanning and games (Part C, 2/2) (L)

#### M7.1 Rival agents

- **Hard dependencies:** M6.2 and M6.3.
- **Tasks:** `game/rivals.py`:
  - Implement the rival type actually assigned by the instructor script as the mandatory primary rival, with a documented evaluation and Alpha-Beta search to `adversary_search_depth`.
  - The reference strategies `chaser` (minimise relaxed distance to agent), `blocker` (increase the agent's route cost by occupying a strategically useful cell) and `interceptor` (seek a position favourable relative to the agent's target progress) are optional sensitivity extensions. Implement them only if the four-page brief explicitly requires multiple rival types or the user authorises the extension; otherwise keep them out of the critical path.
  - If the parcel-stealing twist is assigned/enabled, add its terminal behaviour and rival objective.
  - Tests: assigned rival produces legal deterministic moves with fixed ordering and respects blocked/one-way/twist rules. Add strategy-specific tests only for optional rivals that are actually implemented.
- **Acceptance:** assigned rival is complete and tested; any optional rivals are clearly labelled and are not claimed as required work unless the brief says so.
- **COMMIT:** `feat(game): implement assigned rival strategy`

#### M7.2 Hybrid agent

- **Hard dependencies:** M7.1 and M5 own heuristic.
- **Tasks:** `agents/hybrid.py`: strategic A\* route using own heuristic; when rival enters the configured window, tactical Alpha-Beta chooses a move. Count replan triggers separately: slip, rival block, fog discovery, theft/threat and resource change where relevant. Integrate rival into `DeliveryEnv`; capture/termination follows D7.
- **Acceptance:** no-rival, no-slip deterministic run equals A\* plan cost and delivery result; tactical moves replay legally in the simulator.
- **COMMIT:** `feat(agents): hybrid planner-plus-minimax delivery agent`

#### M7.3 Evaluation function

- **Hard dependencies:** M7.2.
- **Tasks:**
  - [x] `game/evaluation.py`: at least four normalised features with documented ranges/weights, e.g. (1) remaining route cost via own heuristic, (2) delivered fraction, (3) saturated rival distance/threat, (4) rival threat to next-leg cells, (5) escape mobility, (6) twist-specific resource/time margin.
  - [x] Features must be scaled so no ordinary evaluation can exceed terminal utility bounds. Define signs and monotonic expectations carefully; “farther rival is better” is conditional on route/capture feasibility, not a universal rule if the rival is blocking an exit.
  - [x] `docs/eval_design.md`: justification, sign, scaling, twist interaction, why it differs from textbook evaluation.
  - [x] Tests for stated monotonic properties on controlled states, not arbitrary states where features legitimately interact.
- **🛑 A7 (optional):** user reviews feature list and rationale.
- **Acceptance:** features bounded/normalised; tests and docs agree.
- **COMMIT:** `feat(game): own evaluation function` and `docs(game): evaluation design`

#### M7.4 Slip and replanning experiments

- **Hard dependencies:** M7.2 and M7.3.
- **Tasks:**
  - [x] Stage `partC_slip`: sweep `{0, 0.1, 0.2, 0.3, 0.5, instance value}` on the same paired seeds.
  - [x] Metrics: slip attempts, actual displacements, replan triggers (count and per 100 agent moves), extra cost paired against slip=0, capture rate, predicted versus realised delivery cost, deterministic predicted outcomes versus realised outcomes.
  - [x] Analyse slips into costly terrain/toward rival, mismatch between deterministic rival replies and realised play, replanning effectiveness and behaviour at 0.5.
  - [x] Figures/tables with mean ± CI or Wilson CI as appropriate.
- **Acceptance:** same scenario seeds across slip probabilities; distinguish model predictions from stochastic outcomes.
- **COMMIT:** `feat(experiments): slip probability and replanning analysis`

#### M7.5 Weight tuning

- **Hard dependencies:** M7.3 and M7.4.
- **Tasks:**
  - [x] `game/tuning.py`: predeclared small grid or seeded random search over weights. Avoid a combinatorial 3^5 full game sweep unless profiling proves practical; set a fixed tuning budget (e.g. capped candidate count and a smaller fixed tuning-game set).
  - [x] Tuning seeds are disjoint from frozen evaluation seeds and never inspected during tuning. Cache by full configuration/seed/source hash.
  - [x] Tie-break by mean cost after the primary outcome criterion; write `config/tuned_weights.json` and raw tuning results; heatmap/table.
  - [x] Optional branch `exp/eval-tuning`.
- **Acceptance:** chosen weights are fixed before evaluation-set run; no evaluation leakage.
- **COMMIT:** `feat(tuning): tune evaluation weights by game experiments`

#### M7.6 Final games

- **Hard dependencies:** M7.5.
- **Tasks:** stage `partC_games`: `n_games` default 50, never below 30 in full mode, against assigned rival on evaluation seeds; W/D/L with Wilson 95% CI; tuned versus default weights; mean cost and replan metrics. Save replay strips (PNG sequence) of one actual win and one actual loss, if available; if one outcome does not occur, select a representative alternative and state why.
- **🛑 A7:** user watches replays and reports whether behaviour looks sensible.
- **Acceptance:** full-mode count and seed split verified; no invented replay/outcome.
- **COMMIT:** `feat(experiments): 50-game evaluation against the rival`
- **TAG:** `git tag -a v0.3.0 -m "Part C complete"`

### M8 — LLM dispatcher (Part D) (L)

#### M8.1 🛑 Backend and client (A4)

- **Tasks:**
  - [ ] Ask backend choice (hosted API or local OpenAI-compatible server), who creates the key and explicit quota/billing expectations. User creates key locally in `.env`; agent may read it through environment configuration only as needed, but must never display/log/copy it.
  - [ ] `llm/client.py`: `LLMClient` protocol, provider adapter, `ReplayClient` (reads sanitized `results/llm_cache.jsonl`), `MockClient` for tests. Cache key includes provider/model, prompt/messages, tools/schema and generation settings. Support explicit cache bypass for independent live repetitions.
  - [ ] Record provider/model/version where available; temperature 0 is a setting, not a determinism guarantee.
- **Acceptance:** mocked and replay calls work offline; secret scan passes.
- **COMMIT:** `feat(llm): provider-agnostic client with record-replay cache`

#### M8.2 Tools

- **Hard dependencies:** M8.1, M4.5 and M3.1.
- **Tasks:** `llm/tools.py` with JSON schemas and deterministic implementations:
  - `plan_route(algorithm, heuristic?, constraints)` calls Part B planners through `Constraints`.
  - `simulate(plan | steps)` simulates through `DeliveryEnv` and returns structured outcome.
  - Custom tool tied to an assigned twist (Appendix A), plus any supporting twist tools required by the brief.
  - Results include status, cost, route summary, constraints satisfied/violated, and reason codes. Tests use `MockClient` scripted calls.
- **Acceptance:** tool output is deterministic for identical inputs; invalid calls are rejected before side effects.
- **COMMIT:** `feat(llm): plan_route, simulate and twist-specific tools`

#### M8.3 ReAct loop

- **Hard dependencies:** M8.2.
- **Tasks:** `llm/react.py`: system prompt with instance facts (stop IDs/coordinates, terrain, twist rules, tools), strict structured Reason/Action/Observation cycle, robust parser and bounded steps. Ask for concise rationale summaries, not hidden chain-of-thought. Trace logger writes JSONL and rendered text to `results/traces/` with stamp and sanitized content.
- **Acceptance:** parser handles malformed/incomplete tool calls and terminates safely at the step budget.
- **COMMIT:** `feat(llm): ReAct dispatcher loop with trace logging`

#### M8.4 Requests and checker

- **Hard dependencies:** M8.3.
- **Tasks:**
  - [ ] `data/requests.yaml`: 20 natural-language requests across five difficulty levels (single constraint; multiple constraints; twist-specific; ambiguous; infeasible/conflicting), each with a structured ground-truth specification and expected clarification/refusal where appropriate.
  - [ ] `llm/checker.py`: success check (plan valid in simulator, constraints satisfied, cost versus direct planner optimum; infeasible request succeeds only if agent correctly identifies infeasibility/uncertainty according to available evidence).
  - [ ] Direct-planner baseline: spec → planner, one planning call.
- **Acceptance:** all 20 requests have independent ground-truth specs and automated checks.
- **COMMIT:** `feat(llm): 20 dispatcher requests with automatic checker`

#### M8.5 Validation layer

- **Hard dependencies:** M8.4.
- **Tasks:** `llm/validation.py`: schema and semantic checks against instance (algorithm exists, stop IDs exist, terrain names valid, cells in grid/passable, numeric ranges, fuel ≤ capacity, budget ≥0, contradictory constraints). Return machine-readable errors for at most two self-repair attempts; count interventions. Tests include deliberately invalid arguments and attempts to bypass constraints.
- **Acceptance:** invalid tool calls cannot reach simulator/planner side effects; intervention counts are accurate.
- **COMMIT:** `feat(llm): argument validation layer with self-repair feedback`

#### M8.6 Evaluation, failures and traces

- **Hard dependencies:** M8.5.
- **Tasks:**
  - [ ] Stage `partD`: evaluate the 20 requests **with and without validation**, paired on the same recorded initial model completions where possible. Default `R=3` independent live sessions per request only if quota is approved; live repetitions explicitly bypass replay cache and are stored separately. Replay runs are for reproducibility/debugging and are never counted as independent live samples. Analyse repeated runs as observations nested within request IDs (request-level/cluster-aware uncertainty), not as 60 independent requests.
  - [ ] Compare success, intent/tool/argument accuracy, constraint satisfaction, plan validity, cost ratio, tool calls, validation interventions and latency versus direct planner.
  - [ ] Failure log records each wrong-tool, invented-argument, ignored-constraint, invalid-plan and unrecognised-infeasibility case with request ID, tool call, evidence and root cause. Automatic classification is reviewed manually.
  - [ ] Select two traces (success and failure or validation rescue) and write `docs/sra_mapping.md` mapping Sense, concise Reason summary, Action/tool call and Observation to Sense–Reason–Act (Lab 3). Do not claim the log is a full internal reasoning transcript.
  - [ ] Explain model/provider version, prompt version, cache status and remaining nondeterminism.
- **🛑 A4/A7:** first live run requires user's key and explicit consent to spend quota. Ask whether user runs `python run_all.py --only partD --llm-mode live` or authorises the agent to run it. User reviews failure labels.
- **Acceptance:** results distinguish live independent calls from cache replays; failures are not hidden; comparisons use paired request units.
- **COMMIT:** `feat(experiments): LLM dispatcher versus direct planner with and without validation`
- **TAG:** `git tag -a v0.4.0 -m "Part D complete"`

### M9 — Ablation, failure analysis and consolidation (M)

#### M9.1 Ablation

- **Hard dependencies:** M5.3 and M5.2.
- **Tasks:** stage `ablation` with at least four configurations: (1) baseline heuristic, (2) baseline + MST component, (3) baseline + twist-aware component if distinct, (4) full heuristic. Use identical paired scenarios and report nodes, runtime, cost/optimality and memory. Optional second ablation of evaluation features. If the selected heuristic has a different decomposition, define equivalent meaningful components before running.
- **Acceptance:** component definitions are explicit and no ablation removes a term while accidentally changing unrelated configuration.
- **COMMIT:** `feat(experiments): ablation study of heuristic components`

#### M9.2 Failure analysis

- **Hard dependencies:** all experiment stages intended for the report.
- **Tasks:** `experiments/failure_analysis.py` aggregates planner DNF/timeouts, infeasible/unknown scenarios, game losses with reason codes, LLM failures and UI/runtime failures into stamped `results/tables/failure_analysis.csv` (category, count, example seed/request, root cause, mitigation). `docs/failure_analysis.md` explains causes and limitations.
- **Acceptance:** every failure count can be traced to raw/manifest records; missing categories are labelled, not assumed zero.
- **COMMIT:** `feat(experiments): cross-stage failure analysis`

#### M9.3 Numbers and contract checks

- **Hard dependencies:** M9.1 and M9.2.
- **Tasks:** `experiments/numbers.py` writes `results/numbers.json` (stable keys; separate `timing`, `quick`, `full` and completeness metadata) and `report/generated_numbers.tex` (`\newcommand` per report number). `tests/test_experiment_contract.py`: full mode has N≥30 where required, tables include CI columns, outputs are stamped, only full-mode data feeds report macros, provenance links exist.
- **Acceptance:** `pytest -q`; regenerate numbers and macros without manual editing.
- **COMMIT:** `feat(experiments): numbers export and experiment contract tests`

### M10 — UI (Streamlit) (M)

#### M10.1 Layout

- **Hard dependencies:** M7, M8 (for tabs), M2.6 (overrides).
- **Tasks:** `ui/app.py`: stamp on every tab/screen; sidebar controls for algorithm, heuristic, DLS limit, slip probability, stop/start editing (validated through `overrides.py`), twist on/off, registered rival strategy/depth, window radius and seed. Do not expose unimplemented rival choices; if only the assigned strategy is required, show that strategy and its depth without a misleading selector. Main panes: map (route, agent, rival, twist layers), metrics, route/plan table and event log. State in `st.session_state`; planning cache keyed by canonical override hash and source/config.
- **Acceptance:** UI starts; controls are validated; stamp is visible on every tab.
- **COMMIT:** `feat(ui): Streamlit layout with stamp, controls and map`

#### M10.2 Planning and execution

- **Hard dependencies:** M10.1.
- **Tasks:** Plan action runs chosen algorithm within limits and shows nodes/cost/path length/time/status/route. Step/Play/Reset shows slips, rival moves and replans in event log; redraw through a placeholder. Show DNF/limit statuses honestly; never display stale metrics after an override.
- **Acceptance:** parity tests against CLI for the same override/configuration.
- **COMMIT:** `feat(ui): planning and step-by-step execution`

#### M10.3 Rival and dispatcher tabs

- **Hard dependencies:** M10.2.
- **Tasks:** game panel (play a game, show W/D/L and replans); dispatcher tab (chat input, live/replay mode, trace display, validation toggle, visible quota/provider status without secrets).
- **Acceptance:** game and dispatcher actions use the same validated backend logic as CLI.
- **COMMIT:** `feat(ui): rival game panel and LLM dispatcher tab`

#### M10.4 Live-change flow and manual test

- **Hard dependencies:** M10.3.
- **Tasks:** Apply overrides and rerun within the target few seconds for supported lightweight changes; parity with `python -m delivery.cli rerun --set …`. `docs/ui_manual_test.md` checklist: names/IDs on every tab, each algorithm, slip change, moving stop, start change, twist toggle, rival-depth change (and rival-type change only if multiple strategies are delivered), LLM tab, validation and error handling. Screenshot checklist for report; screenshots must carry stamp.
- **🛑 A7:** user runs `streamlit run delivery/ui/app.py`, completes checklist and provides screenshots/issues; report screenshots go to `report/figures/ui_*.png`.
- **Acceptance:** documented manual checklist completed; cache invalidation proven.
- **COMMIT:** `feat(ui): live parameter changes and manual test checklist`
- **TAG:** `git tag -a v0.5.0 -m "UI complete"`

### M11 — Integration, reproducibility and live-change readiness (M)

#### M11.1 Complete `run_all.py`

- **Hard dependencies:** M9.3 and all required experiment stages.
- **Tasks:**
  - [ ] Register stages in dependency order (`partA_*`, `partB_*`, `partC_*`, `partD`, `ablation`, `failure`, `numbers`).
  - [ ] `--quick`: 3 scenarios, 4 games, compact budgets, isolated quick outputs; target <5 minutes. Full mode uses required N and writes official outputs.
  - [ ] Cache/`--fresh`/workers support; serial default until profiling justifies parallelism.
  - [ ] A failed independent stage does not kill unrelated stages; dependent stages are skipped with reason; report generation/release fails if mandatory full results are missing.
  - [ ] `results/MANIFEST.json` records stage statuses, dependencies, duration, run mode, configuration, seeds, versions, commit and artifact hashes.
- **Acceptance:** quick run exercises every registered pipeline component in miniature; failure injection correctly skips dependent stages.
- **COMMIT:** `feat(experiments): complete one-command reproduction pipeline`

#### M11.2 Clean-clone and determinism

- **Hard dependencies:** M11.1.
- **Tasks:** pin all top-level package versions and Python version in `requirements.txt`; retain platform notes. Slow test clones/copies the repo to a temporary directory, creates a venv, installs dependencies, runs `python run_all.py --quick` twice, and compares canonical non-timing outputs. Exclude timestamps, timing and explicitly machine-specific manifest fields. Ensure test does not require access to `.env` or live LLM.
- **🛑 A7:** user runs full `python run_all.py` once and reports duration/errors.
- **Acceptance:** two quick clean runs reproduce non-timing outputs; full run finishes or provides a complete honest failure manifest.
- **COMMIT:** `build: pin requirements and add reproducibility test`

#### M11.3 Live-change drills

- **Hard dependencies:** M11.1 and M10.4.
- **Tasks:** time each live-change drill (target ≤5 minutes end-to-end including rerun): slip probability, stop, start, assigned twist toggle, rival depth, DLS limit, and rival-type switching only when multiple strategies are required/implemented. Verify cache invalidation. Write `docs/viva_live_changes.md` with exact commands, expected outputs and measured times; list only controls that exist in the delivered UI.
- **Acceptance:** each drill is reproducible and documented; target misses are reported, not concealed.
- **COMMIT:** `docs: viva live-change drills and timings`

#### M11.4 🛑 Optional: remaining twists as plug-ins (A9)

- **Hard dependency:** user says yes; all required twists and report data are stable.
- **Tasks:** implement remaining twists from Appendix A as lightweight plug-ins (base dynamics, appropriate capability protocols, relaxed edges, render layer, tests) on `exp/extra-twists`. This is optional and never blocks required submission.
- **Acceptance:** each additional twist has tests and is not claimed as an assigned requirement unless it is.
- **COMMIT (per twist):** `feat(twists): add <twist> plug-in`

#### M11.5 README

- **Hard dependencies:** M11.1 and M10.4.
- **Tasks:** overview, install, quickstart, one-command reproduction, structure, UI usage, instance regeneration from real CMS IDs, D14 brief/script discrepancy and actual generated stop count, AI-assistance disclosure, ethics pointer, known limitations and which outputs are quick versus full.
- **Acceptance:** a clean clone can follow README without undocumented manual steps.
- **COMMIT:** `docs: finalise README`
- **TAG:** `git tag -a v0.9.0-rc.1 -m "Release candidate"`

### M12 — Report (IEEE, 8–10 pages) (M–L)

#### M12.1 🛑 Setup (A8)

- **Tasks:** ask LaTeX/IEEEtran (recommended; Overleaf acceptable if no local TeX install) or Word. For LaTeX create `report/main.tex`, `references.bib`, import `stamp.tex` and `generated_numbers.tex`, create section skeleton and page budget. Proposed budget totals **9.5 pages**: abstract/introduction/instance/twists 1.1; Part A 0.9; Part B 2.3; Part C 1.9; Part D 1.4; ablation/failure/UI 0.9; ethics/conclusion/references 1.0. Confirm whether references/appendices count toward the instructor's page limit. Every figure/table caption uses `\stampline`.
- **Acceptance:** report toolchain selected and skeleton builds.
- **COMMIT:** `docs(report): IEEE report skeleton with generated numbers`

#### M12.2 Parts A and B

- **Hard dependencies:** M3.3 and M5.6.
- **Tasks:** instance/twists, PEAS/classification, benchmark, search formulation, algorithms, own heuristic and proof (condensed), comparisons, inadmissible gap and IDS stored-node study. Numbers only via generated macros.
- **Acceptance:** all claims match full-mode results and documented assumptions.
- **COMMIT:** `docs(report): instance, Part A and Part B sections`

#### M12.3 Parts C, D and experiments

- **Hard dependencies:** M7.6, M8.6, M9.3 and M10.4.
- **Tasks:** game model, pruning, slip analysis, evaluation/tuning, games, LLM dispatcher, validation, traces mapped to Sense–Reason–Act, ablation, failure analysis and UI screenshots.
- **Acceptance:** all Part C/D claims supported by traceable results; distinguish model assumptions and observed stochastic outcomes.
- **COMMIT:** `docs(report): Part C, Part D and experiments sections`

#### M12.4 Ethics, reflection and references

- **Hard dependencies:** M12.2 and M12.3.
- **Tasks:** half-page ethics/reflection: affected parties (recipients, other road users, couriers, operator), concrete failure scenarios (wrong parcel, invented arguments, ignored constraint, poor rival model), safeguards actually implemented (validation, deterministic fallback planner, resource caps, logging, replay audit) and remaining limitations. Add AI-assistance statement, conclusion and references (Russell & Norvig; Hart–Nilsson–Raphael A\*; Korf iterative deepening; Knuth–Moore alpha-beta; ReAct paper). Verify every citation against the actual source; never fabricate bibliographic details.
- **Acceptance:** ethics describes implemented safeguards and does not claim safeguards that do not exist.
- **COMMIT:** `docs(report): ethics, reflection, conclusion and references`

#### M12.5 Build and check

- **Hard dependencies:** M12.4.
- **Tasks:** compile PDF; meet 8–10-page limit under instructor counting rules; all captions stamped; every number comes from a macro and matches `numbers.json` (script `report/check_numbers.py`); IEEE two-column formatting; spell-check; verify figures and citations.
- **🛑 A7:** user reviews PDF.
- **Acceptance:** PDF builds cleanly and all numbers/captions/limits are verified.
- **COMMIT:** `docs(report): final report PDF`

### M13 — Viva preparation and release (S–M)

#### M13.1 Code walkthrough

- **Hard dependencies:** M12.5.
- **Tasks:** `docs/code_walkthrough.md`: module purpose, key functions, invariants, complexity, important design choices and line-level explanation of non-trivial blocks. The user must be able to explain every line; walkthrough should focus teaching attention on meaningful logic rather than generate empty prose for boilerplate.
- **Acceptance:** walkthrough checked against current source commit.
- **COMMIT:** `docs: code walkthrough for viva`

#### M13.2 Viva Q&A

- **Hard dependencies:** M13.1.
- **Tasks:** `docs/viva_qa.md`: at least 40 questions with answers tied to this instance/results: heuristic admissibility, slip=0.5, move ordering, LLM failures, twist integration, vectorisation, IDS versus A\* stored-node budgets, BFS/UCS/IDS guarantees, uniqueness, reproducibility, terminal utility, state completeness and limitations.
- **Acceptance:** answers cite actual generated values and explain assumptions.
- **COMMIT:** `docs: viva question bank`

#### M13.3 🛑 Oral quiz

- **Hard dependencies:** M13.2.
- **Tasks:** quiz user module by module (about ten questions per module, one at a time); record weak areas in git-ignored `docs/viva_weak_spots.md` and suggest targeted rereading. Do not commit personal study notes unless requested.
- **Acceptance:** user has practised each major assessed component.
- **No commit.**

#### M13.4 Release

- **Hard dependencies:** all required milestones and instructor clarifications.
- **Tasks:** regenerate canonical instance with `python oel_instance.py <real CMS IDs>` and compare with committed instance; `git status` clean; `pytest -q`, `pytest -m slow`, `python run_all.py --quick`; verify full-mode results and manifest; submission bundle (`git archive` zip, report PDF, UI demo instructions/recording as instructed); verify names/IDs in README/UI/report/results; verify no secrets and no quick-mode results represented as official.
- **COMMIT:** `chore(release): prepare v1.0.0`
- **TAG:** `git tag -a v1.0.0 -m "OEL submission"`

---

## 9. Appendix A — Twist specifications

For each twist: **Generation** (deterministic, feasibility-safe) · **Effect** (state/cost/transition) · **Heuristic relaxation** · **Simulator** · **LLM tool** · **Tests**. Implement only the two assigned twists plus the approved third unless A9 = yes. The exact semantics are frozen in `docs/transition_model.md` before integration.

### 1. `fuel_limit_with_charging_stations`

- **Generation:** capacity `F`; stations include depot and cells placed along feasible shortest paths from depot to stops at intervals of approximately `F/2` cost units, plus deterministic extras. Calibrate `F` so the feasibility certificate passes; generation must not guarantee feasibility merely by assuming a path.
- **Effect:** state adds `fuel`; each courier move burns fuel equal to the entered-cell movement cost; entering a station refuels to `F` after paying the move cost; a move that would make fuel negative is illegal. Define WAIT/fallback fuel use explicitly (initially the configured WAIT/fallback cost).
- **Relaxation:** ignore fuel restriction and refuelling requirement. This preserves all actual legal movement possibilities at no greater cost, so it is admissible but weaker. Do not add an unproved refuelling-detour term.
- **Simulator:** realised slips and waits consume fuel under the same resource rules; no negative fuel.
- **LLM tool:** `find_charging_station(from_cell)` and/or `fuel_status()`.
- **Tests:** illegal move at zero/insufficient fuel; refuel reset; route never strands under certified conditions; search state includes fuel; planner/simulator parity.

### 2. `one_way_streets`

- **Generation:** orient a deterministic fraction (about 25%) of adjacent-cell edges; repair by removing orientations until start and all stops are in one strongly connected component. Verify using own graph algorithm; retain generation seed.
- **Effect:** directed transition mask `allowed[n²,4]`; no extra state.
- **Relaxation:** preserve directed edges; do not relax by accidentally making prohibited directions legal unless that relaxation is explicitly used and remains a superset. Reverse-Dijkstra fields support distances from positions to fixed targets.
- **Simulator:** slips respect directionality.
- **LLM tool:** `check_street_direction(cell, direction)`.
- **Tests:** strong connectivity of required cells, directed legality, reverse-distance correctness, relaxed distance no greater than feasible true path cost.

### 3. `rush_hour_time_varying_costs`

- **Generation:** period `P` (e.g. 24 ticks) and integer multiplier profile (×1, ×2, ×3 peaks) on selected terrain/road cells.
- **Effect:** state adds `t mod P`; each player/courier action advances one global tick in the game model; in single-agent simulator each action advances one tick. Cost is evaluated at the documented arrival/departure phase (choose and freeze in `transition_model.md`; default: arrival phase). WAIT advances time and pays its configured positive waiting cost.
- **Relaxation:** use the minimum multiplier across the entire period (at least 1) for lower-bound edge costs.
- **Simulator:** time advances exactly once per action, including WAIT and a realised slip; document how a fallback stay advances time.
- **LLM tool:** `get_cost_profile(t)` / `best_departure_window()`.
- **Tests:** phase wraps modulo P; time advances consistently; waiting can beat a peak route in a constructed example; planner/simulator costs agree at each phase.

### 4. `fog_partial_observability`

- **Generation:** percept radius ρ (3–5); stop locations known, map cells beyond ρ unknown.
- **Effect:** actual map remains in simulator; agent maintains a separate belief/known-map representation. For each planning episode, unknown cells are optimistically assumed passable with minimum terrain cost (or another documented optimistic prior); when cells are revealed, update belief and replan. Belief is problem context for that planning episode, not confused with the actual world. If beliefs change inside an active search, that search is invalidated/restarted rather than continuing with stale assumptions.
- **Relaxation:** optimistic unknown-cell assumption makes the known-map planning estimate no more costly than the actual route, subject to the stated unknown-cell model.
- **Simulator:** `percept()` returns only cells within ρ; discovery events trigger replanning.
- **LLM tool:** `scan_area(radius)`.
- **Tests:** percept masks; belief/actual map separation; replan after discovering a block or higher cost; optimistic lower-bound property.

### 5. `toll_gates_budget`

- **Generation:** gates on sparse cells, never start/stops; select so a toll-free connection among key cells exists (own reachability check); fee 3–10; budget B is calibrated from fees and certified route needs.
- **Effect:** state adds `budget_left`; entering a gate deducts fee and is illegal if insufficient; objective remains travel cost, budget is a feasibility resource. Gate fee is not silently added to travel cost unless the instructor's twist definition explicitly says otherwise.
- **Relaxation:** ignore toll restriction/fee resource; retain no-more-expensive movement cost.
- **Simulator:** slips onto gates charge the fee.
- **LLM tool:** `estimate_tolls(route)` / `budget_status()`.
- **Tests:** toll-free route exists; budget never negative; state includes budget; deterministic planner/simulator agree.

### 6. `teleport_tunnels`

- **Generation:** 2–4 endpoint pairs on passable cells; teleport action cost 1–2 and strictly positive.
- **Effect:** from an endpoint, an additional deliberate action moves to partner. Tunnel edge is part of deterministic planner/game transitions.
- **Relaxation:** include tunnel edges. Unadjusted Manhattan B1 is not used/admissible in this configuration; use tunnel-aware relaxed action-count distance or mark baseline unavailable.
- **Simulator:** ordinary slips can land on a tunnel cell but do not automatically teleport; teleport is deliberate.
- **LLM tool:** `list_tunnels()`.
- **Tests:** symmetric pairs where specified; tunnel shortens a constructed example; relaxed distances and B1 eligibility validated.

### 7. `wind_drift`

- **Generation:** fixed wind direction/strength `p_w`, or direction changes every K ticks (then state includes `t mod K`).
- **Effect:** deterministic against-wind surcharge and with-wind discount are applied in planner and simulator, while all transition costs remain strictly positive. Probabilistic one-cell wind push is simulator-only under D6 and is not assumed by deterministic planners/Minimax. If a later design puts drift into planning, update the relaxed graph and re-prove heuristics first.
- **Relaxation:** base costs/minimum legal cost; stochastic push is excluded from deterministic planning objective and reported as model mismatch.
- **Simulator:** with probability `p_w`, an extra push in wind direction occurs if legal; define whether the push is a second transition/tick and how cost/resource updates apply. Default: it is a realised extra displacement within the same attempted action, charges the extra cell's cost, and does not add an extra agent-step count; global tick increments once.
- **LLM tool:** `get_wind_forecast()`.
- **Tests:** push never enters blocked/forbidden edge; surcharge/discount applied; drift frequency on eligible moves; positive costs preserved.

### 8. `rival_steals_parcels`

- **Generation:** deterministic theft behaviour/probability parameters from twist seed.
- **Effect:** game/simulator rival reaching an undelivered stop before the agent triggers immediate terminal loss. The agent's single-agent Part B planners do not model rival theft; this limitation is explicit. Rival evaluation gains a distance-to-undelivered-stop feature.
- **Relaxation:** none needed for single-agent cost heuristic if the rival is excluded; for game state, theft is a terminal rule, not a route edge.
- **LLM tool:** `check_rival_threat(stop)`.
- **Tests:** theft terminal rule and event precedence; rival targets undelivered stops; unaffected when twist off.

---

## 10. Appendix B — Third-twist candidates (user chooses; instructor approves; must be unique)

1. **Fragile parcel (recommended low-risk candidate):** while the fragile parcel is undelivered, mud/ford cells are illegal. The restriction depends on delivered mask and changes the transition model. The selected fragile-parcel destination must be on road/grass, or its destination cell must be explicitly exempted, so the twist does not make delivery impossible. Relaxation ignores the restriction. No additional state variable is required beyond the delivered mask.
2. **Load-dependent travel cost:** cost multiplier grows with the number of undelivered parcels still carried (derived from `mask`; no additional state). Relaxation uses the minimum possible multiplier (e.g. after all possible deliveries) and must prove it is a lower bound.
3. **Priority parcels with deadlines:** lateness penalty per tick past deadline (state includes time or deadline-relevant time); define whether penalty is objective cost or terminal outcome and preserve positive transition costs.
4. **Limited capacity:** agent carries at most C parcels and must return to depot to reload; state must distinguish carried, delivered and uncollected parcels. More complex than the others.
5. **Scheduled road works:** cells close/reopen on schedule (state adds `t mod P`); avoid unintended duplication with rush-hour twist and define waiting/time rules.
6. **Traffic lights:** marked intersections add a waiting cost depending on `t mod 4`; define whether light cycles advance on every action and whether WAIT is compulsory/optional.

The agent drafts the one-paragraph proposal; the user submits it and reports instructor approval. Do not implement an unapproved twist.

---

## 11. Appendix C — Progress tracker (tick in the commit that completes the item)

- [ ] M0.0 Source-document preflight (tick with M0.1 initial commit)
- [ ] M0.1 Repository and tooling
- [ ] M0.2 Inputs collected (A0, A1, A2)
- [ ] M0.3 Stamp infrastructure
- [ ] M0.4 Instance generation/loading; NumPy pinned; internal scenario-generation path verified
- [ ] M1.1 Grid world
- [ ] M1.2 Loop reference and benchmark
- [ ] M1.3 Distance fields and renderer
- [x] M2.1 Twist framework
- [x] M2.2 Assigned twist A
- [x] M2.3 Assigned twist B
- [x] M2.4 Twist interplay and relaxed-graph contract tests
- [x] M2.5 Third twist (requires instructor approval)
- [x] M2.6 Overrides, scenarios, statistics, cache
- [x] M3.1 Stochastic environment and planner/simulator parity
- [x] M3.2 Runner and reflex baseline
- [x] M3.3 PEAS and classification — **v0.1.0** (blocked until required twist approval is resolved)
- [x] M4.1 Search problem
- [x] M4.2 Search infrastructure and resource accounting
- [x] M4.3 BFS, UCS, DLS, IDS
- [x] M4.4 Greedy, A\*, baseline heuristics
- [x] M4.5 CLI and honest feasibility certificate
- [x] M5.1 Own heuristic design/choice
- [x] M5.2 Proof and consistency/exhaustive checks
- [x] M5.3 Heuristic comparison
- [x] M5.4 Inadmissible variant gap analysis
- [x] M5.5 IDS vs A\* under stored-record budget
- [x] M5.6 Six-algorithm table — **v0.2.0**
- [x] M6.1 Game model and A6 rules confirmation
- [x] M6.2 Minimax and Alpha-Beta
- [x] M6.3 Move ordering and pruning experiment
- [x] M7.1 Assigned rival strategy
- [x] M7.2 Hybrid agent
- [x] M7.3 Evaluation function
- [x] M7.4 Slip and replanning analysis
- [x] M7.5 Weight tuning on disjoint seeds
- [x] M7.6 Final games — **v0.3.0**
- [x] M8.1 LLM client/backend
- [x] M8.2 Tools
- [x] M8.3 ReAct loop and safe traces
- [x] M8.4 Requests and checker
- [x] M8.5 Validation layer
- [x] M8.6 Evaluation/failures/traces — **v0.4.0**
- [x] M9.1 Ablation
- [x] M9.2 Failure analysis
- [x] M9.3 Numbers export and contract checks
- [x] M10.1 UI layout
- [x] M10.2 Planning and execution
- [ ] M10.3 Rival and dispatcher tabs
- [ ] M10.4 Live-change flow and manual test — **v0.5.0**
- [ ] M11.1 Complete run_all
- [ ] M11.2 Clean clone and determinism
- [ ] M11.3 Live-change drills
- [ ] M11.4 Optional remaining twists
- [ ] M11.5 README — **v0.9.0-rc.1**
- [ ] M12.1 Report setup
- [ ] M12.2 Parts A and B
- [ ] M12.3 Parts C, D and experiments
- [ ] M12.4 Ethics, references
- [ ] M12.5 Build and check
- [ ] M13.1 Code walkthrough
- [ ] M13.2 Viva Q&A
- [ ] M13.3 Oral quiz
- [ ] M13.4 Release — **v1.0.0**

---

## 12. Appendix D — Risks and mitigations

| Risk                                                          | Mitigation                                                                                                                                                                                                        |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NumPy version changes script random stream                    | Pin NumPy before canonical instance generation; record versions; golden test; compare with instructor output/environment                                                                                          |
| Brief says 4–6 stops but script gives 3–6                     | Implement the script's full 3–6 range; disclose the actual count and discrepancy; do not fabricate a stop or block implementation waiting for an instructor response                                              |
| Script rejects synthetic scenario token                       | Use an existing script seed interface or documented local deterministic scenario derivation; tokens remain internal to experiments; never edit script, impersonate a member or ask the instructor to enter tokens |
| Pure-Python search too slow                                   | Flat positions, precomputed neighbours, positive costs, strong proven heuristic, honest limits/DNF; profile before multiprocessing                                                                                |
| Twist data makes instance/scenario infeasible                 | Feasibility certificate with FEASIBLE/INFEASIBLE/UNKNOWN; calibrated deterministic generation; bounded retries; never confuse timeout with infeasibility                                                          |
| Third twist approval delayed                                  | Isolate M2.5; continue unrelated work; mark Part A final acceptance blocked until resolved                                                                                                                        |
| Heuristic proof invalid under directed/tunnel/resource twists | Define relaxed-edge contract; use directed distances and symmetric lower-bound MST edges; exhaustive small-state checks; no unproved additive terms                                                               |
| Planner and simulator disagree                                | Canonical transition spec and cross-component replay/cost tests                                                                                                                                                   |
| Game state omits twist/cost/time variable                     | Complete `GameState` contract; tests assert state changes whenever future dynamics/terminal status changes                                                                                                        |
| Minimax deterministic model differs from stochastic play      | Explicitly evaluate predicted versus realised costs/capture and replanning benefit                                                                                                                                |
| LLM non-determinism, quota or cost                            | Temperature 0 as default only; provider/model/prompt records; cache/replay; user consent; independent live-run accounting; no secrets in traces                                                                   |
| Evaluation overfits                                           | Freeze disjoint tuning/evaluation seeds before tuning; no evaluation inspection during selection                                                                                                                  |
| Quick results contaminate report                              | Separate quick/full storage and metadata; report macros accept full-only data                                                                                                                                     |
| Game tuning too slow                                          | Fixed candidate budget, smaller tuning suite, profile and cache; keep evaluation seeds disjoint                                                                                                                   |
| Extra rival strategies expand scope without rubric value      | Implement the assigned rival first; add alternatives only when the four-page brief or user explicitly requires them                                                                                               |
| Windows multiprocessing complexity                            | Serial first; add spawn-safe pool only after measured need                                                                                                                                                        |
| AI-written code user cannot explain at viva                   | Understanding checkpoint each algorithmic milestone; M13 walkthrough/Q&A/oral quiz                                                                                                                                |
| Timing differs across machines                                | Separate timing block; record hardware/OS; compare non-timing canonical outputs                                                                                                                                   |
| Report exceeds page limit                                     | 9.5-page planned budget; confirm reference counting; check continuously, not only at final build                                                                                                                  |
| Mandatory stage fails but report still appears complete       | Dependency-aware stages; skip dependants; fail release validation on missing full results                                                                                                                         |

---

## 13. Appendix E — Report outline and page budget (IEEE, 8–10 pages)

| Section                                                                        |   Pages |
| ------------------------------------------------------------------------------ | ------: |
| Abstract, introduction, instance and twists                                    |     1.1 |
| Part A: PEAS, classification, benchmark                                        |     0.9 |
| Part B: formulation, heuristic/proof, comparisons, inadmissible gap, IDS study |     2.3 |
| Part C: game model, pruning, slip, evaluation/tuning, games                    |     1.9 |
| Part D: dispatcher, requests, failures, validation, traces                     |     1.4 |
| Ablation, failure analysis and UI                                              |     0.9 |
| Ethics/reflection, conclusion and references                                   |     1.0 |
| **Total planned**                                                              | **9.5** |

Every figure/table caption carries the stamp; numbers come from generated macros only. Confirm whether references/appendices count toward the instructor's page limit. Do not exceed the permitted total by assuming references are excluded.

---

## 14. Final consistency checklist (run before M0 starts and again before release)

- [ ] All state variables that affect future legal actions, costs, transitions or terminal conditions are represented in the relevant planner/game state or immutable planning context.
- [ ] Deterministic planner, zero-slip simulator and deterministic game transitions obey `docs/transition_model.md`.
- [ ] Stochastic simulator-only effects are clearly separated from deterministic planner assumptions.
- [ ] Every heuristic has an explicit relaxed model, proof assumptions and tests; directed/tunnel movement is handled correctly.
- [ ] Every transition cost is finite, strictly positive and consistent across components.
- [ ] Search limits return distinct statuses and never masquerade as infeasibility.
- [ ] State-space estimator reports an upper bound with its assumptions, not a fabricated exact count.
- [ ] `max_cost` is enforced using accumulated path cost; fuel/toll budgets are represented as state where they affect future legality.
- [ ] Game terminal utilities dominate all non-terminal evaluations; terminal precedence and C\* definition are fixed.
- [ ] BFS/IDS results are interpreted as minimum-action results, UCS/A\* as minimum-cost results.
- [ ] Stored-record counts are not labelled physical memory; byte memory is measured separately.
- [ ] Scenario/request/game comparisons use paired or cluster-aware analysis; tuning and evaluation seeds are disjoint.
- [ ] Quick-mode outputs cannot satisfy full-mode report/release requirements.
- [ ] Every final output is stamped and traceable to configuration, seed, source and commit.
- [ ] The four-page brief and full instructor script were audited at M0.0; the report's page budget and every source-document discrepancy are represented accurately.
- [ ] Internal scenario keys are never confused with real member IDs or stamp data; the instructor is not expected to provide custom scenario inputs.
- [ ] Game action generation includes all enabled twist-specific actions and the canonical plan tie-break defines `T_max` reproducibly.
- [ ] The weighted-A\* constructed counterexample passes and is clearly distinguished from generated delivery-scenario observations.
- [ ] All user/instructor approvals are real, recorded and complete before their checkpoints are ticked.
- [ ] M0.0 source audit is included in and ticked with the first M0.1 Git commit; no pre-repository commit is attempted.
