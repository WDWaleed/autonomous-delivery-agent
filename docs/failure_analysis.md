# Cross-Stage Failure Analysis & Limitations (M9.2)

**Group:** Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)

---

## 1. Failure Taxonomy Across Project Stages

All counts in [`results/tables/failure_analysis.md`](../results/tables/failure_analysis.md) are traced directly to the raw experiment logs:

1. **Part A — Reflex Agent Failures (`30/30` failed):**
   - **Root Cause:** The stateless `ReflexAgent` lacks a map model, delivery-state memory, or lookahead, causing `LOCAL_MINIMUM_OSCILLATION` or `STEP_CAP_EXCEEDED` behind concave obstacles (`SCN-000`–`SCN-029`).
   - **Mitigation:** Replaced with stateful A\* multi-stop search (`CombinedMSTHeuristic`) in Part B (`30/30` solved).

2. **Part B — Uninformed Bounded Search DNF (`90/210` runs across `DLS`/`IDS`):**
   - **Root Cause:** `DLS(d=10)` hits `CUTOFF` below the ~70-step optimal multi-stop depth, while `DLS(d=25)` and `IDS` hit `NODELIMIT` on full 5-stop branching without heuristic guidance.
   - **Mitigation:** Informed A\* with `CombinedMSTHeuristic` solves `30/30` in `387.4` mean expansions (`0.0225` s).

3. **Part C — Adversarial Game Losses & Draws (`50` Eval Games × `2` Configs):**
   - **Capture Losses:** `20.0%` (`Default`) / `22.0%` (`Tuned`) caused primarily by stochastic perpendicular slip ($p = 0.23$) displacing the agent into the `chaser` rival or dead-end corridors.
   - **Step-Cap / Excess-Cost Draws:** `38.0%` (`Default`) / `36.0%` (`Tuned`) when repeated rival evasion and slip-induced replans exhaust $T_{\max} = 4L^* + 20$ or $\kappa C^*$.
   - **Mitigation:** Disjoint-tuned `route_heavy` weights reduced draws (`38% -> 36%`) and lowered mean cost (`417.3 -> 407.9`).

4. **Part D — Live LLM Dispatcher Failures (`groq/openai/gpt-oss-120b`):**
   - **Rate Limits & Timeouts:** `7` failures across `NoVal` / `WithVal` (`REQ-02/03` timeouts, `REQ-06/11/16` HTTP 429).
   - **Schema / Reasoning Errors:** `HTTP 400` malformed tool JSON (`REQ-10`, `REQ-14`), plus premature stopping after `inspect_twists` (`REQ-09`).
   - **Mitigation:** `ArgumentValidator` improved overall success (`65.0% -> 70.0%`) and intercepted contradictory constraints (`REQ-19`: `CONTRADICTORY_START_FORBIDDEN`) with zero planner side effects.

5. **UI / Runtime Failures (`PENDING_M10`):**
   - Explicitly marked as `PENDING_M10` until the Streamlit UI (M10) manual verification checklist is executed.
