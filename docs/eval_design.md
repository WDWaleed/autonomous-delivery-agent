# Evaluation Function Design (§7.5, M7.3)

**Group:** Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)

---

## 1. Normalized Features, Ranges, and Signs

`MultiFeatureEvaluator` (`delivery/game/evaluation.py`) combines five normalized features, each bounded in $[0, 1]$:

| Feature | Range | Default Weight | Sign & Rationale |
| :--- | :---: | :---: | :--- |
| `delivered_fraction` | $[0, 1]$ | $+350.0$ | **Positive:** Monotonically rewards completing more stops toward the goal. |
| `remaining_route_norm` | $[0, 1]$ | $-300.0$ | **Negative:** Uses our admissible `CombinedMSTHeuristic` normalized by $C^*$ (or grid scale). Lower remaining cost is better. |
| `rival_distance_sat` | $[0, 1]$ | $+180.0$ | **Positive (conditional):** $\min(d_{\text{rival}}, R_{\text{sat}}) / R_{\text{sat}}$ with $R_{\text{sat}}=6$. Keeping distance from the chaser reduces capture risk, saturated at $6$ so distant threats do not override delivery progress. |
| `escape_mobility` | $[0, 1]$ | $+70.0$ | **Positive:** Fraction of legal cardinal exits ($\text{deg}/4$). Discourages entering dead-end corridors where the chaser can trap the agent. |
| `cost_margin` | $[0, 1]$ | $+50.0$ | **Positive:** Remaining budget $(\kappa C^* - \text{agent\_cost}) / (\kappa C^*)$, encouraging cheaper routes and preventing $\kappa$-budget exhaustion draws. |

---

## 2. Terminal Utility Bound Safety

- Terminal utilities are $\pm M = \pm 10{,}000.0$ (`WIN` / `LOSS`) and $0.0$ (`DRAW`).
- The sum of absolute weights is $350 + 300 + 180 + 70 + 50 = 950.0$, and the output is explicitly clamped to $[-999.0, +999.0]$.
- Thus, **no non-terminal evaluation can ever equal or exceed a terminal `WIN` or `LOSS` utility**.

---

## 3. Monotonicity & Why It Differs from Textbook Evaluation

- **Controlled Monotonicity:** Holding all other features fixed, increasing `delivered_fraction`, `rival_distance_sat`, `escape_mobility`, or `cost_margin` strictly increases $f(s)$, while increasing `remaining_route_norm` strictly decreases $f(s)$.
- **Conditional Nature of Rival Distance:** Unlike a textbook zero-sum game (e.g., Chess material count), "farther rival is better" is **not** unconditional in delivery: if moving away from the rival traps the agent in a dead end (`escape_mobility` drops) or forces a massive detour (`remaining_route_norm` rises and `cost_margin` drops), the combined evaluation correctly balances evasion against delivery progress.
- **Twist Interaction:** `CombinedMSTHeuristic` respects `one_way_streets` via directed relaxed distance fields, and `escape_mobility` respects `one_way_streets` and `fragile_parcel` (banning mud/ford cells before stop $(5,15)$), so dead ends induced by twists are immediately penalized.
