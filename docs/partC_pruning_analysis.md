# Part C Move Ordering & Alpha-Beta Pruning Analysis (§7.5, M6.3)

**Group:** Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)  
**Experiment:** 210 sampled positions across 30 certified scenarios, depths $2\dots 6$ plies, random ordering seed `20260603`.

---

## 1. Value Invariance Verification

Across all `210` positions and all depths ($2\dots 6$), Alpha-Beta under `natural`, `random`, and `own` move orderings produced **0 value mismatches** against unpruned Minimax. Move ordering affects only the number of pruned subtrees, never the backed-up minimax value.

---

## 2. Empirical Pruning Results

| Depth | Minimax Evals | AB (`natural`) | AB (`random`) | AB (`own`) | `own` Saved vs Minimax | `own` Saved vs `natural` AB |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | $14.1$ | $9.2$ | $9.7$ | **$7.6$** | $42.4\%$ $[40.1, 44.8]$ | $11.1\%$ $[7.0, 15.2]$ |
| **3** | $54.9$ | $27.8$ | $30.0$ | **$18.6$** | $62.6\%$ $[60.9, 64.4]$ | $25.3\%$ $[21.7, 28.8]$ |
| **4** | $216.7$ | $67.6$ | $72.1$ | **$31.7$** | $81.4\%$ $[79.6, 83.3]$ | $35.6\%$ $[30.2, 41.1]$ |
| **5** | $609.8$ | $188.1$ | $198.3$ | **$79.7$** | $84.1\%$ $[77.8, 90.3]$ | $37.7\%$ $[31.3, 44.1]$ |
| **6** | $2293.8$ | $481.3$ | $495.5$ | **$137.4$** | **$89.7\%$** $[83.1, 96.3]$ | **$44.6\%$** $[35.1, 54.1]$ |

---

## 3. Why `own` Move Ordering Helps in This Game

1. **Tightening $\alpha$ Early at MAX Nodes (Agent):**
   `own` ordering sorts the agent's actions by `(pv_hint, target_relaxed_distance, -rival_distance, action)`. By evaluating the principal variation from the previous depth and the move that makes the strongest progress toward the nearest undelivered stop (without stepping toward the rival) first, MAX establishes a high lower bound $\alpha$ on its very first branch.
2. **Tightening $\beta$ Early at MIN Nodes (Chaser Rival):**
   At MIN nodes, `own` ordering sorts the `chaser` rival's moves by `(pv_hint, distance_to_agent, action)`. Since the `chaser` wants to minimize distance to the agent (and capture gives $-M = -10{,}000$), trying the most aggressive closing move first immediately drives $\beta$ down—often triggering an immediate $\alpha \ge \beta$ cutoff when the agent previously made a move into danger.
3. **Compounding Savings with Depth:**
   At depth $2$, `own` saves $11.1\%$ over `natural` AB; by the assigned search depth of $6$ plies, savings grow to **$44.6\%$ over `natural` AB** ($137.4$ vs $481.3$ evaluations) and **$89.7\%$ over unpruned Minimax** ($137.4$ vs $2293.8$ evaluations).
4. **When `own` Ordering Can Hurt Slightly:**
   In rare tactical traps where the shortest relaxed path toward a stop walks into a dead-end corridor guarded by the rival, the greedy target-distance sort explores the trapped branch first before discovering the safer detour. Combining `pv_hint` from iterative deepening mitigates this by preserving the true minimax best move from depth $d-1$.
