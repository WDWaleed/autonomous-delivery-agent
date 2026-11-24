# Part C Game Formalisation: Reduced Window Adversarial Delivery Game (§7.3, §7.5, D7, D12, M6.1)

**Group:** Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)  
**Canonical Instance:** `432382d9` ($22 \times 22$, assigned adversary: `chaser`, depth `6` plies)

---

## 1. Players & Turn Structure

- **MAX Player (`AGENT = 0`):** Autonomous delivery agent seeking to complete deliveries while avoiding capture and keeping path cost within $\kappa \cdot C^*$.
- **MIN Player (`RIVAL = 1`):** Assigned `chaser` adversary seeking to capture the agent (or force a loss/draw).
- **Alternating Plies:** Each player takes one action per ply (`AGENT` moves at even plies, `RIVAL` moves at odd plies). Depth $d$ in Minimax / Alpha-Beta is measured in **plies** (half-moves).

---

## 2. Complete State Representation (`GameState`)

Every node in the game tree carries the complete immutable `GameState`:
$$s = (\text{agent\_pos},\; \text{rival\_pos},\; \text{delivered\_mask},\; \text{extra\_state},\; \text{turn},\; \text{agent\_cost},\; \text{steps},\; \text{plies})$$

- `agent_pos`: $(r_A, c_A)$ grid coordinates.
- `rival_pos`: $(r_R, c_R)$ grid coordinates.
- `delivered_mask`: bitmask over $K$ delivery stops.
- `extra_state`: twist-specific state tuple (e.g., `fragile_parcel` status).
- `turn`: active player (`0` for AGENT, `1` for RIVAL).
- `agent_cost`: accumulated terrain movement cost paid by AGENT.
- `steps`: number of AGENT actions taken so far.
- `plies`: total half-moves elapsed.

---

## 3. Action Space & Canonical Tie-Breaking

Base actions are ordered canonically as:
$$\text{NORTH } (0) \prec \text{EAST } (1) \prec \text{SOUTH } (2) \prec \text{WEST } (3) \prec \text{WAIT } (4)$$
When twist-specific actions are enabled (e.g., `TELEPORT = 5` for `teleport_tunnels`), they follow in registry order.

`legal_actions(state, player)` filters moves by:
1. Grid bounds $[0, N-1]^2$ and terrain passability (`grid[r, c] >= 0`).
2. Active window boundary constraint (when `window_radius` $W$ is set).
3. One-way street restrictions (`one_way_streets`) and active twist legality hooks (`fragile_parcel` mud/ford ban for the agent before stop $(5, 15)$).

---

## 4. Chebyshev Window Approximation & Strategic Target

- **Window Extraction (`D12`):** Around `window_center` with radius $W$ (default $W = 4$), we extract the square Chebyshev region $\max(|r - c_r|, |c - c_c|) \le W$, clipped to $[0, N-1]^2$. Actors inside the window cannot step outside the window boundary during tactical lookahead.
- **Strategic Target:** The strategic A\* planner (`Own_CombinedMST`) provides the next-stop target and remaining route cost estimate, while tactical Alpha-Beta handles local rival evasion inside the window.

---

## 5. Canonical Baseline ($C^*, L^*$), Step Cap ($T_{\max}$), and $\kappa$ Threshold

- On the deterministic rival-free, slip-free instance with all active twists, `compute_canonical_baseline` finds the canonical optimal plan minimizing $( \text{total\_cost},\; \text{action\_count},\; \text{action-rank sequence} )$.
- **Optimal Cost:** $C^*$, **Canonical Action Count:** $L^*$.
- **Step Cap:** $T_{\max} = 4 \cdot L^* + 20$ (bounded on `steps`).
- **Cost Threshold:** $\kappa = 1.5$ (win requires $\text{agent\_cost} \le \kappa \cdot C^*$).
- If $C^*$ and $L^*$ cannot be certified, the scenario is marked uncertified (`certified = False`) and no synthetic threshold is invented.

---

## 6. Event Precedence & Capture Semantics (D7)

Terminal checks execute in strict **D7 precedence order**:
1. **Capture (`LOSS_CAPTURE`):** `agent_pos == rival_pos` (immediate same-cell contact). A reusable `check_edge_swap_capture(prev_a, curr_a, prev_r, curr_r)` helper is also provided for paired/joint transitions and simulator replay semantics; under pure alternating one-ply transitions, same-cell contact resolves immediately after each ply.
2. **Theft (`LOSS_THEFT`):** Active only when `rival_steals_parcels` is enabled and the rival enters an undelivered stop.
3. **Completed Delivery (`WIN` vs `DRAW_EXCESS_COST`):** All required stops delivered (and at depot if `return_to_depot=True`):
   - `WIN` if $\text{agent\_cost} \le \kappa \cdot C^*$ (or uncertified).
   - `DRAW_EXCESS_COST` if $\text{agent\_cost} > \kappa \cdot C^*$.
4. **Excessive-Cost Draw (`DRAW_EXCESS_COST`):** $\text{agent\_cost} > \kappa \cdot C^*$ mid-episode.
5. **Step-Cap Draw (`DRAW_STEP_CAP`):** $\text{steps} \ge T_{\max}$.

---

## 7. Utility Bounds

$$U(s) = \begin{cases}
+M - \min(99.0,\; 0.001 \cdot \text{agent\_cost}) & \text{if } \text{WIN} \\
-M & \text{if } \text{LOSS\_CAPTURE} \text{ or } \text{LOSS\_THEFT} \\
0 & \text{if } \text{DRAW\_EXCESS\_COST} \text{ or } \text{DRAW\_STEP\_CAP}
\end{cases}$$

where $M = 10{,}000.0$, ensuring $M \gg \max |f(s)|$ (non-terminal evaluations are strictly bounded within $[-1{,}000.0, +1{,}000.0]$) and the small cost tie-break on `WIN` can never cross into draw or loss territory.
