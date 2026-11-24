# Transition Model Specification

**Status:** Formal Specification (Updated in M2.4)  
**Governing Decisions:** D1, D2, D3, D6, D15, D16 (Single unified transition semantics)

---

## 1. World Representation
- **Grid Coordinates:** $(r, c)$ with $0 \le r < n, 0 \le c < n$.
- **Terrain Types:**
  - `0`: Road (Cost $c_0 = 3$)
  - `1`: Grass (Cost $c_1 = 4$)
  - `2`: Mud (Cost $c_2 = 5$)
  - `3`: Ford (Cost $c_3 = 7$)
  - `-1`: Blocked obstacle (Impassable)
  - Invariant: $1 \le c_0 < c_1 < c_2 < c_3 \le 9$.

### 1.1 Cost Map & Sentinel Convention
- **Vectorized Cost Map:** $C \in \mathbb{R}^{n \times n}$ represents entry traversal costs.
- **Sentinel for Blocked Cells:** Blocked obstacles (`-1`) are assigned $C[r, c] = +\infty$ (`np.inf`).
- **Mathematical Rationale:** Preserves strictly positive transition costs ($C[r, c] > 0$) for traversable cells per D3 and D15, while ensuring Dijkstra, Bellman-Ford, and heuristic distance fields natively treat blocked obstacles as unreachable.

## 2. Actions & Deterministic Nominal Transitions
- **Cardinal Actions:**
  - North (0): $(-1, 0)$
  - East (1): $(0, 1)$
  - South (2): $(1, 0)$
  - West (3): $(0, -1)$
- **Special Actions:**
  - `WAIT`: $(0, 0)$
- **Nominal Successor:**
  $$s' = (r + \Delta r, c + \Delta c)$$
  Legal if $0 \le r' < n, 0 \le c' < n$ and $\text{grid}[r'][c'] \ge 0$.

## 3. Stochastic Slip Model (Simulator Only)
- In the simulator, an intended cardinal action slips with probability $p_{\text{slip}} \in [0.10, 0.30]$ (for instance `432382d9`, $p_{\text{slip}} = 0.23$).
- Upon slip, the agent is displaced by one of the two perpendicular directions uniformly at random:
  - If intended action is North or South: perpendicular candidates are East and West.
  - If intended action is East or West: perpendicular candidates are North and South.
- If neither perpendicular cell is traversable, the agent remains in place and pays fallback cost.
- Per Decision D6, planning algorithms plan over nominal transitions; online execution and simulation handle stochastic transitions.

## 4. Assigned Twist Specifications (Instance `432382d9`)

### 4.1 Twist A: Fog of War / Partial Observability (`fog_partial_observability`)
- **Perception Window:** At agent position $(r, c)$, cells in Chebyshev radius $\rho = 4$ are visible:
  $$\text{Visible}(r, c) = \{(r', c') \mid \max(|r - r'|, |c - c'|) \le 4, 0 \le r' < n, 0 \le c' < n\}$$
- **Belief State:** Agent maintains a discovered mask $M \in \{0, 1\}^{n \times n}$.
- **Optimistic Belief Costs:** For unobserved cells ($M[r, c] = 0$), agent assumes $c_{\text{optimistic}} = c_{\min} = 3$ (road cost).
- **Online Replanning:** When newly observed cells reveal an obstacle or higher terrain cost along the planned route, the agent dynamically replans.

### 4.2 Twist B: One-Way Streets (`one_way_streets`)
- **Directed Edge Generation:** ~25% of traversable grid edges are deterministically directed using seed `instance.seed ^ 0x01E_04A`.
- **Allowed Transition Mask:** $A \in \{0, 1\}^{n^2 \times 4}$, where $A[u, d] = 1$ if leaving node $u$ via action $d$ is permitted.
- **Reachability Guarantee:** Edges are rejected from being one-way if making them one-way disconnects any target in $\{\text{start}\} \cup \text{delivery\_stops}$. Hence, all targets remain mutually reachable.

### 4.3 Third Twist: Fragile Parcel (`fragile_parcel`)
- **Restricted Terrains:** While fragile parcel is undelivered (`fragile_delivered == False`), entering `mud` (terrain 2) or `ford` (terrain 3) is illegal.
- **Delivery Trigger:** Upon reaching the designated fragile stop, `fragile_delivered` becomes `True`, restoring standard terrain traversal.

## 5. Combined Search State Space
For multi-stop delivery with all active twists:
$$S = (r, c, \text{delivered\_mask}, \text{fragile\_delivered})$$
- $(r, c) \in [0, n-1]^2$: agent position
- $\text{delivered\_mask} \in [0, 2^k - 1]$: bitmask of completed deliveries ($k$ stops)
- $\text{fragile\_delivered} \in \{0, 1\}$: fragile delivery status

### Legal Transition Check:
A transition from $(r, c)$ to $(r', c')$ via action $a$ is legal if and only if:
1. $0 \le r' < n, 0 \le c' < n$
2. $\text{grid}[r', c'] \ne -1$ (passable cell)
3. $A[\text{idx}(r, c), a] = 1$ (permitted by one-way street rule)
4. If $\text{fragile\_delivered} = 0$, then $\text{grid}[r', c'] \notin \{2, 3\}$ (fragile parcel restriction)

## 6. Relaxed Graph & Admissible Lower Bounds

To compute admissible heuristics for A* search and TSP (Minimum Spanning Tree / Held-Karp bounds):
- **Relaxed Graph $G_{\text{relaxed}} = (V, E_{\text{relaxed}})$:**
  - One-way street directions are relaxed: for any adjacent passable cells $u$ and $v$, both directed edges $(u, v)$ and $(v, u)$ are present.
  - Step costs equal the optimistic entering terrain costs: $c_{\text{relaxed}}(u \to v) = \text{cost}(v)$.
  - Fragile parcel terrain restrictions are relaxed (mud and ford are traversable).

### Lower-Bound Invariant:
Since $E_{\text{actual}} \subseteq E_{\text{relaxed}}$ and $c_{\text{relaxed}}(e) \le c_{\text{actual}}(e)$ for all edges $e$:
$$\forall u, v \in V: \quad d_{\text{relaxed}}(u, v) \le d_{\text{actual}}(u, v)$$

### Symmetric TSP Metric:
For MST-based multi-goal tour estimation, directed distances may be asymmetric ($d(u, v) \ne d(v, u)$). We define the symmetric lower-bound edge weight:
$$w_{\text{MST}}(u, v) = \min(d_{\text{relaxed}}(u, v), d_{\text{relaxed}}(v, u))$$
By definition, $w_{\text{MST}}(u, v) = w_{\text{MST}}(v, u)$ and:
$$w_{\text{MST}}(u, v) \le \min(d_{\text{actual}}(u, v), d_{\text{actual}}(v, u)) \le d_{\text{actual}}(u, v)$$
Thus, any MST computed over $w_{\text{MST}}$ strictly lower-bounds the true optimal directed tour cost, guaranteeing A* admissibility.
