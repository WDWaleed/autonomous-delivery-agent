# Mathematical Proof of Admissibility and Consistency Analysis (§7.4, M5.2)

**Project:** CS-370 Artificial Intelligence (Fall 2026) Open-Ended Lab  
**Group:** Muhammad Waleed Ahmed (513303), Ahmad Raza (522356), Ali Hamza (514549)  
**Heuristic:** Combined Minimum Spanning Tree and Farthest Reach ($h_{\text{comb}}$)

---

## 1. Problem Formulation & Definitions

Let the state space be $\mathcal{S}$, where each state $s = (\text{pos}, \text{mask}, \text{extra})$ consists of:
- The agent's current position $\text{pos} \in \mathcal{V}_{\text{grid}}$.
- The bitmask $\text{mask} \in [0, 2^K - 1]$ recording delivered stops among $K$ targets $\{u_1, \dots, u_K\}$.
- The set of remaining undelivered stops $U(s) = \{u_i \mid \text{bit } i \text{ of mask is } 0\}$.
- The goal predicate $\text{Goal}(s) \iff U(s) = \emptyset$ (and $\text{pos} = \text{start}$ if return-to-depot is required).
- Non-uniform step costs $c(s, a, s') > 0$ for all legal transitions $s \xrightarrow{a} s'$.

Let $h^*(s)$ denote the true optimal accumulated step cost of reaching any goal state from $s$.
If no goal state is reachable from $s$, $h^*(s) = \infty$.

---

## 2. Definition of the Relaxed Graph $G_{\text{relaxed}}$

Let $G = (V, E, c)$ be the directed graph induced by the delivery city under all active rules and twists.
We define the relaxed graph $G_{\text{relaxed}} = (V_R, E_R, c_R)$ as follows:
1. **Fog of War Relaxation:** Static terrain passability and terrain movement costs (Road: 3, Grass: 4, Mud: 5, Ford: 7) are fully known.
2. **One-Way Street Enforcement:** Permitted directed movement directions are respected. For any directed edge $(p, q)$, $c_R(p, q) = c(p, q)$ if movement from $p$ to $q$ is permitted, else $\infty$.
3. **Fragile Parcel Relaxation:** The fragile parcel constraint (which bans entry into mud and ford cells while the fragile parcel is carried) is relaxed. That is, cells with mud and ford are treated as passable with their base terrain entry costs in $G_{\text{relaxed}}$.

### Lemma 1 (Cost Relaxation Invariant)
*For any legal sequence of moves in the true environment transition system from $s$ to $s'$, the relaxed distance in $G_{\text{relaxed}}$ satisfies:*
$$d_R(\text{pos}(s), \text{pos}(s')) \le \text{true\_cost}(s \to s')$$
*Proof:* Every legal movement in the true environment corresponds to a valid sequence of directed edges in $G_{\text{relaxed}}$ with edge cost $c_R(e) \le c(e)$. Because shortest paths minimize the sum of edge costs, $d_R(\text{pos}(s), \text{pos}(s'))$ cannot exceed the cost of any feasible path. $\blacksquare$

---

## 3. Component Proofs

### Component 1: Baseline B2 (Farthest Directed Reach)
$$h_{B2}(s) = \max_{u \in U(s)} d_R(\text{pos}(s), u)$$

#### Lemma 2 (Admissibility of $h_{B2}$)
*For all states $s \in \mathcal{S}$, $h_{B2}(s) \le h^*(s)$.*
*Proof:*
Any valid delivery plan starting from $s$ must visit every stop in $U(s)$.
In particular, for any specific stop $u_j \in U(s)$, the plan must contain a subpath from $\text{pos}(s)$ to $u_j$.
By Lemma 1, the cost of reaching $u_j$ is lower-bounded by $d_R(\text{pos}(s), u_j)$.
Since the total plan cost is at least the cost of reaching $u_j$, we have $h^*(s) \ge d_R(\text{pos}(s), u_j)$ for all $u_j \in U(s)$.
Taking the maximum over all $u_j \in U(s)$ yields:
$$h^*(s) \ge \max_{u \in U(s)} d_R(\text{pos}(s), u) = h_{B2}(s)$$
Thus $h_{B2}$ is admissible. $\blacksquare$

---

### Component 2: Relaxed Minimum Spanning Tree ($h_{\text{MST}}$)
Let $V_U = U(s) \cup \{\text{pos}(s)\}$ be the set of waypoints that must be visited.
For any pair $u, v \in V_U$, define the symmetric lower-bound pair weight:
$$w(u, v) = \min\big(d_R(u, v), d_R(v, u)\big)$$
Let $K_{V_U} = (V_U, E_U, w)$ be the complete undirected graph on $V_U$ with edge weights $w$.
$$h_{\text{MST}}(s) = \text{weight}\big(\text{MST}(K_{V_U})\big)$$

#### Lemma 3 (Admissibility of $h_{\text{MST}}$)
*For all states $s \in \mathcal{S}$, $h_{\text{MST}}(s) \le h^*(s)$.*
*Proof:*
Consider any optimal delivery plan $P^*$ from $s$, with total cost $h^*(s)$.
$P^*$ induces a directed walk $W$ in the grid that starts at $\text{pos}(s)$ and visits every stop $u \in U(s)$.
The walk $W$ connects all vertices in $V_U$.
Now consider the sequence of waypoints visited by $P^*$: $(v_0, v_1, v_2, \dots, v_m)$ where $v_0 = \text{pos}(s)$ and $\{v_1, \dots, v_m\} = U(s)$.
The total cost of $P^*$ is:
$$h^*(s) = \sum_{i=0}^{m-1} \text{cost}(v_i \to v_{i+1})$$
By Lemma 1 and the definition of $w$:
$$\text{cost}(v_i \to v_{i+1}) \ge d_R(v_i, v_{i+1}) \ge \min\big(d_R(v_i, v_{i+1}), d_R(v_{i+1}, v_i)\big) = w(v_i, v_{i+1})$$
Therefore:
$$h^*(s) \ge \sum_{i=0}^{m-1} w(v_i, v_{i+1})$$
The edges $(v_i, v_{i+1})$ form a spanning walk connecting all vertices of $V_U$.
Any spanning walk on $V_U$ contains a spanning tree as a subgraph.
Since the Minimum Spanning Tree minimizes the total edge weight among all spanning subgraphs on $V_U$:
$$\sum_{i=0}^{m-1} w(v_i, v_{i+1}) \ge \text{weight}\big(\text{MST}(K_{V_U})\big) = h_{\text{MST}}(s)$$
Combining inequalities yields $h^*(s) \ge h_{\text{MST}}(s)$. Thus $h_{\text{MST}}$ is admissible. $\blacksquare$

---

## 4. Theorem: Admissibility and Dominance of $h_{\text{comb}}$

### Theorem 1 (Admissibility of Combined Heuristic)
*Let $h_{\text{comb}}(s) = \max\big(h_{\text{MST}}(s), h_{B2}(s)\big)$. Then for all $s \in \mathcal{S}$:*
$$h_{\text{comb}}(s) \le h^*(s)$$
*Proof:*
By Lemma 2, $h_{B2}(s) \le h^*(s)$.
By Lemma 3, $h_{\text{MST}}(s) \le h^*(s)$.
Since both components are upper-bounded by $h^*(s)$, their maximum must also be upper-bounded by $h^*(s)$:
$$\max\big(h_{\text{MST}}(s), h_{B2}(s)\big) \le h^*(s)$$
Hence $h_{\text{comb}}$ is strictly admissible. $\blacksquare$

### Theorem 2 (Heuristic Dominance)
*For all states $s \in \mathcal{S}$:*
$$h_{\text{comb}}(s) \ge h_{\text{MST}}(s) \quad \text{and} \quad h_{\text{comb}}(s) \ge h_{B2}(s)$$
*Proof:* Direct consequence of the definition of the $\max$ operator. $\blacksquare$

---

## 5. Consistency (Monotonicity) Analysis

A heuristic is consistent (monotonic) if for all states $s$ and legal transitions $s \xrightarrow{a} s'$:
$$h(s) \le c(s, a, s') + h(s')$$
and $h(s_{\text{goal}}) = 0$.

### Theoretical Analysis
1. **Component $h_{B2}$:** The directed distance field $d_R(\cdot, u)$ satisfies the triangle inequality on $G_{\text{relaxed}}$. When a transition does not deliver a stop, $U(s) = U(s')$, so $d_R(\text{pos}(s), u) \le c_R(\text{pos}(s), \text{pos}(s')) + d_R(\text{pos}(s'), u) \le c(s, a, s') + d_R(\text{pos}(s'), u)$. Hence $h_{B2}$ is consistent on non-delivery transitions. Upon delivery of stop $u$, $u$ drops out of $U(s')$, which can only decrease the remaining maximum, preserving $h(s) - h(s') \le c$.
2. **Component $h_{\text{MST}}$:** When pos transitions to $\text{pos}'$, the vertex set changes from $U \cup \{\text{pos}\}$ to $U \cup \{\text{pos}'\}$. By the triangle inequality on shortest path distances, adding an edge from $\text{pos}$ to $\text{pos}'$ bounds the change in MST weight by $d_R(\text{pos}, \text{pos}') \le c$.
3. **Pointwise Maximum:** The maximum of two consistent heuristics is consistent. However, because $w(u, v) = \min(d_R(u, v), d_R(v, u))$ uses symmetric relaxation on a directed graph, slight local consistency violations can occur under asymmetric one-way constraints.
4. **Preservation of Optimality:** In our search engine (`delivery/search/astar.py`), A* is implemented with **closed-node re-opening**: whenever a path with strictly lower $g(n)$ is discovered to an already closed state, the node is re-opened and returned to the priority queue. Therefore, even in the presence of any empirical inconsistency, **A* guarantees finding an optimal-cost solution plan**.
