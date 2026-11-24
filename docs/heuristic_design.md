# Heuristic Design Document (§7.4, M5.1)

**Project:** CS-370 Artificial Intelligence (Fall 2026) Open-Ended Lab  
**Group:** Muhammad Waleed Ahmed (513303), Ahmad Raza (522356), Ali Hamza (514549)  
**Assigned Twists:** `fog_partial_observability` ($\rho=4$), `one_way_streets` (~25% directed)  
**Approved Third Twist:** `fragile_parcel` (mud/ford cells illegal until stop $(5, 15)$ is delivered)

---

## 1. Problem Formulation & Objective

In multi-stop delivery planning, the state is $s = (r, c, \text{mask}, \text{extra})$, where:
- $(r, c)$ is the agent's current position on the $N \times N$ grid city.
- $\text{mask} \in [0, 2^K - 1]$ is the delivery bitmask representing delivered stops among $K$ targets.
- $\text{extra}$ represents twist state (e.g. fragile parcel status).
- The set of undelivered stops is $U(s) = \{u_i \mid \text{bit } i \text{ of mask is } 0\}$.
- The goal is reached when $U(s) = \emptyset$ (all parcels delivered) and, if required, the agent has returned to the start depot.

To guarantee that A* returns an optimal-cost solution under non-uniform strictly positive edge costs ($c(e) > 0$), any heuristic evaluation function $h(s)$ must be **admissible**:
$$h(s) \le h^*(s) \quad \forall s \in \mathcal{S}$$
where $h^*(s)$ is the true optimal cost from state $s$ to any goal state.

---

## 2. Candidate Heuristic Designs

### Candidate A — Relaxed Minimum Spanning Tree ($h_{\text{MST}}$)
- **Formula:**
  $$h_{\text{MST}}(s) = \begin{cases} 
    0 & \text{if } U(s) = \emptyset \\
    \text{MST}\Big(U(s) \cup \{\text{pos}(s)\}\Big) & \text{otherwise}
  \end{cases}$$
- **Graph Construction:**
  For any pair of waypoints $u, v \in U(s) \cup \{\text{pos}(s)\}$, the directed shortest-path distance in the relaxed graph $G_{\text{relaxed}}$ is $d_R(u, v)$. Because $G_{\text{relaxed}}$ is directed due to the `one_way_streets` twist, a symmetric lower-bound pair weight is defined as:
  $$w(u, v) = \min\big(d_R(u, v), d_R(v, u)\big)$$
  $h_{\text{MST}}(s)$ is the weight of the Minimum Spanning Tree on the complete undirected graph of waypoints $V_U = U(s) \cup \{\text{pos}(s)\}$.
- **Proof Outline:**
  Any valid delivery plan from $s$ that visits every remaining stop in $U(s)$ traces a directed walk that spans all vertices in $V_U$. This walk forms a connected subgraph spanning $V_U$. Because $w(u, v) \le d_R(u, v)$, the undirected weight of any spanning walk is lower-bounded by the weight of the Minimum Spanning Tree on $(V_U, w)$. Since the true edge costs in the full problem are $\ge$ the relaxed costs in $G_{\text{relaxed}}$, $h_{\text{MST}}(s) \le h^*(s)$.
- **Strength & Computational Cost:**
  Captures the joint distribution of stops and penalizes widely separated delivery clusters. Computing MST on $|V_U| \le 6$ vertices with Kruskal's algorithm takes $< 5\,\mu\text{s}$. However, symmetrization $w(u, v) = \min(d_R(u, v), d_R(v, u))$ loses directed penalties caused by one-way streets.

---

### Candidate B — Combined Lower Bound ($h_{\text{comb}} = \max(h_{\text{MST}}, h_{B2})$) [SELECTED]
- **Formula:**
  $$h_{\text{comb}}(s) = \max\Big(h_{\text{MST}}\big(U(s) \cup \{\text{pos}(s)\}\big),\; \max_{u \in U(s)} d_R(\text{pos}(s), u)\Big)$$
- **Component Justification:**
  1. **$h_{\text{MST}}(s)$ Component:** Accounts for the total interconnectivity cost among all remaining stops.
  2. **$h_{B2}(s) = \max_{u \in U(s)} d_R(\text{pos}(s), u)$ Component:** Accounts for the true directed shortest-path distance from the current position to the farthest remaining stop, preserving directed one-way street asymmetry.
- **Proof of Admissibility:**
  Let $h_1(s) = h_{\text{MST}}(s)$ and $h_2(s) = h_{B2}(s)$.
  - By Candidate A, $h_1(s) \le h^*(s)$.
  - By baseline B2, any path visiting all $u \in U(s)$ must visit every $u \in U(s)$, so $h^*(s) \ge d_R(\text{pos}(s), u)$ for each $u \in U(s)$, hence $h_2(s) \le h^*(s)$.
  - Therefore, $h_{\text{comb}}(s) = \max(h_1(s), h_2(s)) \le h^*(s)$.
- **Dominance:**
  For all states $s$, $h_{\text{comb}}(s) \ge h_{\text{MST}}(s)$ and $h_{\text{comb}}(s) \ge h_{B2}(s)$. Thus, $h_{\text{comb}}$ dominates both Candidate A and baseline B2, guaranteeing equal or fewer node expansions in A* up to tie-breaking.
- **Twist Awareness:**
  - `one_way_streets`: Directed reverse distance fields $d_R(\cdot, u)$ are precomputed on $G_{\text{relaxed}}$, ensuring directed one-way street costs are explicitly accounted for in $h_{B2}$.
  - `fragile_parcel`: The relaxed graph relaxes terrain bans, guaranteeing lower-bound admissibility.
  - `fog_partial_observability`: The planning environment assumes known nominal grid terrain for offline search, which is admissible.

---

### Candidate C — Return-to-Depot Extended MST (Optional Variant)
- **Formula:**
  $$h_{\text{depot}}(s) = \text{MST}\Big(U(s) \cup \{\text{pos}(s)\} \cup \{\text{start}\}\Big)$$
- Only applicable when `return_to_depot=True`. When return-to-depot is false, Candidate C reduces to Candidate A. Candidate B subsumes this by incorporating both start depot distances when return is active.

---

## 3. Decision & Selected Heuristic

**Selected Heuristic:** **Candidate B ($h_{\text{comb}} = \max(h_{\text{MST}}, h_{B2})$)**
- **User Approval:** Approved for implementation in M5.1.
- **Key Advantage for Viva Defense:** Mathematically dominates both individual lower bounds, avoids illegal heuristic addition/double counting, directly counters one-way street asymmetry via exact directed distance, and evaluates in microseconds via precomputed pairwise distance tables.
