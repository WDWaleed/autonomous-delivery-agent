# Transition Model Specification

**Status:** Skeleton Specification (Initialised in M0.1)  
**Governing Decision:** D16 (Single unified transition semantics)

---

## 1. World Representation
- **Grid Coordinates:** $(r, c)$ with $0 \le r < n, 0 \le c < n$.
- **Terrain Types:**
  - `0`: Road (Cost $c_0$)
  - `1`: Grass (Cost $c_1$)
  - `2`: Mud (Cost $c_2$)
  - `3`: Ford (Cost $c_3$)
  - `-1`: Blocked obstacle (Impassable)
  - Invariant: $1 \le c_0 < c_1 < c_2 < c_3 \le 9$.

## 2. Actions & Deterministic Nominal Transitions
- **Cardinal Actions:**
  - North: $(-1, 0)$
  - East: $(0, 1)$
  - South: $(1, 0)$
  - West: $(0, -1)$
- **Special Actions:**
  - `WAIT`: $(0, 0)$
- **Nominal Successor:**
  $$s' = (r + \Delta r, c + \Delta c)$$
  Legal if $0 \le r' < n, 0 \le c' < n$ and $\text{grid}[r'][c'] \ge 0$.

## 3. Stochastic Slip Model (Simulator Only)
- In the simulator, an intended cardinal action slips with probability $p_{\text{slip}} \in [0.10, 0.30]$.
- Upon slip, the agent is displaced by one of the two perpendicular directions uniformly at random:
  - If intended action is North or South: perpendicular candidates are East and West.
  - If intended action is East or West: perpendicular candidates are North and South.
- If neither perpendicular cell is traversable, the agent remains in place and pays fallback cost.

## 4. Twist Transition Extensions
Twists hook into the transition model via `delivery/twists/base.py`:
- `fuel_limit_with_charging_stations`: Depletes fuel by step; charging stations replenish fuel.
- `one_way_streets`: Restricts allowable directions on selected edges.
- `rush_hour_time_varying_costs`: Modifies cell traversal costs as a function of global tick $t$.
- `fog_partial_observability`: Restricts visible radius to Chebyshev window around agent.
- `toll_gates_budget`: Requires toll payment against a budget.
- `teleport_tunnels`: Non-local instant transitions between tunnel endpoints.
- `wind_drift`: Additional stochastic lateral displacement.
- `rival_steals_parcels`: Rival occupying stop steals parcel if not yet delivered.
