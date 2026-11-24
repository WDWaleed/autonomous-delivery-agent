# Part A Formalisation: PEAS, Environment Classification & Agent Architecture

**Authors:** Muhammad Waleed Ahmed (513303) | Ahmad Raza (522356) | Ali Hamza (514549)  
**Milestone:** M3.3 (Completes Part A — v0.1.0)  
**Canonical Instance:** `432382d9` ($22 \times 22$ Grid, 5 Delivery Stops, Adversary: `chaser`, Slip: $0.23$)  
**Active Twists:** `fog_partial_observability` (Twist A), `one_way_streets` (Twist B), `fragile_parcel` (Approved Third Twist)

---

## 1. PEAS Formalism

The autonomous delivery problem is formally defined through the PEAS (Performance Measure, Environment, Actuators, Sensors) framework tailored to canonical instance `432382d9`:

### 1.1 Performance Measure ($P$)
The agent's objective is to maximize total operational utility by successfully completing all parcel drop-offs while minimizing traversal expenditure and risk:
1. **Goal Completion:** Complete delivery of all $k=5$ packages to designated customer stops: $(5, 15), (16, 1), (15, 12), (9, 7), (6, 10)$.
2. **Path Cost Minimization:** Minimize cumulative terrain traversal cost $C = \sum_{t=1}^T c(s_t, a_t, s_{t+1})$, where entry costs are: Road ($c_0 = 3$), Grass ($c_1 = 4$), Mud ($c_2 = 5$), Ford ($c_3 = 7$).
3. **Competitive Survival:** Avoid capture or parcel theft by the adversarial `chaser` rival throughout the entire operational lifetime.
4. **Efficiency Bound ($\kappa$ Criterion):** Ensure the realised delivery cost satisfies $C \le \kappa \cdot C^*$ (with default threshold $\kappa = 1.5$) against the omniscient optimal offline benchmark $C^*$.
5. **Penalty Minimization:** Avoid step-cap exhaustion ($T_{\max} = 100$ in baseline evaluation) and invalid transition attempts (each incurring a fallback stay penalty of $c = 3.0$).

### 1.2 Environment ($E$)
The operational environment is a complex, synthetic urban grid city:
- **Topology & Geometry:** $22 \times 22$ Cartesian grid (484 cells) with 26.9% obstacle density (impassable blocks with sentinel cost $+\infty$).
- **Depot & Target Geometry:** Start depot at $(4, 20)$ with 5 spatially dispersed delivery stops.
- **Assigned Twist A (`fog_partial_observability`):** Sensory shroud limiting visibility to a Chebyshev radius $\rho = 4$ around the vehicle ($9 \times 9$ observation window). Unobserved cells must be approached with optimistic belief assumptions ($c_{\min} = 3.0$).
- **Assigned Twist B (`one_way_streets`):** Directed network topology where ~25% of traversable internal edges are one-way streets generated deterministically from seed `seed ^ 0x01E04A`. Mutual reachability among depot and stops is strictly preserved.
- **Approved Third Twist (`fragile_parcel`):** Fragile package assigned to stop $(5, 15)$. While undelivered, high-friction terrain cells (`mud` and `ford`) are strictly illegal to traverse. Traversal is unlocked upon delivery.
- **Adversary Dynamics:** Competitive `chaser` agent actively computing minimax/alpha-beta pursuit plies up to search depth 6.
- **Actuation Noise:** Simulator lateral slip probability $p_{\text{slip}} = 0.23$.

### 1.3 Actuators ($A$)
- **Cardinal Displacements:** $\mathcal{A}_{\text{cardinal}} = \{\text{North (0)}, \text{East (1)}, \text{South (2)}, \text{West (3)}\}$, translating the vehicle by $\Delta \in \{(-1, 0), (0, 1), (1, 0), (0, -1)\}$.
- **Idling Action:** $\text{WAIT (4)}$ ($\Delta = (0, 0)$), holding position to await favorable adversarial clearance or avoid hazardous timing windows.
- **Physical Output:** Motor commands applied to the vehicle drivetrain, subject to stochastic lateral slip displacement under the simulator physics.

### 1.4 Sensors ($S$)
- **Position Sensor:** Returns absolute grid coordinate $(r, c) \in [0, 21]^2$.
- **Perceptual Vision Array:** Scans terrain codes $\{-1, 0, 1, 2, 3\}$ within Chebyshev window $|r' - r| \le 4, |c' - c| \le 4$.
- **Delivery Register:** Provides active delivery bitmask $\text{mask} \in [0, 31]$ indicating completed stops.
- **Traffic Regulation Reader:** Detects one-way street arrows on adjacent edges before movement.
- **Package Integrity Monitor:** Flags the delivery state of the fragile parcel (`fragile_delivered \in \{0, 1\}`).
- **Rival Proximity Sensor:** Tracks adversary coordinates $(r_{\text{rival}}, c_{\text{rival}})$ when within sensory range.

---

## 2. Seven-Row Environment Classification

| Environmental Dimension | Classification | Primary Governing Factors in Canonical Instance `432382d9` |
|---|---|---|
| **1. Observability** | **Partially Observable** | Twist A (`fog_partial_observability`) restricts visual perception to Chebyshev radius $\rho=4$. Distant obstacles and terrain costs are initially unknown. |
| **2. Determinism** | **Stochastic** | Simulator lateral slip probability $p_{\text{slip}} = 0.23$ replaces intended cardinal steps with perpendicular moves at runtime. |
| **3. Episodic vs. Sequential** | **Sequential** | Present actions alter future battery/step expenditure, parcel bitmasks, one-way street reachability, and adversarial distance. |
| **4. Static vs. Dynamic** | **Dynamic** | The adversarial `chaser` rival moves concurrently between agent plies; road delays and competitive pursuit evolve over time. |
| **5. Discrete vs. Continuous** | **Discrete** | Finite grid coordinates $(r, c) \in \{0, \dots, 21\}^2$, discrete cardinal action space, and integer simulation ticks. |
| **6. Single-Agent vs. Multi-Agent** | **Multi-Agent (Competitive)** | The autonomous agent competes against a hostile rival (`chaser`, depth 6) seeking interception or parcel theft. |
| **7. Known vs. Unknown** | **Known Rules / Partially Unknown State** | Transition physics, terrain cost definitions, and game rules are known; exact cell occupancy beyond fog radius is unknown until explored. |

### Justification Details:

1. **Partially Observable:** While textbook grid worlds are often fully observable, Twist A imposes a strict perception horizon ($\rho = 4$). The agent cannot sense obstacle clusters or ford locations beyond 4 Chebyshev units from its current position. Consequently, it must maintain an internal belief grid $M \in \{0, 1\}^{n \times n}$ and dynamically reconcile optimistic assumptions ($c_{\min} = 3$) with discovered reality.
2. **Stochastic:** The transition model is strictly non-deterministic in simulation. While planners assume nominal transitions per Decision D6, actual execution experiences a $23\%$ slip rate, deflecting intended cardinal motion into perpendicular channels or triggering fallback stays.
3. **Sequential:** The problem exhibits deep temporal dependence. Committing to a one-way street permanently closes reverse paths; delivering the fragile parcel alters the terrain traversability of all mud and ford cells across the remaining tour; and early energy expenditure constrains the feasibility of later detours.
4. **Dynamic:** The environment changes while the agent is deliberating or executing its tour. The rival `chaser` computes adversarial search trees up to depth 6, continually altering the hazard landscape and closing off corridors.
5. **Discrete:** The state space decomposes cleanly into discrete cell indices $(r, c)$, discrete delivery masks ($2^5 = 32$ subsets), discrete twist states, and a discrete five-action space ($\{N, E, S, W, \text{WAIT}\}$).
6. **Multi-Agent:** The agent does not operate in isolation. The `chaser` adversary has explicitly conflicting utility: its objective is to minimize distance to the agent to force capture, requiring game-theoretic reasoning (Minimax / Alpha-Beta with positional evaluation).
7. **Known Physics, Unknown Map:** The agent possesses full knowledge of the game mechanics, cost schedules ($3, 4, 5, 7$), and slip distributions. However, because the environment is shrouded in fog, the specific map instances must be learned online through sensor percepts.

---

## 3. Agent Architecture Justification

### 3.1 Goal-Based vs. Utility-Based Agent
An autonomous delivery vehicle operating in instance `432382d9` **cannot be effectively implemented as a purely goal-based agent**. 

A goal-based agent only differentiates between goal states (all 5 parcels delivered) and non-goal states. In our city grid, there exist hundreds of topologically distinct paths that successfully deliver all parcels. However, these paths differ dramatically in:
- Cumulative energy expenditure (traversing road costs 3, while mud costs 5 and ford costs 7).
- Exposure to stochastic slip displacement (open terrain allows slips, whereas confined alleys risk collision fallback).
- Spatial proximity to the adversarial `chaser` (paths crossing central avenues expose the agent to capture, whereas peripheral detours may offer safety at higher distance).
- Trade-offs between detour length and fragile parcel unlock timing.

Therefore, the system requires a **Utility-Based Agent**. The utility function $U(s)$ maps complete trajectory histories to real-valued scalar utility:
$$U(\tau) = - C(\tau) - \lambda_{\text{risk}} \cdot \text{Risk}_{\text{rival}}(\tau) - \lambda_{\text{slip}} \cdot \text{SlipPenalty}(\tau)$$
subject to $C(\tau) \le 1.5 \cdot C^*$. A utility-based architecture enables optimal trade-off reasoning across conflicting sub-goals (e.g., taking a slightly longer road route to bypass risky ford terrain and maintain distance from the chaser).

---

## 4. Why Simple Reflex Agents Fail (Empirical Benchmark Analysis)

In Milestone M3.2, a pure `SimpleReflexAgent` was implemented and evaluated across **30 certified experimental scenarios** ($k=0..29$) generated via deterministic scenario tokens (`SCN-000` to `SCN-029`).

### 4.1 Empirical Findings ([`results/tables/partA_reflex.md`](file:///d:/University/5th%20Semester/Artificial%20Intelligence/Lab/OEL/results/tables/partA_reflex.md))
- **Total Scenarios Evaluated:** 30
- **Successful Deliveries:** 1 / 30
- **Empirical Success Rate:** **3.3%** (95% Wilson Score CI: $[0.6\%, 16.7\%]$)
- **Mean Cost of Successful Runs:** 283.00
- **Total Failures:** 29 / 30 (96.7%)
  - *Local Minimum Oscillation:* 20 / 30 scenarios (66.7%)
  - *Step-Cap Exhaustion:* 9 / 30 scenarios (30.0%)

### 4.2 Structural Failure Mechanisms

```
        Obstacle Wall
      #################
      #               #
      #   Agent (p)   #  ---> Greedy vector points toward Stop (g)
      #       |       #       blocked by concave U-wall!
      #       v       #
      #   Oscillates  #  ---> Steps back & forth between (r, c) & (r, c-1)
      #################
```

1. **Concave Obstacles & Local Minima:**
   The simple reflex agent operates without memory or lookahead ($h(p) = |r - r_g| + |c - c_g|$). When an obstacle wall lies between the agent and its nearest target stop, stepping away from the wall increases Manhattan distance. Because the reflex rule greedily penalizes distance increases, the agent becomes trapped in local minima, endlessly oscillating between two adjacent cells until the step cap ($T_{\max} = 100$) expires. This failure mode was visually captured in the stamped report artifact [`results/figures/partA_reflex_failure.png`](file:///d:/University/5th%20Semester/Artificial%20Intelligence/Lab/OEL/results/figures/partA_reflex_failure.png).
2. **Inability to Handle One-Way Streets:**
   Under Twist B, when a one-way street leads into a dead-end or obstacle barrier, a stateless agent cannot plan a detour. Attempting to reverse violates one-way regulations, resulting in repeated illegal move penalties.
3. **Fragile Parcel Blockades:**
   While the fragile parcel is undelivered, mud and ford cells are illegal. When the straight-line Manhattan path traverses a mud lake, the reflex agent lacks the global search capacity to find an alternate circuitous road corridor.

### 4.3 Conclusion for Part B
These empirical results decisively demonstrate that greedy, reactive behavior is wholly inadequate for the delivery agent city. Systematic graph search algorithms (BFS, UCS, A*) equipped with admissible heuristic estimators that look beyond local horizons are strictly required. This motivates the complete algorithmic search suite implemented in Part B.
