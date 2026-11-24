# Metrics & Accounting Specification

**Status:** Skeleton Specification (Initialised in M0.1)  
**Governing Decision:** D18 (`peak_stored` definition and memory accounting)

---

## 1. Search Performance Metrics

For every search algorithm execution (BFS, UCS, DLS, IDS, Greedy, A*):

1. **`nodes_expanded`**: Count of nodes removed from the frontier whose valid successors were generated.
2. **`nodes_generated`**: Total count of successor nodes instantiated and evaluated.
3. **`peak_stored`**:
   - For Graph Search (BFS, UCS, Greedy, A*): Maximum size of $(\text{Frontier} \cup \text{Explored})$ during search.
   - For Tree Search (DLS, IDS): Maximum recursion depth + path bookkeeping records.
4. **`solution_cost`**: Total accumulated transition cost $g(\text{goal})$.
5. **`solution_length`**: Number of actions in the returned path.
6. **`runtime_seconds`**: Wall-clock duration measured with `time.perf_counter()`.
7. **`peak_memory_bytes`**: Physical peak memory sampled via Python `tracemalloc`.

## 2. Adversarial Game Metrics

For Minimax and Alpha-Beta search:
1. **`nodes_evaluated`**: Total leaf evaluations.
2. **`cutoff_count`**: Number of alpha-beta branches pruned.
3. **`pruning_percentage`**:
   $$\text{Pruning \%} = \frac{\text{Minimax Nodes} - \text{AlphaBeta Nodes}}{\text{Minimax Nodes}} \times 100$$
4. **`win_rate`**, **`draw_rate`**, **`loss_rate`**: Proportion of outcomes over $\ge 30$ games with Wilson 95% confidence intervals.
5. **`slip_replan_count`**: Total replans triggered by simulator slip events.
6. **`slip_overhead_cost`**: Extra cost incurred compared to nominal deterministic plan.

## 3. Statistical Standards
- Continuous metrics (cost, nodes, runtime): Sample mean $\bar{x} \pm t_{n-1, 0.975} \frac{s}{\sqrt{n}}$.
- Categorical / binary outcomes (win/loss, success rate): Wilson score interval for binomial proportions.
