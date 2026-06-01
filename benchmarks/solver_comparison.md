# LP solver comparison

SIROM solves continuous LPs and auto-selects **GLOP** (OR-Tools' simplex). This
note compares GLOP against the other LP backends available in this OR-Tools
build — **CLP** (COIN-OR simplex) and **PDLP** (first-order) — on speed and on
the frontier they produce. (`SCIP`/`CBC` are the MIP backends, used only when
integer variables are present.) Measured over **replayed scenarios**, so the
solver is the only variable.

## Correctness

All three solve the canonical problem to the exact optimum (−34.0).

## Performance — solve + cluster + tree, fixed problem

| solver | 20×50, N=300 | 50×150, N=400 | vs GLOP @ 50×150 |
|---|---|---|---|
| **GLOP** | 0.745s | 3.813s | 1.0× |
| CLP | 0.972s | 11.279s | 3.0× slower |
| PDLP | 1.927s | 13.706s | 3.6× slower |

GLOP is fastest at both sizes, and its lead **grows** with scale (CLP 1.3×→3.0×,
PDLP 2.6×→3.6×). PDLP does not close the gap — it is a first-order method built
for very large sparse LPs; these per-scenario LPs are small and dense, squarely
in simplex's sweet spot.

## Frontier (same objective/feasibility envelope for all three)

| pair | overlap @ 20×50 | overlap @ 50×150 |
|---|---|---|
| GLOP vs CLP | 1.00 (identical) | 1.00 (identical) |
| GLOP vs PDLP | 0.52 | 0.39 |

- **GLOP and CLP are interchangeable on results** — both simplex, so they pick
  the same optimal vertices ⇒ identical clustering ⇒ identical frontier.
- **PDLP produces a different frontier** (and diverges more at scale) — a
  first-order method converges to a non-vertex point on the optimal face, so the
  clustering features differ. Same as the slack-fix finding: solver choice
  reshuffles *which* solutions make the frontier, never its reach.

## Conclusion

GLOP is the right default: fastest, scales best, vertex-stable with CLP. CLP is
a safe drop-in for *identical* results at a speed cost. PDLP offers no benefit
for this problem class (slower and result-shifting); it would only help at a
scale/sparsity far beyond typical SIROM workloads.

## Reproduce

`ProblemsBucket` takes an explicit `solver_selection` that overrides the
auto-pick. Replay a fixed scenario set (see `frontier_diff.py`) and vary it:

```python
bucket = ProblemsBucket(c, lb_A, ub_A, lb_b, ub_b,
                        number_of_scenarios=N, solver_selection="CLP")  # or "PDLP", "GLOP"
bucket.coefficient.scenarios_constraint = fixed_A   # replay identical scenarios
bucket.coefficient.scenarios_rhs = fixed_b
bucket.solve(); bucket.cluster_and_selection(); bucket.solve_cluster_tree()
```
