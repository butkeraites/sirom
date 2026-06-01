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
PDLP 2.6×→3.6×). PDLP — a first-order method — does not close the gap (see
"Does PDLP catch up?" below).

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

## Does PDLP catch up at large sparse scale? (no)

PDLP is built for large *sparse* LPs, so we timed a single LP built directly in
OR-Tools (box-bounded variables + sparse coupling rows), isolating `Solve()`:

| LP (box + sparse coupling) | density | GLOP | CLP | PDLP |
|---|---|---|---|---|
| 3000 vars × 8000 rows, 6 nnz/row | 0.20% | 0.006s | 0.006s | 0.097s (16×) |
| 8000 vars × 20000 rows, 8 nnz/row | 0.10% | 0.046s | 0.049s | 1.204s (26×) |

The gap **widens** with size; GLOP/CLP solve an 8000×20000 sparse LP in ~50 ms.
The reason is **problem hardness, not size or sparsity**: these LPs are
box-dominated with loose coupling, so the optimum is trivial for simplex (few
pivots). PDLP pays a fixed first-order iteration cost regardless, so it can't
win on an *easy* LP however large. PDLP wins only on large **and hard** LPs
(highly degenerate / tightly-coupled / network-flow), which interval-perturbed
box-bounded SIROM problems are not.

## Conclusion

GLOP is the right default: fastest, scales best, vertex-stable with CLP. CLP is
a safe drop-in for *identical* results at a speed cost. PDLP offers no benefit
for this problem class — slower and result-shifting — and does not catch up at
scale, because SIROM's LPs are easy for simplex regardless of size.

## Commercial solvers

SIROM passes the solver name straight to OR-Tools' `CreateSolver`, so **any
backend OR-Tools is built with works** — including the commercial **GUROBI**,
**CPLEX**, and **XPRESS** — given a build linked against them and a valid
license. No SIROM code changes are needed:

- **Library:** `ProblemsBucket(..., solver_selection="GUROBI")`.
- **HTTP API:** `{"options": {"solver": "GUROBI"}}` on `POST /solve`.

Auto-selection (GLOP for LP, SCIP for MILP) applies when no solver is given.
A requested solver that isn't in the current build fails fast with a clear
message rather than a crash. Commercial solvers are not included in the default
OR-Tools wheel; they can pay off on large *hard* MILPs (the integer side), where
their branch-and-cut is far stronger than the open-source SCIP/CBC — though for
SIROM's continuous LP path GLOP is already optimal (above).

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
