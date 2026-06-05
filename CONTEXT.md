# SIROM

Domain language for the Simulation-based Robust Optimization Method — sampling an
interval-valued linear program across scenarios, solving each, clustering the
solutions, and scoring how often each stays feasible. This file names the
concepts; `CLAUDE.md` covers conventions and mechanics.

## Language

### The pipeline

**Scenario**:
One concrete realization of the uncertain `(A, b)`, materialized by interpolating
`lb + (ub - lb)·δ` from a Latin-Hypercube `δ`.
_Avoid_: sample, draw, future.

**Scoring**:
The stage (`apply_quality_measure`) that takes a solved problem and measures the
fraction of fresh scenarios in which its solution stays feasible.
_Avoid_: quality measure (the method name), evaluation.

**Feasibility probability**:
The fraction of scenarios in which a solution satisfies `A·x - b ≤ 0`; the
robustness score. The one number scoring produces.
_Avoid_: robustness (that is the demos' display label for the same quantity).

### A solution's two stages

A solution is produced in two stages by two different owners; the type system
makes that explicit.

**Unscored Solution**:
What `MiniOrtoolsSolver` produces from one scenario — `solve_status` always, plus
`variable`/`constraint`/`objective_value` only on an optimal solve. It carries no
feasibility probability: the solver has no business producing one.
_Avoid_: raw solution, partial solution.

**Scored Solution**:
An Unscored Solution after Scoring has added its **feasibility probability**. The
only stage allowed to add that key.
_Avoid_: final solution, evaluated solution.

**score**:
The transform `(UnscoredSolution, probability) → ScoredSolution`. The single,
named place the feasibility probability enters a solution — replacing the silent
in-place mutation.
_Avoid_: fill, annotate, finalize.

**is_optimal**:
The accessor that answers "did this solve reach optimality" (`solve_status == 0`).
The one place that test lives, replacing the hand-rolled `solve_status == 0 and
"variable" in r` guards at the call sites.
_Avoid_: is_solved, is_valid.
