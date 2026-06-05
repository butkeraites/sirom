# ADR-0001: Retain OptimizationProblem + Coefficient separation

- Status: Accepted (2026-06-05)
- Context: architecture review finding #5 (refuted on adversarial verification)

## Decision

Keep `Coefficient` as a separate read-only holder inside
`sirom/optimization_problem.py`. Do **not** inline it into `OptimizationProblem`,
and do **not** drop the `pandas.DataFrame` wrapping of `c` / `A` / `b`.

## Context

A review candidate proposed collapsing `Coefficient` into `OptimizationProblem`
and validating directly on numpy arrays, claiming the `DataFrame` was a redundant
intermediate that gets discarded.

## Why this was refused

- **The DataFrame is the canonical *stored* form, not a discarded intermediate.**
  `OptimizationProblem.__coefficient_validation` returns `pd.DataFrame(coefficient)`
  (`optimization_problem.py:63`) and `__set_coefficient` stores those DataFrames in
  `Coefficient` (`:53`). `__dimension_validation` reads `.shape` straight off them
  (`:77–82`).
- **It is load-bearing for the validation it enables.** The inputs are 1-D lists
  (`c=[3,1]`, `b=[2,1,2,0,0]`). `pd.DataFrame([2,1,2,0,0]).shape` is `(5, 1)`, so the
  `rows, cols = …shape` unpacking works; `np.asarray([2,1,2,0,0]).shape` is `(5,)`
  and that unpack raises. "Validate directly on numpy" would have to re-introduce
  column-vector reshaping plus the `try/except` that emits the exact tested error
  strings — so the indirection the proposal removes is the thing the tests rely on.
- **House style.** `Coefficient` deliberately mirrors the parallel `Coefficients`
  class in `batch_solver.py`. Keeping them paired keeps the two coefficient bundles
  symmetric.

The change fails the deletion test: it would move complexity (re-spelling the
DataFrame normalisation in numpy) rather than remove it, while risking the tested
1-D inputs.

## Consequence

`Coefficient` stays a thin holder. If it is ever revisited, the only defensible move
is inlining the holder while **keeping** the DataFrame normalisation — which removes
little real depth.
