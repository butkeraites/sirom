# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SIROM = **Simulation-based Robust Optimization Method**. Given a linear program whose coefficients are uncertain — specified as intervals `[lb, ub]` rather than fixed values — it samples scenarios across those intervals, solves each, clusters the solutions, and re-solves per cluster to produce a set of candidate robust solutions, each annotated with a feasibility probability. The method follows the paper in `1-s2.0-S0957417422006947-main.pdf` (present in the working tree but untracked / not committed).

The canonical problem form is `min c·x  s.t.  A·x - b <= 0`, with `A` drawn from `[lb_A, ub_A]` and `b` from `[lb_b, ub_b]`.

## Commands

```bash
pip install -r requirements.txt        # pinned versions; ortools provides the SCIP solver
pip install -e .                        # install the sirom package (editable)

pytest                                  # run all tests
pytest tests/test_batch_solver.py       # run one test file
pytest tests/test_mini_ortools_solver.py::test_mini_ortools_solver_solution_objective_value  # single test

python -m sirom.main                    # run the full end-to-end pipeline (see below)
mypy sirom                              # type check (mypy 1.8 is pinned; code is annotated)
black sirom tests                       # format
```

There is no Makefile, lint config, or CI; tooling is invoked directly. `mypy`/`black` config lives only as pinned deps — no `[tool.*]` sections in `pyproject.toml`.

## Architecture

The pipeline is a four-stage sequence, orchestrated by `ProblemsBucket` in `sirom/batch_solver.py`. `sirom/main.py` shows the canonical call order:

```
ProblemsBucket(c, lb_A, ub_A, lb_b, ub_b, number_of_scenarios)
  .solve()                  # 1. sample N scenarios, solve each LP
  .cluster_and_selection()  # 2. KMeans-cluster the solution points into a tree
  .solve_cluster_tree()     # 3. re-solve, combining each leaf's scenarios
  .apply_quality_measure()  # 4. score each solution's feasibility on fresh scenarios
```

Module responsibilities (each is a self-validating class):

- **`batch_solver.py`** — the orchestrator. `Coefficients` holds the interval data. `ProblemsBucket.__init__` runs a validation chain that builds the scenario set: it Latin-Hypercube-samples deltas in `[0,1]` (via `smt.sampling_methods.LHS`) and interpolates `lb + (ub - lb)·delta` to materialize `number_of_scenarios` concrete `A`/`b` matrices. `cluster_and_selection` builds a `ClusterTree` from solution points `[objective_value] + constraint_slacks`, recursively splitting each splittable node into 3 KMeans clusters; `close_nodes` marks only the node with the most points and the node with the highest WCSS for further splitting (the selection heuristic). `apply_quality_measure` re-samples scenarios and sets `feasibility_probability` = fraction of scenarios where the solution stays feasible.
- **`optimization_problem.py`** — `OptimizationProblem` wraps a single concrete `(c, A, b)` and validates dimensions. Pure data + validation; does not solve.
- **`mini_ortools_solver.py`** — `MiniOrtoolsSolver` translates an `OptimizationProblem` into OR-Tools (`pywraplp`, SCIP by default), solves, and returns a `Solution` TypedDict (`variable`, `constraint`, `objective_value`, `solve_status`, `feasibility_probability`, `log`). Variables are non-negative (`NumVar(0, inf)`).
- **`cluster_tree.py`** — `ClusterTree` is a UUID-keyed dict of nodes (`RootData`/`Leaf` TypedDicts). Each node carries its point coordinates, point ids, count, and WCSS. The `replicate` flag drives whether a node gets split further.

### The status-string convention (important)

There are no exceptions for domain errors. Every class accumulates a `self.status: list[str]` of messages prefixed `[OK]` / `[ERROR]` / `[INFO]`. The next step in a validation chain gates on `any("[ERROR]" in s for s in self.status)` and short-circuits if a prior step failed. Tests assert on these exact strings (e.g. `"[ERROR] Optimization batch creation failed" in opt_problem.status`). When adding a validation step, follow this pattern and keep the message text stable — changing a string breaks the corresponding test.

### Access style: TypedDict subscript vs. property — don't mix them

Two distinct conventions coexist, and they are not interchangeable:

- `Solution` (in `mini_ortools_solver.py`) and the `Leaf`/`RootData`/`SolvedLeaf` types (in `cluster_tree.py`) are `TypedDict`s — access them by **subscript**: `solution["objective_value"]`, `tree_nodes[node]["problem"]`.
- `Coefficient`/`Coefficients` (in `optimization_problem.py` / `batch_solver.py`) expose data via `@property` — access them by **attribute**: `coefficient.constraint`, `coefficient.scenarios_constraint`.

Mixing the two (e.g. `coefficient["constraint"]` or `solution.objective_value`) raises `TypeError`/`AttributeError` at runtime, not at import — and several such mismatches previously made the pipeline crash mid-run. When you touch one of these objects, confirm which kind it is first.
