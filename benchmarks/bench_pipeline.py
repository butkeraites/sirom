"""Wall-clock benchmark for the SIROM pipeline, phase by phase.

Standalone script (not collected by pytest). Builds a feasible, bounded
interval LP at several sizes and times each pipeline phase separately so we can
see where time goes and compare before/after an optimization.

Usage:
    python benchmarks/bench_pipeline.py            # default grid
    python benchmarks/bench_pipeline.py --full     # larger grid (slower)

Each row reports seconds for: generate (scenario coefficients, done in
__init__), solve (N scenario LPs), cluster (KMeans tree), tree (per-node
re-solve), quality (feasibility scoring over M scenarios), and total.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
from typing import List, Tuple

import numpy as np

from sirom.batch_solver import ProblemsBucket


def make_problem(
    n_vars: int, n_constraints: int, seed: int = 0
) -> Tuple[list, list, list, list, list]:
    """A feasible, bounded interval LP of the requested size.

    The first ``n_vars`` constraint rows are box upper bounds (so every solve is
    bounded), the rest are random non-negative coupling rows. The objective has
    mixed signs so the optimum is a non-trivial vertex. ``x = 0`` is always
    feasible, so every sampled scenario solves to OPTIMAL.
    """
    if n_constraints < n_vars:
        raise ValueError("benchmark expects n_constraints >= n_vars")
    rng = np.random.default_rng(seed)

    base_A = np.zeros((n_constraints, n_vars))
    for j in range(n_vars):
        base_A[j, j] = 1.0  # box: x_j <= u_j
    for i in range(n_vars, n_constraints):
        base_A[i] = rng.uniform(0.1, 1.0, size=n_vars)  # coupling rows

    u = rng.uniform(1.0, 5.0, size=n_vars)
    base_b = np.empty(n_constraints)
    base_b[:n_vars] = u
    for i in range(n_vars, n_constraints):
        base_b[i] = float(base_A[i] @ u)  # feasible at x = u

    c = rng.uniform(-1.0, 1.0, size=n_vars)
    lb_A, ub_A = base_A * 0.95, base_A * 1.05
    lb_b, ub_b = base_b * 0.95, base_b * 1.05
    return (
        c.tolist(),
        lb_A.tolist(),
        ub_A.tolist(),
        lb_b.tolist(),
        ub_b.tolist(),
    )


def time_pipeline(n_vars: int, n_con: int, n_scenarios: int, m_quality: int) -> dict:
    c, lb_A, ub_A, lb_b, ub_b = make_problem(n_vars, n_con)
    timings = {}
    with contextlib.redirect_stdout(io.StringIO()):
        t = time.perf_counter()
        bucket = ProblemsBucket(
            c, lb_A, ub_A, lb_b, ub_b, number_of_scenarios=n_scenarios
        )
        timings["generate"] = time.perf_counter() - t

        t = time.perf_counter()
        bucket.solve()
        timings["solve"] = time.perf_counter() - t

        t = time.perf_counter()
        bucket.cluster_and_selection()
        timings["cluster"] = time.perf_counter() - t

        t = time.perf_counter()
        bucket.solve_cluster_tree()
        timings["tree"] = time.perf_counter() - t

        t = time.perf_counter()
        bucket.apply_quality_measure(number_of_scenarios=m_quality)
        timings["quality"] = time.perf_counter() - t

    timings["total"] = sum(timings.values())
    return timings


DEFAULT_GRID: List[Tuple[int, int, int, int]] = [
    (2, 5, 100, 100),
    (20, 50, 100, 100),
    (50, 200, 100, 100),
    (20, 50, 500, 500),
]

FULL_GRID: List[Tuple[int, int, int, int]] = DEFAULT_GRID + [
    (50, 200, 500, 500),
    (50, 200, 1000, 1000),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the larger grid")
    args = parser.parse_args()
    grid = FULL_GRID if args.full else DEFAULT_GRID

    cols = ["vars", "con", "N", "M", "generate", "solve", "cluster", "tree", "quality", "total"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for n_vars, n_con, n_scen, m_qual in grid:
        t = time_pipeline(n_vars, n_con, n_scen, m_qual)
        row = [n_vars, n_con, n_scen, m_qual]
        cells = [str(v) for v in row] + [
            f"{t[k]:.3f}" for k in ["generate", "solve", "cluster", "tree", "quality", "total"]
        ]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
