"""Compare SIROM Pareto frontiers across code versions or configurations.

The algorithm samples scenarios randomly, so two runs of the same code already
differ. To compare a *change* (a code version, a solver, a clustering tweak)
fairly, this tool fixes the scenarios once and replays them, so any frontier
difference is attributable to the change alone.

Workflow:

    # 1. Capture a fixed problem + scenarios once.
    python benchmarks/frontier_diff.py prep --dir /tmp/fd

    # 2. Run on the current code -> a frontier JSON.
    python benchmarks/frontier_diff.py run --dir /tmp/fd --out /tmp/fd/after.json

    #    ... git checkout <other-commit> (or change a setting) ...
    python benchmarks/frontier_diff.py run --dir /tmp/fd --out /tmp/fd/before.json

    # 3. Compare the two frontiers.
    python benchmarks/frontier_diff.py diff /tmp/fd/before.json /tmp/fd/after.json

A frontier file is a JSON list of ``[objective, feasibility]`` non-dominated
points (objective minimized, feasibility maximized).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from typing import List, Tuple

import numpy as np

from bench_pipeline import make_problem  # sibling module in benchmarks/
from sirom.batch_solver import ProblemsBucket

Point = Tuple[float, float]


def _paths(directory: str) -> dict:
    return {
        "problem": os.path.join(directory, "problem.json"),
        "solve_A": os.path.join(directory, "solve_A.npy"),
        "solve_b": os.path.join(directory, "solve_b.npy"),
        "qual_A": os.path.join(directory, "qual_A.npy"),
        "qual_b": os.path.join(directory, "qual_b.npy"),
    }


def _scenarios(bucket: ProblemsBucket) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(bucket.coefficient.scenarios_constraint, dtype=float),
        np.asarray(bucket.coefficient.scenarios_rhs, dtype=float),
    )


def cmd_prep(args: argparse.Namespace) -> None:
    os.makedirs(args.dir, exist_ok=True)
    p = _paths(args.dir)
    c, lb_A, ub_A, lb_b, ub_b = make_problem(args.vars, args.con, seed=args.seed)
    with open(p["problem"], "w") as fh:
        json.dump(
            {"c": c, "lb_A": lb_A, "ub_A": ub_A, "lb_b": lb_b, "ub_b": ub_b,
             "scenarios": args.scenarios, "quality": args.quality},
            fh,
        )
    with contextlib.redirect_stdout(io.StringIO()):
        solve = ProblemsBucket(c, lb_A, ub_A, lb_b, ub_b, number_of_scenarios=args.scenarios)
        qual = ProblemsBucket(c, lb_A, ub_A, lb_b, ub_b, number_of_scenarios=args.quality)
    sA, sb = _scenarios(solve)
    qA, qb = _scenarios(qual)
    np.save(p["solve_A"], sA)
    np.save(p["solve_b"], sb)
    np.save(p["qual_A"], qA)
    np.save(p["qual_b"], qb)
    print(
        f"prepared {args.vars}x{args.con} problem, {args.scenarios} solve / "
        f"{args.quality} quality scenarios in {args.dir}"
    )


def _pareto(points: List[Point]) -> List[Point]:
    rounded = [(round(o, 6), round(f, 6)) for o, f in points]
    front = []
    for i, (oi, fi) in enumerate(rounded):
        dominated = any(
            oj <= oi and fj >= fi and (oj < oi or fj > fi)
            for j, (oj, fj) in enumerate(rounded)
            if j != i
        )
        if not dominated:
            front.append((oi, fi))
    return sorted(set(front), key=lambda p: (-p[1], p[0]))


def cmd_run(args: argparse.Namespace) -> None:
    p = _paths(args.dir)
    with open(p["problem"]) as fh:
        problem = json.load(fh)

    with contextlib.redirect_stdout(io.StringIO()):
        bucket = ProblemsBucket(
            problem["c"], problem["lb_A"], problem["ub_A"],
            problem["lb_b"], problem["ub_b"],
            number_of_scenarios=problem["scenarios"],
        )
        bucket.coefficient.scenarios_constraint = np.load(p["solve_A"])
        bucket.coefficient.scenarios_rhs = np.load(p["solve_b"])
        bucket.solve()
        bucket.cluster_and_selection()
        bucket.solve_cluster_tree()

    qual_A = np.load(p["qual_A"])  # (M, n_con, n_var)
    qual_b = np.load(p["qual_b"]).reshape(qual_A.shape[0], qual_A.shape[1])
    points: List[Point] = []
    for result in bucket.results:
        if result.get("solve_status") == 0 and "variable" in result:
            x = np.asarray(result["variable"], dtype=float)
            feasibility = float(np.mean((qual_A @ x - qual_b).max(axis=1) <= 0.0))
            points.append((float(result["objective_value"]), feasibility))

    frontier = _pareto(points)
    with open(args.out, "w") as fh:
        json.dump([list(point) for point in frontier], fh)
    print(f"wrote {len(frontier)}-point frontier to {args.out}")


def cmd_diff(args: argparse.Namespace) -> None:
    before = [tuple(p) for p in json.load(open(args.before))]
    after = [tuple(p) for p in json.load(open(args.after))]

    def describe(name: str, front: List[Point]) -> None:
        objs = [o for o, _ in front]
        feas = [f for _, f in front]
        print(
            f"{name}: {len(front)} points | objective "
            f"[{min(objs):.3f}, {max(objs):.3f}] | feasibility "
            f"[{min(feas):.3f}, {max(feas):.3f}]"
        )

    describe("before", before)
    describe("after ", after)
    shared = set(before) & set(after)
    print(f"identical points: {len(shared)} (before {len(before)}, after {len(after)})")

    def best_objective_at(front: List[Point], threshold: float):
        candidates = [o for o, f in front if f >= threshold]
        return round(min(candidates), 6) if candidates else None

    print("best objective at feasibility >= threshold:")
    for threshold in (0.9, 0.8, 0.5, 0.2):
        print(
            f"  >= {threshold}: before={best_objective_at(before, threshold)}  "
            f"after={best_objective_at(after, threshold)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prep", help="capture a fixed problem + scenarios")
    prep.add_argument("--dir", default="/tmp/sirom_frontier")
    prep.add_argument("--vars", type=int, default=6)
    prep.add_argument("--con", type=int, default=15)
    prep.add_argument("--scenarios", type=int, default=120)
    prep.add_argument("--quality", type=int, default=300)
    prep.add_argument("--seed", type=int, default=3)
    prep.set_defaults(func=cmd_prep)

    run = sub.add_parser("run", help="compute the frontier on the current code")
    run.add_argument("--dir", default="/tmp/sirom_frontier")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    diff = sub.add_parser("diff", help="compare two frontier JSON files")
    diff.add_argument("before")
    diff.add_argument("after")
    diff.set_defaults(func=cmd_diff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
