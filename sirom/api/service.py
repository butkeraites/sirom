"""Run a SIROM problem end-to-end and shape the result for HTTP clients.

This is the single boundary between the HTTP layer and the algorithm. It builds
a :class:`~sirom.batch_solver.ProblemsBucket`, runs the four pipeline phases,
and post-processes the raw solution pool into a clean, non-dominated Pareto
frontier (objective value vs. feasibility probability) — the meaningful answer
a decision-maker wants. Keeping this self-contained means a job runner can call
``run_solve_job`` in a thread, a child process, or inline without changes.
"""

from __future__ import annotations

import contextlib
import io
import time
from typing import Any, Dict, List, Tuple, cast

from sirom.batch_solver import ProblemsBucket
from sirom.mini_ortools_solver import (
    ScoredSolution,
    feasibility,
    is_optimal,
    solver_available,
)

from .errors import SolveError, friendly_messages, has_errors
from .schemas import (
    RobustSolution,
    SolveRequest,
    SolveResponse,
    SolveSummary,
)


def _pareto_front(
    candidates: List[Tuple[float, float, List[float]]]
) -> List[Tuple[float, float, List[float]]]:
    """Return the non-dominated set from ``(objective, feasibility, vars)``.

    Minimizes objective, maximizes feasibility. A point is dominated if another
    is at least as good on both axes and strictly better on one. Exact-duplicate
    points are collapsed.
    """
    # Deduplicate on a rounded key to avoid float noise producing near-copies.
    unique: Dict[Tuple[float, float, Tuple[float, ...]], Tuple[float, float, List[float]]] = {}
    for obj, feas, variables in candidates:
        key = (round(obj, 9), round(feas, 9), tuple(round(v, 9) for v in variables))
        unique.setdefault(key, (obj, feas, variables))
    points = list(unique.values())

    front: List[Tuple[float, float, List[float]]] = []
    for i, (obj_i, feas_i, vars_i) in enumerate(points):
        dominated = False
        for j, (obj_j, feas_j, _) in enumerate(points):
            if i == j:
                continue
            if (
                obj_j <= obj_i
                and feas_j >= feas_i
                and (obj_j < obj_i or feas_j > feas_i)
            ):
                dominated = True
                break
        if not dominated:
            front.append((obj_i, feas_i, vars_i))

    # Most robust first, then cheapest objective.
    front.sort(key=lambda p: (-p[1], p[0]))
    return front


def solve_problem(request: SolveRequest) -> SolveResponse:
    """Run the full SIROM pipeline and return the robustness/cost frontier.

    Raises :class:`SolveError` with client-safe messages if the problem is
    rejected by the algorithm or fails mid-run.
    """
    opts = request.options
    started = time.time()
    log_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(log_buffer):
            if opts.solver is not None and not solver_available(opts.solver):
                raise SolveError(
                    [
                        f"Solver '{opts.solver}' is not available in this "
                        "OR-Tools build. Open-source backends (GLOP, CLP, PDLP, "
                        "SCIP, CBC) ship by default; commercial solvers (GUROBI, "
                        "CPLEX, XPRESS) require an OR-Tools build linked against "
                        "them and a valid license."
                    ]
                )
            bucket = ProblemsBucket(
                request.objective,
                request.lb_A,
                request.ub_A,
                request.lb_b,
                request.ub_b,
                number_of_scenarios=opts.number_of_scenarios,
                number_of_clusters=opts.clusters,
                integer_variables=request.integer_variables,
                solver_selection=opts.solver,
            )
            if has_errors(bucket.status):
                raise SolveError(friendly_messages(bucket.status))

            bucket.solve()
            bucket.cluster_and_selection()
            bucket.solve_cluster_tree()
            bucket.apply_quality_measure(number_of_scenarios=opts.quality_scenarios)
    except SolveError:
        raise
    except Exception as exc:  # noqa: BLE001 - convert any run failure to safe text
        raise SolveError(
            [
                "The solver could not complete this problem. It may be "
                "infeasible, unbounded, or degenerate. Try tighter coefficient "
                f"bounds or fewer scenarios. ({type(exc).__name__})"
            ]
        )

    # Every result is scored by this point (apply_quality_measure has run).
    results = cast(List[ScoredSolution], bucket.results)
    n_scenarios = opts.number_of_scenarios
    # The first n_scenarios results are the scenario solves; the cluster
    # re-solves follow. Count optimal scenarios from the cheap order-stable
    # prefix; the cluster node count comes from the tree itself (below), not
    # from slicing the results list by append order.
    scenarios_optimal = sum(1 for r in results[:n_scenarios] if is_optimal(r))

    candidates: List[Tuple[float, float, List[float]]] = []
    for r in results:
        # Every result is scored by this point; optimal ones carry the vector.
        if is_optimal(r):
            candidates.append(
                (
                    float(r["objective_value"]),
                    feasibility(r),
                    [float(v) for v in r["variable"]],
                )
            )

    front = _pareto_front(candidates)
    solutions = [
        RobustSolution(
            variables=variables,
            objective_value=obj,
            feasibility_probability=feas,
        )
        for obj, feas, variables in front
    ]

    warnings: List[str] = []
    if scenarios_optimal < n_scenarios:
        warnings.append(
            f"{n_scenarios - scenarios_optimal} of {n_scenarios} sampled "
            "scenarios were not solved to optimality (likely infeasible) and "
            "were excluded from the frontier."
        )
    if not solutions:
        warnings.append(
            "No optimal solutions were found; the frontier is empty. The "
            "problem may be infeasible across the sampled uncertainty set."
        )

    summary = SolveSummary(
        scenarios_solved=n_scenarios,
        scenarios_optimal=scenarios_optimal,
        cluster_nodes=(
            len(bucket.cluster_tree.get_all_nodes())
            if getattr(bucket, "cluster_tree", None) is not None
            else 0
        ),
        candidate_solutions=len(solutions),
        best_feasibility=max((s.feasibility_probability for s in solutions), default=0.0),
        runtime_seconds=round(time.time() - started, 4),
    )

    return SolveResponse(
        solutions=solutions,
        summary=summary,
        warnings=warnings,
        log=log_buffer.getvalue().splitlines() if opts.include_log else None,
    )


def run_solve_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process-pool entry point: take a plain dict, return a plain dict.

    Kept picklable (dict in, dict out) so it can run in a ``ProcessPoolExecutor``
    child. Re-validates the payload (cheap) to apply schema defaults.
    """
    request = SolveRequest(**payload)
    return solve_problem(request).model_dump()
