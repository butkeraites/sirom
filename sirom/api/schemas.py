"""Pydantic request/response models for the SIROM API.

Requests speak the user's domain language (an objective vector and constraint
bounds); the algorithm's knobs live in an optional ``options`` block with safe
defaults. Validators reject malformed or oversized problems *before* any solve
runs, so hostile or mistaken input becomes a clean ``422`` instead of an OOM or
a hang.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from .examples import EXAMPLE_PROBLEM


def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


# Safety caps (env-overridable). They bound the work a single request can
# trigger: the solve phase allocates and solves roughly
# ``number_of_scenarios`` LPs of size ``constraints x variables``.
MAX_VARS = _env_int("SIROM_MAX_VARS", 200)
MAX_CONSTRAINTS = _env_int("SIROM_MAX_CONSTRAINTS", 500)
MAX_SCENARIOS = _env_int("SIROM_MAX_SCENARIOS", 2000)
MAX_CLUSTERS = _env_int("SIROM_MAX_CLUSTERS", 50)
# Multiplicative guard for problems that are individually within limits but
# huge in combination (scenarios x variables x constraints).
CELL_BUDGET = _env_int("SIROM_CELL_BUDGET", 5_000_000)


class SolveOptions(BaseModel):
    """Advanced tuning knobs. Defaults match the published SIROM settings."""

    number_of_scenarios: int = Field(
        default=100,
        ge=1,
        le=MAX_SCENARIOS,
        description="How many uncertainty scenarios to sample and solve. "
        "More scenarios = better coverage but slower.",
    )
    quality_scenarios: int = Field(
        default=100,
        ge=1,
        le=MAX_SCENARIOS,
        description="How many fresh scenarios to draw when scoring each "
        "candidate solution's feasibility probability.",
    )
    clusters: int = Field(
        default=3,
        ge=2,
        le=MAX_CLUSTERS,
        description="Branching factor of the internal clustering heuristic "
        "(advanced; the published default is 3).",
    )
    include_log: bool = Field(
        default=False,
        description="If true, the run's internal timing log is returned with "
        "the result.",
    )


class SolveRequest(BaseModel):
    """A robust linear program with interval-valued coefficients.

    Solves ``min c.x`` subject to ``A.x <= b`` where each scenario draws ``A``
    uniformly from ``[lb_A, ub_A]`` and ``b`` from ``[lb_b, ub_b]``. **All
    decision variables are constrained to be non-negative (x >= 0).**
    """

    objective: List[float] = Field(
        ...,
        min_length=1,
        description="Cost coefficients c in `min c.x`. One entry per decision "
        "variable.",
    )
    lb_A: List[List[float]] = Field(
        ...,
        min_length=1,
        description="Lower bound of the constraint matrix. Rows = constraints, "
        "columns = variables.",
    )
    ub_A: List[List[float]] = Field(
        ...,
        min_length=1,
        description="Upper bound of the constraint matrix (same shape as lb_A).",
    )
    lb_b: List[float] = Field(
        ..., min_length=1, description="Lower bound of the right-hand-side vector."
    )
    ub_b: List[float] = Field(
        ..., min_length=1, description="Upper bound of the right-hand-side vector."
    )
    options: SolveOptions = Field(default_factory=SolveOptions)

    model_config = {"json_schema_extra": {"examples": [EXAMPLE_PROBLEM]}}

    @model_validator(mode="after")
    def _check_shapes_and_bounds(self) -> "SolveRequest":
        n_vars = len(self.objective)
        if n_vars > MAX_VARS:
            raise ValueError(
                f"Too many variables: {n_vars} > {MAX_VARS} (SIROM_MAX_VARS)."
            )

        n_constraints = len(self.lb_A)
        if n_constraints > MAX_CONSTRAINTS:
            raise ValueError(
                f"Too many constraints: {n_constraints} > {MAX_CONSTRAINTS} "
                "(SIROM_MAX_CONSTRAINTS)."
            )

        # Cross-field shape consistency.
        if len(self.ub_A) != n_constraints:
            raise ValueError("lb_A and ub_A must have the same number of rows.")
        if not (len(self.lb_b) == len(self.ub_b) == n_constraints):
            raise ValueError(
                "lb_b and ub_b must have one entry per constraint row "
                f"(expected {n_constraints})."
            )
        for label, matrix in (("lb_A", self.lb_A), ("ub_A", self.ub_A)):
            for i, row in enumerate(matrix):
                if len(row) != n_vars:
                    raise ValueError(
                        f"{label} row {i} has {len(row)} entries; expected "
                        f"{n_vars} (one per variable)."
                    )

        # Interval orientation: lower bound must not exceed upper bound.
        for i, (lo_row, hi_row) in enumerate(zip(self.lb_A, self.ub_A)):
            for j, (lo, hi) in enumerate(zip(lo_row, hi_row)):
                if lo > hi:
                    raise ValueError(
                        f"lb_A[{i}][{j}] ({lo}) exceeds ub_A[{i}][{j}] ({hi}); "
                        "lower bounds must not exceed upper bounds."
                    )
        for i, (lo, hi) in enumerate(zip(self.lb_b, self.ub_b)):
            if lo > hi:
                raise ValueError(
                    f"lb_b[{i}] ({lo}) exceeds ub_b[{i}] ({hi}); lower bounds "
                    "must not exceed upper bounds."
                )

        # Multiplicative work guard.
        cells = self.options.number_of_scenarios * n_vars * n_constraints
        if cells > CELL_BUDGET:
            raise ValueError(
                f"Problem too large: number_of_scenarios x variables x "
                f"constraints = {cells} exceeds the budget of {CELL_BUDGET} "
                "(SIROM_CELL_BUDGET). Reduce scenarios or problem size."
            )
        return self


class RobustSolution(BaseModel):
    """One Pareto-optimal candidate: a decision vector and its robustness."""

    variables: List[float] = Field(
        ..., description="Decision vector x* (always non-negative)."
    )
    objective_value: float = Field(..., description="Objective value c.x*.")
    feasibility_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability the solution stays feasible across "
        "random realizations of the uncertain coefficients (the robustness "
        "score; higher is more robust).",
    )


class SolveSummary(BaseModel):
    """At-a-glance run statistics."""

    scenarios_solved: int
    scenarios_optimal: int
    cluster_nodes: int
    candidate_solutions: int = Field(
        ..., description="Solutions on the returned Pareto frontier."
    )
    best_feasibility: float = Field(
        ..., description="Highest feasibility probability among candidates."
    )
    runtime_seconds: float


class SolveResponse(BaseModel):
    """The result of a solved job: the robustness/cost trade-off frontier."""

    solutions: List[RobustSolution] = Field(
        ...,
        description="Non-dominated trade-off between objective value and "
        "feasibility probability, sorted by robustness then objective.",
    )
    summary: SolveSummary
    warnings: List[str] = Field(default_factory=list)
    log: Optional[List[str]] = Field(
        default=None, description="Internal run log (only if include_log was set)."
    )


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobCreatedResponse(BaseModel):
    """Returned by ``POST /solve`` (HTTP 202)."""

    job_id: str
    status: JobStatus
    result_url: str = Field(..., description="Poll this URL for the result.")


class JobStatusResponse(BaseModel):
    """Returned by ``GET /jobs/{job_id}``."""

    job_id: str
    status: JobStatus
    result: Optional[SolveResponse] = Field(
        default=None, description="Present once status is `succeeded`."
    )
    errors: Optional[List[str]] = Field(
        default=None, description="Present once status is `failed`."
    )
