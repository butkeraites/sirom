"""FastAPI application exposing the SIROM solver over HTTP.

Endpoints:

* ``GET  /``            -> redirect to the interactive docs
* ``GET  /health``      -> liveness probe
* ``GET  /example``     -> a ready-to-POST sample problem
* ``POST /solve``       -> enqueue a solve, returns 202 + job id
* ``GET  /jobs/{id}``   -> poll a job's status/result
* ``GET  /jobs``        -> list submitted jobs

The interactive docs at ``/docs`` are the intended starting point: they render
the request schema with field-level explanations and a prefilled working
example, so callers never need to read the algorithm's source.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from .examples import EXAMPLE_PROBLEM
from .jobs import JobManager
from .schemas import (
    JobCreatedResponse,
    JobStatus,
    JobStatusResponse,
    SolveRequest,
    SolveResponse,
)

DESCRIPTION = """
**SIROM** solves linear programs whose coefficients are *uncertain* — given as
intervals `[lb, ub]` instead of fixed numbers — and returns a set of candidate
solutions that trade objective value against robustness.

It solves `min c·x  s.t.  A·x ≤ b`, where each scenario draws `A` from
`[lb_A, ub_A]` and `b` from `[lb_b, ub_b]`. **All variables are non-negative
(x ≥ 0).** You don't need to understand the internals (scenario sampling,
clustering, per-cluster re-solving): POST a problem, poll for the result, and
read back a Pareto frontier of `objective_value` vs. `feasibility_probability`.

**Quickstart:** `GET /example` → paste the body into `POST /solve` (use *Try it
out* below) → poll `GET /jobs/{job_id}` until `succeeded`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jobs = JobManager()
    try:
        yield
    finally:
        app.state.jobs.shutdown()


app = FastAPI(
    title="SIROM API",
    version="0.1.0",
    description=DESCRIPTION,
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"], summary="Liveness probe")
def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/example",
    tags=["solve"],
    summary="A ready-to-POST sample problem",
    response_model=SolveRequest,
)
def example() -> dict:
    return EXAMPLE_PROBLEM


@app.post(
    "/solve",
    tags=["solve"],
    summary="Submit a problem to solve",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreatedResponse,
)
def solve(problem: SolveRequest, request: Request) -> JobCreatedResponse:
    jobs: JobManager = request.app.state.jobs
    job_id = jobs.submit(problem.model_dump())
    current = jobs.get(job_id) or {"status": JobStatus.pending}
    return JobCreatedResponse(
        job_id=job_id,
        status=current["status"],
        result_url=str(request.url_for("get_job", job_id=job_id)),
    )


@app.get(
    "/jobs/{job_id}",
    tags=["solve"],
    summary="Poll a job's status and result",
    response_model=JobStatusResponse,
    responses={404: {"description": "Unknown job id"}},
)
def get_job(job_id: str, request: Request) -> JSONResponse:
    jobs: JobManager = request.app.state.jobs
    record = jobs.get(job_id)
    if record is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"No job with id {job_id!r}."},
        )
    result = record["result"]
    response = JobStatusResponse(
        job_id=job_id,
        status=record["status"],
        result=SolveResponse(**result) if result is not None else None,
        errors=record["errors"],
    )
    return JSONResponse(content=response.model_dump(exclude_none=True))


@app.get("/jobs", tags=["solve"], summary="List submitted jobs")
def list_jobs(request: Request) -> list:
    jobs: JobManager = request.app.state.jobs
    return jobs.list_jobs()
