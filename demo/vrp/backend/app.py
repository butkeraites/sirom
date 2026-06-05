"""FastAPI backend for the SIROM robust-CVRP demo.

It owns the VRP-specific concerns — fetching VRP-REP datasets, parsing the
unified XML, subsampling to a solvable size, building the robust MILP, and
decoding routes — and delegates the *solving* to the SIROM HTTP API over the
container network. It never imports SIROM.

    GET  /vrp/datasets                  -> curated catalog
    GET  /vrp/instances/{slug}/{name}   -> parsed + subsampled preview
    POST /vrp/solve                     -> robustness/distance frontier of routes
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import cvrp
import vrprep

app = FastAPI(title="SIROM — Robust CVRP Demo (VRP-REP)")

# The browser calls this backend directly (a solve can take a minute or two,
# longer than a server-side proxy's timeout), so allow cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_K = 8


def _subsample_note(instance: vrprep.Instance, used: int) -> Optional[str]:
    if used >= instance.total_customers:
        return None
    return (
        f"Solving the depot + {used} of {instance.total_customers} customers "
        "(the ones nearest the depot). SIROM solves a full MILP across many "
        "scenarios, so the instance is reduced to a tractable size."
    )


class SolveBody(BaseModel):
    slug: str
    name: str
    mode: str = Field(default="demand", pattern="^(demand|travel_time)$")
    # A higher default α makes capacity/time bind in enough scenarios to spread
    # the frontier; at very low α the optimal routing is simply robust (one point).
    alpha: float = Field(default=0.4, ge=0.0, lt=1.0)
    k: int = Field(default=DEFAULT_K, ge=2, le=15)
    n_scenarios: int = Field(default=40, ge=10, le=200)
    # travel_time only: scales the synthesized per-route shift limit. Lower =
    # tighter = the time constraint bites harder (lower robustness).
    shift_factor: float = Field(default=1.1, ge=0.7, le=2.0)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/vrp/datasets", tags=["vrp"])
def datasets() -> dict:
    return {"datasets": vrprep.list_datasets()}


@app.get("/vrp/datasets/{slug}/instances", tags=["vrp"])
def dataset_instances(slug: str) -> dict:
    """List a dataset's instances (smallest first), loaded lazily from its zip."""
    try:
        return {"instances": vrprep.list_instances(slug)}
    except vrprep.DatasetError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/vrp/instances/{slug}/{name}", tags=["vrp"])
def instance_preview(slug: str, name: str, k: int = DEFAULT_K) -> dict:
    try:
        full = vrprep.load_instance(slug, name)
        reduced = vrprep.subsample(full, k)
    except vrprep.DatasetError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "instance": {
            "dataset": reduced.dataset,
            "name": reduced.name,
            "capacity": reduced.capacity,
            "vehicles": cvrp.vehicles_needed(reduced),
            "nodes": cvrp._nodes_payload(reduced),
            "total_customers": full.total_customers,
            "used_customers": reduced.n,
        },
        "subsample_note": _subsample_note(full, reduced.n),
    }


@app.post("/vrp/solve", tags=["vrp"])
def solve(body: SolveBody) -> dict:
    try:
        full = vrprep.load_instance(body.slug, body.name)
        reduced = vrprep.subsample(full, body.k)
    except vrprep.DatasetError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    note = _subsample_note(full, reduced.n)
    try:
        return cvrp.solve(
            reduced, body.mode, body.alpha, body.n_scenarios, note,
            shift_factor=body.shift_factor,
        )
    except cvrp.SolveError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
