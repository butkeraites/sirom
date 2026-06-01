"""Tests for the FastAPI layer.

Runs jobs with the synchronous ``inline`` executor so a POST is finished by the
time the test polls for it, and uses tiny scenario counts to stay fast.
"""

import os

# Must be set before the app's lifespan builds the JobManager.
os.environ["SIROM_EXECUTOR"] = "inline"

import pytest
from fastapi.testclient import TestClient

from sirom.api.app import app
from sirom.api.examples import EXAMPLE_PROBLEM

# A fast, known-good problem (small scenario counts).
GOOD_PROBLEM = {
    **EXAMPLE_PROBLEM,
    "options": {"number_of_scenarios": 6, "quality_scenarios": 6, "clusters": 3},
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _solve(client, body):
    """POST a problem and return the polled job record (inline => done)."""
    created = client.post("/solve", json=body)
    assert created.status_code == 202, created.text
    job_id = created.json()["job_id"]
    polled = client.get(f"/jobs/{job_id}")
    assert polled.status_code == 200, polled.text
    return polled.json()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_redirects_to_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


def test_example_is_a_valid_request(client):
    resp = client.get("/example")
    assert resp.status_code == 200
    # The example must itself be accepted by /solve's validation.
    assert client.post("/solve", json=resp.json()).status_code == 202


def test_solve_example_succeeds(client):
    job = _solve(client, GOOD_PROBLEM)
    assert job["status"] == "succeeded"
    result = job["result"]
    assert result["solutions"], "expected at least one candidate solution"
    for sol in result["solutions"]:
        assert 0.0 <= sol["feasibility_probability"] <= 1.0
        assert len(sol["variables"]) == len(GOOD_PROBLEM["objective"])
        assert all(v >= -1e-9 for v in sol["variables"])  # x >= 0
    assert result["summary"]["candidate_solutions"] == len(result["solutions"])


def test_returned_frontier_is_non_dominated(client):
    job = _solve(client, GOOD_PROBLEM)
    sols = job["result"]["solutions"]
    for a in sols:
        for b in sols:
            if a is b:
                continue
            # b must not dominate a (>= feasibility and <= objective, strict once)
            dominates = (
                b["objective_value"] <= a["objective_value"]
                and b["feasibility_probability"] >= a["feasibility_probability"]
                and (
                    b["objective_value"] < a["objective_value"]
                    or b["feasibility_probability"] > a["feasibility_probability"]
                )
            )
            assert not dominates


def test_dimension_mismatch_returns_422(client):
    bad = {**GOOD_PROBLEM, "lb_b": [2, 1, 2, 0]}  # one short of the 5 rows
    assert client.post("/solve", json=bad).status_code == 422


def test_over_cap_scenarios_returns_422(client):
    bad = {**GOOD_PROBLEM, "options": {"number_of_scenarios": 10_000_000}}
    assert client.post("/solve", json=bad).status_code == 422


def test_inverted_interval_returns_422(client):
    bad = {**GOOD_PROBLEM, "lb_b": [9, 9, 9, 9, 9], "ub_b": [3, 2, 3, 0, 0]}
    assert client.post("/solve", json=bad).status_code == 422


def test_unknown_job_returns_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_infeasible_problem_does_not_crash(client):
    # x <= 0.5 AND x >= 1 simultaneously => every scenario is infeasible.
    infeasible = {
        "objective": [1.0],
        "lb_A": [[1], [-1]],
        "ub_A": [[1], [-1]],
        "lb_b": [0.5, -1.0],
        "ub_b": [0.5, -1.0],
        "options": {"number_of_scenarios": 5, "quality_scenarios": 5},
    }
    job = _solve(client, infeasible)
    assert job["status"] in ("failed", "succeeded")  # never a 500
    if job["status"] == "failed":
        assert job["errors"]
    else:
        assert job["result"]["solutions"] == []
