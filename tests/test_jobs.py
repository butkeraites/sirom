"""Tests for the job stores and the store-agnostic JobManager.

The Redis path is exercised against an in-process fakeredis client, so no
server is required.
"""

import os

import fakeredis
import pytest

from sirom.api.examples import EXAMPLE_PROBLEM
from sirom.api.jobs import (
    InMemoryJobStore,
    JobManager,
    RedisJobStore,
    build_store_from_env,
)
from sirom.api.schemas import SolveRequest

# Small, fast, valid payload (as POSTed and dumped through the schema).
PAYLOAD = SolveRequest(
    **{
        **EXAMPLE_PROBLEM,
        "options": {"number_of_scenarios": 6, "quality_scenarios": 6, "clusters": 3},
    }
).model_dump()


def _redis_store():
    return RedisJobStore(client=fakeredis.FakeStrictRedis(decode_responses=True))


@pytest.fixture(params=["memory", "redis"])
def store(request):
    return InMemoryJobStore() if request.param == "memory" else _redis_store()


def test_store_lifecycle(store):
    store.create("job1")
    assert store.fetch("job1")["status"] == "pending"

    store.record_success("job1", {"solutions": []})
    rec = store.fetch("job1")
    assert rec["status"] == "succeeded"
    assert rec["result"] == {"solutions": []}
    assert rec["errors"] is None


def test_store_failure(store):
    store.create("job2")
    store.record_failure("job2", ["boom"])
    rec = store.fetch("job2")
    assert rec["status"] == "failed"
    assert rec["errors"] == ["boom"]
    assert rec["result"] is None


def test_store_unknown_is_none(store):
    assert store.fetch("nope") is None


def test_store_list_ids(store):
    store.create("a")
    store.create("b")
    assert set(store.list_ids()) == {"a", "b"}


def test_in_memory_eviction():
    store = InMemoryJobStore(max_jobs=2)
    store.create("a")
    store.create("b")
    store.create("c")  # should evict the oldest ("a")
    assert store.fetch("a") is None
    assert set(store.list_ids()) == {"b", "c"}


@pytest.fixture(params=["memory", "redis"])
def manager(request):
    backing = InMemoryJobStore() if request.param == "memory" else _redis_store()
    mgr = JobManager(executor_mode="inline", store=backing)
    yield mgr
    mgr.shutdown()


def test_manager_solves_and_records(manager):
    job_id = manager.submit(PAYLOAD)
    record = manager.get(job_id)
    assert record["status"] == "succeeded"
    assert record["result"]["solutions"]
    assert {"job_id": job_id, "status": "succeeded"} in manager.list_jobs()


def test_manager_unknown_job(manager):
    assert manager.get("missing") is None


def test_build_store_from_env(monkeypatch):
    monkeypatch.delenv("SIROM_JOB_STORE", raising=False)
    assert isinstance(build_store_from_env(), InMemoryJobStore)

    monkeypatch.setenv("SIROM_JOB_STORE", "redis")
    monkeypatch.setenv("SIROM_REDIS_URL", "redis://localhost:6379/0")
    assert isinstance(build_store_from_env(), RedisJobStore)

    monkeypatch.setenv("SIROM_JOB_STORE", "bogus")
    with pytest.raises(ValueError):
        build_store_from_env()
