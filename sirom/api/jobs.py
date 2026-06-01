"""Asynchronous job execution with a pluggable, optionally shared store.

A solve can take seconds to minutes, so ``POST /solve`` enqueues work and
returns immediately; clients poll ``GET /jobs/{id}``.

Two things are configurable and independent:

* **Executor** (``SIROM_EXECUTOR``) — how a job runs: ``process`` (default,
  ``ProcessPoolExecutor``: CPU parallelism + crash isolation), ``thread``, or
  ``inline`` (synchronous; used by tests and the simplest deployments).
* **Store** (``SIROM_JOB_STORE``) — where job state lives: ``memory`` (default,
  process-local — run a single uvicorn worker) or ``redis`` (shared across
  workers and durable across restarts, so you can scale out).

Job outcomes are always recorded by the **parent** process (via the future's
done-callback for pool executors), so the same code path works for both stores:
with Redis, a poll that lands on any worker reads the result the executing
worker wrote. Execution stays local to the worker that received the request —
Redis provides shared *state*, not a distributed task queue (a crashed worker
won't hand its in-flight job to another). That is the documented boundary; a
Celery/RQ-style queue would be the next step beyond it.
"""

from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .errors import SolveError
from .schemas import JobStatus
from .service import run_solve_job

_GENERIC_FAILURE = "The job failed unexpectedly."


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default


def _max_workers() -> int:
    return _env_int("SIROM_WORKERS", min(os.cpu_count() or 2, 4))


# ---------------------------------------------------------------------------
# Job stores
# ---------------------------------------------------------------------------

# A stored record is a plain dict: {"status": str, "result": dict|None,
# "errors": list[str]|None}, where status is one of the JobStatus values.


class JobStore(ABC):
    """Persists the lifecycle state of jobs."""

    @abstractmethod
    def create(self, job_id: str) -> None:
        """Register a new job in the ``pending`` state."""

    @abstractmethod
    def record_success(self, job_id: str, result: Dict[str, Any]) -> None: ...

    @abstractmethod
    def record_failure(self, job_id: str, errors: List[str]) -> None: ...

    @abstractmethod
    def fetch(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{status, result, errors}`` or ``None`` if unknown."""

    @abstractmethod
    def list_ids(self) -> List[str]: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class InMemoryJobStore(JobStore):
    """Process-local store. Fine for a single worker; lost on restart."""

    def __init__(self, max_jobs: int = 1000):
        self._lock = threading.Lock()
        self._data: "Dict[str, Dict[str, Any]]" = {}
        self._max_jobs = max_jobs

    def _set(self, job_id: str, record: Dict[str, Any]) -> None:
        with self._lock:
            self._data[job_id] = record

    def create(self, job_id: str) -> None:
        with self._lock:
            while len(self._data) >= self._max_jobs:
                self._data.pop(next(iter(self._data)))
            self._data[job_id] = {
                "status": JobStatus.pending.value,
                "result": None,
                "errors": None,
            }

    def record_success(self, job_id: str, result: Dict[str, Any]) -> None:
        self._set(job_id, {"status": JobStatus.succeeded.value, "result": result, "errors": None})

    def record_failure(self, job_id: str, errors: List[str]) -> None:
        self._set(job_id, {"status": JobStatus.failed.value, "result": None, "errors": errors})

    def fetch(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._data.get(job_id)
            return dict(record) if record is not None else None

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())


class RedisJobStore(JobStore):
    """Shared store backed by Redis, so multiple workers see the same jobs.

    Each job is a JSON value under ``{key_prefix}{job_id}`` with a TTL, so old
    jobs expire automatically. ``client`` can be injected (e.g. fakeredis) for
    tests; otherwise a client is created from ``url``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 86_400,
        client: Any = None,
        key_prefix: str = "sirom:job:",
    ):
        if client is None:
            try:
                import redis  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "The redis job store requires the 'redis' package "
                    "(install sirom[api])."
                ) from exc
            client = redis.Redis.from_url(url, decode_responses=True)
        self._redis = client
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def _write(self, job_id: str, record: Dict[str, Any]) -> None:
        self._redis.set(self._key(job_id), json.dumps(record), ex=self._ttl)

    def create(self, job_id: str) -> None:
        self._write(
            job_id,
            {"status": JobStatus.pending.value, "result": None, "errors": None},
        )

    def record_success(self, job_id: str, result: Dict[str, Any]) -> None:
        self._write(job_id, {"status": JobStatus.succeeded.value, "result": result, "errors": None})

    def record_failure(self, job_id: str, errors: List[str]) -> None:
        self._write(job_id, {"status": JobStatus.failed.value, "result": None, "errors": errors})

    def fetch(self, job_id: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._key(job_id))
        return json.loads(raw) if raw else None

    def list_ids(self) -> List[str]:
        start = len(self._prefix)
        return [key[start:] for key in self._redis.scan_iter(match=f"{self._prefix}*", count=100)]

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:  # pragma: no cover
            pass


def build_store_from_env() -> JobStore:
    kind = os.getenv("SIROM_JOB_STORE", "memory").lower()
    if kind == "memory":
        return InMemoryJobStore(max_jobs=_env_int("SIROM_MAX_JOBS", 1000))
    if kind == "redis":
        return RedisJobStore(
            url=os.getenv("SIROM_REDIS_URL", "redis://localhost:6379/0"),
            ttl_seconds=_env_int("SIROM_JOB_TTL", 86_400),
        )
    raise ValueError(
        f"Unknown SIROM_JOB_STORE {kind!r}; expected 'memory' or 'redis'."
    )


# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------


class JobManager:
    """Runs jobs on an executor and records their outcomes in a store."""

    def __init__(
        self,
        executor_mode: Optional[str] = None,
        store: Optional[JobStore] = None,
        max_workers: Optional[int] = None,
    ):
        self.mode = (executor_mode or os.getenv("SIROM_EXECUTOR", "process")).lower()
        self._store = store if store is not None else build_store_from_env()
        self._lock = threading.Lock()
        self._futures: "Dict[str, Future]" = {}
        workers = max_workers or _max_workers()

        if self.mode == "process":
            self._executor: Optional[Any] = ProcessPoolExecutor(max_workers=workers)
        elif self.mode == "thread":
            self._executor = ThreadPoolExecutor(max_workers=workers)
        elif self.mode == "inline":
            self._executor = None
        else:
            raise ValueError(
                f"Unknown SIROM_EXECUTOR mode {self.mode!r}; expected "
                "'process', 'thread', or 'inline'."
            )

    def submit(self, payload: Dict[str, Any]) -> str:
        """Enqueue a solve and return its job id."""
        job_id = uuid4().hex
        self._store.create(job_id)
        if self.mode == "inline":
            self._run_inline(job_id, payload)
        else:
            assert self._executor is not None
            future: Future = self._executor.submit(run_solve_job, payload)
            with self._lock:
                self._futures[job_id] = future
            future.add_done_callback(self._make_recorder(job_id))
        return job_id

    def _run_inline(self, job_id: str, payload: Dict[str, Any]) -> None:
        try:
            self._store.record_success(job_id, run_solve_job(payload))
        except SolveError as exc:
            self._store.record_failure(job_id, exc.messages)
        except Exception:  # noqa: BLE001
            self._store.record_failure(job_id, [_GENERIC_FAILURE])

    def _make_recorder(self, job_id: str):
        def _record(future: Future) -> None:
            try:
                self._store.record_success(job_id, future.result())
            except SolveError as exc:
                self._store.record_failure(job_id, exc.messages)
            except Exception:  # noqa: BLE001 (incl. CancelledError on shutdown)
                self._store.record_failure(job_id, [_GENERIC_FAILURE])
            finally:
                with self._lock:
                    self._futures.pop(job_id, None)

        return _record

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{status, result, errors}`` for a job, or ``None``."""
        record = self._store.fetch(job_id)
        if record is None:
            return None
        # Refine pending -> running for jobs we're executing on this worker.
        if record["status"] == JobStatus.pending.value:
            with self._lock:
                future = self._futures.get(job_id)
            if future is not None and future.running():
                record = {**record, "status": JobStatus.running.value}
        return record

    def list_jobs(self) -> List[Dict[str, Any]]:
        result = []
        for job_id in self._store.list_ids():
            record = self.get(job_id)
            if record is not None:
                result.append({"job_id": job_id, "status": record["status"]})
        return result

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._store.close()
