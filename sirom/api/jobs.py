"""In-memory job store and pluggable executor for asynchronous solves.

A solve can take seconds to minutes, so ``POST /solve`` enqueues work and
returns immediately; clients poll ``GET /jobs/{id}``. Three executor modes are
supported via the ``SIROM_EXECUTOR`` env var:

* ``process`` (default) — ``ProcessPoolExecutor``: true CPU parallelism, crash
  isolation, and per-child stdout capture (so the algorithm's prints can't
  interleave across concurrent jobs).
* ``thread`` — ``ThreadPoolExecutor``: lighter, relies on OR-Tools releasing
  the GIL during solves.
* ``inline`` — runs synchronously inside ``submit``: used by the test suite and
  for the simplest single-request deployments.

The store is process-local, so run a **single** uvicorn worker. Swapping this
class for a Redis-backed store is the documented path to horizontal scaling;
nothing else in the API needs to change.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .errors import SolveError
from .schemas import JobStatus
from .service import run_solve_job

_GENERIC_FAILURE = "The job failed unexpectedly."


def _max_workers() -> int:
    raw = os.getenv("SIROM_WORKERS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return min(os.cpu_count() or 2, 4)


def _max_jobs() -> int:
    raw = os.getenv("SIROM_MAX_JOBS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 1000


class JobManager:
    """Tracks submitted jobs and exposes their normalized status/result."""

    def __init__(self, mode: Optional[str] = None, max_workers: Optional[int] = None):
        self.mode = (mode or os.getenv("SIROM_EXECUTOR", "process")).lower()
        self._lock = threading.Lock()
        self._jobs: "Dict[str, Dict[str, Any]]" = {}
        self._max_jobs = _max_jobs()
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
        if self.mode == "inline":
            record = self._run_inline(payload)
        else:
            assert self._executor is not None
            future: Future = self._executor.submit(run_solve_job, payload)
            record = {"future": future}
        with self._lock:
            self._evict_locked()
            self._jobs[job_id] = record
        return job_id

    @staticmethod
    def _run_inline(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {"future": None, "status": JobStatus.succeeded, "result": run_solve_job(payload)}
        except SolveError as exc:
            return {"future": None, "status": JobStatus.failed, "errors": exc.messages}
        except Exception:  # noqa: BLE001
            return {"future": None, "status": JobStatus.failed, "errors": [_GENERIC_FAILURE]}

    def _evict_locked(self) -> None:
        """Drop the oldest finished jobs once the store exceeds its cap."""
        while len(self._jobs) >= self._max_jobs:
            oldest_id = next(iter(self._jobs))
            del self._jobs[oldest_id]

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{status, result, errors}`` for a job, or ``None`` if unknown."""
        with self._lock:
            record = self._jobs.get(job_id)
        if record is None:
            return None
        return self._normalize(record)

    @staticmethod
    def _normalize(record: Dict[str, Any]) -> Dict[str, Any]:
        future: Optional[Future] = record.get("future")
        if future is None:  # inline: status decided at submit time
            return {
                "status": record["status"],
                "result": record.get("result"),
                "errors": record.get("errors"),
            }
        if not future.done():
            status = JobStatus.running if future.running() else JobStatus.pending
            return {"status": status, "result": None, "errors": None}
        exc = future.exception()
        if exc is None:
            return {"status": JobStatus.succeeded, "result": future.result(), "errors": None}
        if isinstance(exc, SolveError):
            return {"status": JobStatus.failed, "result": None, "errors": exc.messages}
        return {"status": JobStatus.failed, "result": None, "errors": [_GENERIC_FAILURE]}

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(self._jobs.keys())
            records = {jid: self._jobs[jid] for jid in ids}
        return [
            {"job_id": jid, "status": self._normalize(records[jid])["status"]}
            for jid in ids
        ]

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
