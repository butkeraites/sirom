# sirom

**Simulation-based Robust Optimization Method** — solve linear programs whose
coefficients are *uncertain* (given as intervals `[lb, ub]` instead of fixed
numbers) and get back a set of candidate solutions trading objective value
against robustness.

SIROM solves `min c·x  s.t.  A·x ≤ b`, where each scenario draws `A` from
`[lb_A, ub_A]` and `b` from `[lb_b, ub_b]`. **All variables are non-negative
(x ≥ 0).** You don't need to understand the internals (scenario sampling,
clustering, per-cluster re-solving) — POST a problem to the HTTP API and read
back a Pareto frontier of `objective_value` vs. `feasibility_probability`.

## Run the API with Docker

```bash
docker compose up            # builds the image and serves on :8000
# or:
docker build -t sirom-api .
docker run -p 8000:8000 sirom-api
```

Then open the interactive docs: **http://localhost:8000/docs**

The docs are the fastest way in — they render every field with an explanation
and a prefilled working example. Click **Try it out** on `POST /solve`.

## Call it with curl

Solving is asynchronous: submit a problem, then poll for the result.

```bash
# 1. Grab a ready-to-use sample problem
curl -s localhost:8000/example > problem.json

# 2. Submit it -> returns a job id (HTTP 202)
JOB=$(curl -s -X POST localhost:8000/solve \
      -H 'content-type: application/json' \
      --data @problem.json | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

# 3. Poll until the status is "succeeded"
curl -s localhost:8000/jobs/$JOB
```

A succeeded job returns:

```jsonc
{
  "status": "succeeded",
  "result": {
    "solutions": [
      {"variables": [...], "objective_value": 12.5, "feasibility_probability": 0.98},
      ...
    ],
    "summary": {"scenarios_solved": 50, "best_feasibility": 0.98, ...},
    "warnings": []
  }
}
```

Each entry in `solutions` is a Pareto-optimal trade-off: `objective_value` is
the cost `c·x`, and `feasibility_probability` ∈ [0,1] is how often that
solution stays feasible under random realizations of the uncertain
coefficients (higher = more robust).

### Endpoints

| Method & path        | Purpose                                  |
|----------------------|------------------------------------------|
| `GET  /health`       | Liveness probe                           |
| `GET  /example`      | A ready-to-POST sample problem           |
| `POST /solve`        | Submit a problem → `202` + `job_id`      |
| `GET  /jobs/{id}`    | Poll a job's status / result             |
| `GET  /jobs`         | List submitted jobs                      |
| `GET  /docs`         | Interactive API documentation            |

### Configuration (env vars)

| Variable             | Default     | Meaning                                   |
|----------------------|-------------|-------------------------------------------|
| `SIROM_EXECUTOR`     | `process`   | Job runner: `process`, `thread`, `inline` |
| `SIROM_WORKERS`      | `min(cpu,4)`| Concurrent solve workers                  |
| `SIROM_JOB_STORE`    | `memory`    | Job state store: `memory` or `redis`      |
| `SIROM_REDIS_URL`    | `redis://localhost:6379/0` | Redis URL (when store is `redis`) |
| `SIROM_JOB_TTL`      | `86400`     | Seconds a finished job is kept (redis)    |
| `SIROM_MAX_SCENARIOS`| `2000`      | Per-request scenario cap                  |
| `SIROM_MAX_VARS`     | `200`       | Variable-count cap                        |
| `SIROM_MAX_CONSTRAINTS` | `500`    | Constraint-count cap                      |

### Scaling out

With the default **`memory`** store, job state is process-local — run a
**single** uvicorn worker (the process pool still parallelizes solves).

Set **`SIROM_JOB_STORE=redis`** to share job state across workers and survive
restarts, so you can run multiple uvicorn workers (or replicas) behind a load
balancer — a poll that lands on any worker sees the result. `docker compose up`
is wired this way (a `redis` service + 2 workers). Execution still happens on
the worker that received the request; Redis provides shared *state*, not a
distributed task queue (a crashed worker won't hand its in-flight job to
another). See `sirom/api/jobs.py`.

## Develop

```bash
pip install -e ".[api,dev]"          # algorithm + server + test deps
pytest                                # run the test suite
uvicorn sirom.api.app:app --reload    # serve locally
python -m sirom.main                  # run the algorithm end-to-end (no server)
```

For a deeper tour of the algorithm internals, see `CLAUDE.md`.
