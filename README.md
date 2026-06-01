# SIROM — Simulation-based Robust Optimization Method

[![CI](https://github.com/butkeraites/sirom/actions/workflows/ci.yml/badge.svg)](https://github.com/butkeraites/sirom/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

**Robust optimization under uncertainty — without having to pick an uncertainty
budget.**

Many optimization problems have coefficients you don't know exactly: demands,
costs, capacities that live in a *range* rather than at a fixed value. Classical
robust optimization makes you specify, up front, *how much* uncertainty to
protect against (an "uncertainty budget"). SIROM doesn't. It **samples** the
uncertain region, **solves** each realization, **clusters** the resulting
solutions by how they behave, and returns a **Pareto frontier** of candidate
solutions that trade objective value against robustness — so the decision-maker
chooses the trade-off *after* seeing the options, not before.

You give it a linear program whose coefficients are intervals; you get back a
small set of solutions, each labeled with its objective value and the
probability it stays feasible under uncertainty.

```mermaid
flowchart LR
    A["Uncertain LP<br/>intervals on A, b"] --> B["1· Sample<br/>scenarios (LHS)"]
    B --> C["2· Solve<br/>each scenario"]
    C --> D["3· Cluster &amp; select<br/>by solution behavior"]
    D --> E["4· Re-solve<br/>per cluster"]
    E --> F["Pareto frontier<br/>objective vs robustness"]
```

The payoff is a frontier of candidate solutions — here, the built-in
`/example` problem:

![Example Pareto frontier: objective value vs. feasibility probability](docs/frontier_example.png)

*Each point is a candidate solution. Moving right trades a better objective
value for higher robustness; you pick the operating point.* (Regenerate with
`python benchmarks/plot_frontier.py`.)

It implements the method from the peer-reviewed paper (Butkeraites, de Salles
Neto & Gendreau, *Expert Systems with Applications*, 2022 — see
[Citation](#citation)) and ships it as a deployable HTTP service.

## Why SIROM

- **No uncertainty budget required** — explore the whole trade-off instead of
  committing to one protection level a priori.
- **A Pareto frontier, not a single answer** — every candidate is labeled with
  `objective_value` and `feasibility_probability` (robustness), so you pick.
- **LP and MILP** — continuous by default; declare integer variables for
  mixed-integer problems.
- **Fast** — auto-selects the right OR-Tools backend (GLOP for LPs); the
  pipeline is vectorized and ~**9–29× faster** than the original implementation
  ([benchmarks](benchmarks/)).
- **Bring your own solver** — GLOP / CLP / PDLP / SCIP / CBC out of the box, and
  commercial **GUROBI / CPLEX / XPRESS** when licensed.
- **Deploy in one command** — Docker image + async job API with a
  self-documenting `/docs`; Redis-backed job store for horizontal scaling.

## Quickstart (Docker)

```bash
docker compose up        # builds the image, starts the API (+ Redis) on :8000
```

Open the interactive docs at **http://localhost:8000/docs** — every field is
documented and pre-filled with a working example. Click **Try it out** on
`POST /solve`.

Solving is asynchronous — submit a problem, then poll for the result:

```bash
# 1. grab a ready-to-use sample problem
curl -s localhost:8000/example > problem.json

# 2. submit it -> HTTP 202 with a job id
JOB=$(curl -s -X POST localhost:8000/solve -H 'content-type: application/json' \
      --data @problem.json | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

# 3. poll until status is "succeeded"
curl -s localhost:8000/jobs/$JOB
```

A succeeded job returns the frontier — representative output for the sample
problem (exact values vary run-to-run with the random sampling):

```jsonc
{
  "status": "succeeded",
  "result": {
    "solutions": [
      {"variables": [4.47, 3.95], "objective_value": -29.2, "feasibility_probability": 0.98},
      {"variables": [5.02, 4.16], "objective_value": -31.7, "feasibility_probability": 0.68},
      {"variables": [5.15, 4.09], "objective_value": -31.8, "feasibility_probability": 0.63}
    ],
    "summary": {"scenarios_solved": 60, "candidate_solutions": 27, "best_feasibility": 0.98},
    "warnings": []
  }
}
```

Each entry is a Pareto-optimal trade-off: pushing `objective_value` lower (a
better `c·x`) typically costs robustness, and `feasibility_probability ∈ [0,1]`
is how often that solution stays feasible across random realizations of the
uncertain coefficients. **All variables are non-negative (`x ≥ 0`).**

## Use as a Python library

```python
from sirom.batch_solver import ProblemsBucket

# min c·x  s.t.  A·x ≤ b,  with A ∈ [lb_A, ub_A], b ∈ [lb_b, ub_b], x ≥ 0
# (this is the /example problem: maximize 3x + 4y under interval uncertainty)
bucket = ProblemsBucket(
    c_value=[-3, -4],
    lb_A_value=[[1, 2], [-3, 1], [1, -1], [-1, 0], [0, -1]],
    ub_A_value=[[1.3, 2.3], [-2.7, 1.2], [1.2, -0.8], [-1, 0], [0, -1]],
    lb_b_value=[14, 0, 2, 0, 0],
    ub_b_value=[16, 0, 3, 0, 0],
    number_of_scenarios=100,
    n_jobs=-1,              # parallelize scenario solves across all cores
    # integer_variables=[0],     # -> mixed-integer (auto-uses a MIP solver)
    # solver_selection="GUROBI", # -> override the backend (if licensed)
)
bucket.solve()                                  # solve each sampled scenario
bucket.cluster_and_selection()                  # cluster the solution space
bucket.solve_cluster_tree()                     # robust re-solve per cluster
bucket.apply_quality_measure(number_of_scenarios=100)  # score feasibility

for s in bucket.results:
    if s.get("solve_status") == 0:
        print(s["objective_value"], s["feasibility_probability"], s["variable"])
```

```bash
pip install -e ".[api]"      # or ".[api,dev]" for the test/dev tools
```

## How it works

SIROM treats robust optimization as an *unsupervised learning* problem over the
solution space. For a problem `min c·x  s.t.  A·x ≤ b` with `A ∈ [lb_A, ub_A]`
and `b ∈ [lb_b, ub_b]`:

1. **Sample** `N` scenarios from the uncertainty set (Latin-hypercube sampling).
2. **Solve** each scenario's LP independently.
3. **Cluster & select** the solutions, using each solution's objective value and
   constraint slacks as features, into a tree of "similarly-behaving" groups.
4. **Re-solve** a robust sub-problem over each cluster's scenarios.
5. Collect everything into a **Pareto frontier** of objective vs. feasibility.

The frontier composition depends on the sampling and clustering, but its
*envelope* (the achievable objective/robustness range) is stable — see the
[frontier analysis](benchmarks/slack_fix_frontier.md). Full method details are
in the [paper](#citation); an internals tour is in [CLAUDE.md](CLAUDE.md).

## Solvers

The backend is auto-selected — **GLOP** (OR-Tools' simplex) for pure LPs,
**SCIP** when integer variables are present. Override it via
`ProblemsBucket(solver_selection=...)` or the API's `options.solver`:

| kind | options |
|------|---------|
| LP (open-source) | `GLOP` (default), `CLP`, `PDLP` |
| MILP (open-source) | `SCIP` (default), `CBC` |
| commercial | `GUROBI`, `CPLEX`, `XPRESS` — when the OR-Tools build is linked against them and a license is present |

A solver that isn't in the build fails fast with a clear message. See the
[solver comparison](benchmarks/solver_comparison.md) for why GLOP is the default.

## API reference

| Method & path        | Purpose                                  |
|----------------------|------------------------------------------|
| `GET  /health`       | Liveness probe                           |
| `GET  /example`      | A ready-to-POST sample problem           |
| `POST /solve`        | Submit a problem → `202` + `job_id`      |
| `GET  /jobs/{id}`    | Poll a job's status / result             |
| `GET  /jobs`         | List submitted jobs                      |
| `GET  /docs`         | Interactive API documentation            |

### Configuration (environment variables)

| Variable                | Default     | Meaning                                   |
|-------------------------|-------------|-------------------------------------------|
| `SIROM_EXECUTOR`        | `process`   | Job runner: `process`, `thread`, `inline` |
| `SIROM_WORKERS`         | `min(cpu,4)`| Concurrent solve workers                  |
| `SIROM_JOB_STORE`       | `memory`    | Job state store: `memory` or `redis`      |
| `SIROM_REDIS_URL`       | `redis://localhost:6379/0` | Redis URL (when store is `redis`) |
| `SIROM_JOB_TTL`         | `86400`     | Seconds a finished job is kept (redis)    |
| `SIROM_MAX_SCENARIOS`   | `2000`      | Per-request scenario cap                  |
| `SIROM_MAX_VARS`        | `200`       | Variable-count cap                        |
| `SIROM_MAX_CONSTRAINTS` | `500`       | Constraint-count cap                      |

### Scaling out

The default **`memory`** store is process-local — run a **single** uvicorn
worker (the internal process pool still parallelizes solves). Set
**`SIROM_JOB_STORE=redis`** to share job state across workers and survive
restarts, so you can run multiple workers/replicas behind a load balancer
(`docker compose up` is wired this way). Redis provides shared *state*, not a
distributed task queue — execution stays on the worker that received the
request. See [`sirom/api/jobs.py`](sirom/api/jobs.py).

## Develop

```bash
pip install -e ".[api,dev]"          # algorithm + server + test deps
pytest                                # run the test suite
uvicorn sirom.api.app:app --reload    # serve locally
python -m sirom.main                  # run the algorithm end-to-end (no server)
python benchmarks/bench_pipeline.py   # per-phase performance benchmark
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
conventions, and the PR workflow.

## Citation

If you use SIROM in academic work, please cite:

> R. B. C. Butkeraites, L. L. de Salles Neto, M. Gendreau. *A sampling-based
> multi-objective iterative robust optimization method for the bandwidth packing
> problem.* Expert Systems with Applications, 203:117337, 2022.

```bibtex
@article{butkeraites2022sirom,
  title   = {A sampling-based multi-objective iterative robust optimization method for the bandwidth packing problem},
  author  = {Butkeraites, Renan Brito Cano and de Salles Neto, Luiz Leduino and Gendreau, Michel},
  journal = {Expert Systems with Applications},
  volume  = {203},
  pages   = {117337},
  year    = {2022},
  doi     = {10.1016/j.eswa.2022.117337}
}
```

## License

[GNU General Public License v3.0](LICENSE).
