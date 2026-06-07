# Changelog

All notable changes to SIROM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.4] — 2026-06-07

Documentation patch. No library code or API change.

### Added
- A "Release history" table in `README.md` linking each version to its GitHub
  release, its milestone (with PR count), and `CHANGELOG.md`.

## [0.4.3] — 2026-06-07

Tooling / process patch. No library code or API change.

### Added
- `scripts/refresh_release_index.py` — regenerates the cross-release "Changelog
  index" footer on every GitHub release (versions from `gh release list`,
  blurbs from this file); idempotent, with a `--dry-run` preview.
- A "Cutting a release" section in `CONTRIBUTING.md` documenting the release
  flow end to end.

## [0.4.2] — 2026-06-07

Tooling / process patch. No library code or API change.

### Added
- This `CHANGELOG.md`, documenting every release.
- A three-layer guard ensuring a version bump always ships its changelog entry:
  a CI `changelog-guard` job (the un-bypassable merge gate), a git `pre-commit`
  hook (`.githooks/`, enabled per clone with `git config core.hooksPath
  .githooks`), and a Claude Code `PreToolUse` hook (`.claude/`). See
  `CONTRIBUTING.md`.

## [0.4.1] — 2026-06-07

Documentation patch over 0.4.0. No code or API change.

### Documentation
- The robust-CVRP demo README records that it runs on SIROM v0.4.0 and clarifies
  that the v0.4.0 refactors don't change the `POST /solve` → `GET /jobs/{id}`
  contract the demo consumes.
- `CONTEXT.md` gains the rest of the v0.4.0 vocabulary — the cluster tree
  (`Cluster tree`, `build`, `replicate`) and validation status (`Status`,
  `has_errors`).

## [0.4.0] — 2026-06-06

Architecture-deepening release: make modules honest and concentrate scattered
logic. No behavioural change to the solver or the HTTP API contract.

### Changed
- **Honest two-stage `Solution` contract** — the solver produces an
  `UnscoredSolution` (decision vector only on optimal solves); the scoring stage
  adds `feasibility_probability` via an explicit `score()` transform, yielding a
  `ScoredSolution`. `is_optimal()` / `feasibility()` accessors replace the
  hand-rolled key-presence guards, and `solve_status` is typed `int`.
- **`ClusterTree.build()`** now owns the KMeans / WCSS / selection / subdivision
  algorithm that previously lived in the orchestrator; the `replicate` gate no
  longer leaks. `batch_solver` shrinks ~70 lines.
- **One core-level `has_errors` gate** (`sirom/status_checks.py`) replaces the
  `any("[ERROR]" in s …)` prose-match repeated at eight sites.
- The API drops a brittle `results` slice and counts cluster nodes from the tree.

### Added
- `CONTEXT.md` — domain glossary (including the `feasibility probability` vs
  *robustness* naming split).
- `docs/adr/` — first Architecture Decision Records, capturing the decisions
  **not** to make: keep `OptimizationProblem` + `Coefficient` (0001), keep the
  portfolio and CVRP request builders separate (0002), and count cluster nodes
  from the tree rather than add a stage-tag seam (0003).

## [0.3.0] — 2026-06-05

### Added
- **Robust CVRP demo over VRP-REP** (`demo/vrp/`) — pick a real CVRP instance
  from [VRP-REP](http://www.vrp-rep.org/datasets.html) and get a Pareto frontier
  of candidate routings trading distance against the probability the plan stays
  feasible. A "SIROM-as-a-service" demo: the backend builds a `SolveRequest`,
  POSTs it to the SIROM HTTP API, and polls the job — it never imports SIROM.
  - Two uncertainty modes (demand → capacity; travel time → a synthesized shift
    limit), toggled in the UI with α / customers / scenarios / shift sliders.
  - Every coordinate-based CVRP dataset on VRP-REP (Augerat A/B/P, Fisher F,
    Christofides CMT/Set M, Uchoa 2014 X-instances, Golden, Li, VeRoLog);
    instances listed lazily from each zip; large instances subsampled to the
    depot + K nearest customers.
  - Route-based robustness (Monte-Carlo over decoded routes), an SVG route map,
    and a recharts distance-vs-robustness frontier, all in Docker Compose.

## [0.2.0] — 2026-06-01

First substantial release of SIROM as a deployable, documented service.

### Added
- **HTTP API** — async job API (`POST /solve` → poll `GET /jobs/{id}`) with a
  self-documenting Swagger `/docs` and a ready-to-run `/example`.
- **Docker** — multi-stage image + `docker compose up`, with a Redis-backed job
  store for multi-worker scale-out.
- **MILP support** — integer variables, with solver auto-selection (GLOP for LPs,
  SCIP for MILPs) and overrides for CLP, PDLP, and commercial GUROBI/CPLEX/XPRESS.
- A polished robust **investment-portfolio** web demo (`demo/`).

### Changed
- **~9–29× faster** — GLOP auto-selection, vectorized quality measure and
  scenario generation, and threaded scenario solves (see `benchmarks/`).

### Fixed
- Corrected the constraint-slack used as a clustering feature (it previously
  computed `A_i·x - b_i·sum(x)` instead of `A_i·x - b_i`).

## [0.1.0]

Initial implementation of the Simulation-based Robust Optimization Method
(unreleased; superseded by 0.2.0). Implements the method from Butkeraites,
de Salles Neto & Gendreau, *Expert Systems with Applications* 203:117337 (2022).

[0.4.4]: https://github.com/butkeraites/sirom/releases/tag/v0.4.4
[0.4.3]: https://github.com/butkeraites/sirom/releases/tag/v0.4.3
[0.4.2]: https://github.com/butkeraites/sirom/releases/tag/v0.4.2
[0.4.1]: https://github.com/butkeraites/sirom/releases/tag/v0.4.1
[0.4.0]: https://github.com/butkeraites/sirom/releases/tag/v0.4.0
[0.3.0]: https://github.com/butkeraites/sirom/releases/tag/v0.3.0
[0.2.0]: https://github.com/butkeraites/sirom/releases/tag/v0.2.0
