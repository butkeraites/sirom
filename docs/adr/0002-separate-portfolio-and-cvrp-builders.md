# ADR-0002: Keep the portfolio and CVRP request builders separate

- Status: Accepted (2026-06-05)
- Context: architecture review finding #6 (refuted on adversarial verification)

## Decision

Do **not** extract a shared `ProblemBuilder` across `demo/portfolio.py` and
`demo/vrp/backend/cvrp.py`. Each keeps its own assembly of the
`lb_A`/`ub_A`/`lb_b`/`ub_b` interval arrays.

## Context

Both builders construct an interval-valued robust LP and both apply the rule that
negating an interval `[lo, hi]` gives `[-hi, -lo]`. A review candidate proposed a
shared builder to reason that rule once.

## Why this was refused

- **The two callers sit across a deliberate process division.**
  `demo/vrp/backend/cvrp.py` **never imports SIROM** — its only `sirom` tokens are
  the `SIROM_API_URL` env var (`cvrp.py:46`), a comment, and a local
  `solve_via_sirom` function. It reaches SIROM solely over HTTP (`httpx`), runs in
  its own Dockerfile, and **re-declares `MAX_VARS=200` locally** (`cvrp.py:52`) as a
  deliberate firewall. `demo/portfolio.py`, by contrast, imports SIROM in-process
  (`from sirom.api.service import solve_problem`, `portfolio.py:25`). A shared
  builder living in the `sirom` package could not be imported by `cvrp.py` without
  breaking that decoupling; "sharing" would mean copying it into a second container.
- **The shared rule is already enforced centrally.** The `SolveRequest` validator
  rejects any `lo > hi` interval for every caller, including the HTTP one, so a
  mistake is a clean `422` — a shared builder would add no safety.
- **The "common reasoning" is thin.** Portfolio negates an *A-coefficient* interval
  on a `≥` row; CVRP negates an *RHS* interval on a capacity row — different formulas
  at different positions. The only shared rule is the one-line interval negation.

## Consequence

The duplication is intentional defensive design. The local `MAX_VARS`/`CELL_BUDGET`
re-declarations in `cvrp.py` are a feature: if SIROM's schema changes incompatibly,
the divergence surfaces as a failure rather than silently propagating across a
process boundary.
