# Contributing to SIROM

Thanks for your interest in improving SIROM! This guide covers how to set up a
dev environment, the conventions the codebase follows, and how changes get
merged. For an architectural tour, see [CLAUDE.md](CLAUDE.md).

## Development setup

Python **3.11** is the tested target (the package declares `>=3.8`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"     # algorithm + HTTP API + test/dev tools
pytest                          # run the suite (should be all green)
```

Useful commands:

```bash
uvicorn sirom.api.app:app --reload    # serve the API locally -> http://localhost:8000/docs
python -m sirom.main                  # run the algorithm end-to-end (no server)
python benchmarks/bench_pipeline.py   # per-phase performance benchmark
python benchmarks/frontier_diff.py    # compare frontiers across versions/configs
```

## Before you open a PR

- **Tests pass:** `pytest` is green. Add tests for new behavior or bug fixes.
- **Format & types:** `black sirom tests benchmarks` and `mypy sirom`.
- **Keep the docs honest:** if you change the API, the model, or defaults,
  update `README.md` and the relevant `benchmarks/*.md` notes.

CI (`.github/workflows/ci.yml`) runs the test suite and a Docker image build on
every PR; both must be green to merge. Benchmarks are **not** run in CI.

## Project conventions

These are load-bearing — please follow them (details in [CLAUDE.md](CLAUDE.md)):

- **Status-string protocol.** The algorithm classes report problems by appending
  `"[OK] ..."` / `"[ERROR] ..."` strings to a `self.status` list rather than
  raising. Downstream code gates on `any("[ERROR]" in s for s in status)`, and
  tests assert on **exact** strings — don't reword an existing message without
  updating its test.
- **Access style.** `Solution` and the cluster-tree types are `TypedDict`s
  (subscript: `solution["objective_value"]`); `Coefficient`/`Coefficients` use
  `@property` (attribute: `coefficient.constraint`). Mixing them raises at
  runtime.
- **Solver / numerics.** GLOP (a floating-point simplex) is the default LP
  backend, so assert objective values with `pytest.approx`, not `==`. No test
  pins exact decision-vector or frontier values (solver/clustering choices may
  shift which vertices appear); assert envelopes and invariants instead.
- **Preserve the robustness guards** in `apply_quality_measure` and
  `cluster_and_selection` (non-optimal scenarios, empty clusters) — they keep
  arbitrary user input from crashing a run; they're covered by tests.

## Branching & commits

- Branch off `main`; open a PR against `main`.
- Keep PRs focused; write a clear title and a description of *why*.
- Reference any related issue (`Fixes #123`).
- Prefer small, reviewable commits with descriptive messages.

## Reporting issues

Open a GitHub issue with a minimal reproducible example: the problem inputs
(objective, `lb_A`/`ub_A`, `lb_b`/`ub_b`, options) and what you expected vs. saw.
For API issues, include the request body and the job response.

## License

SIROM is licensed under the [GNU GPL v3.0](LICENSE). By contributing, you agree
that your contributions are licensed under the same terms.
