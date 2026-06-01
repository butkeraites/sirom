# Benchmarks

Wall-clock timing of the SIROM pipeline, phase by phase, used to prioritize and
validate solving-time optimizations. Not run by `pytest`.

```bash
pip install -e .                       # make `sirom` importable
python benchmarks/bench_pipeline.py    # default grid
python benchmarks/bench_pipeline.py --full   # larger, slower grid
```

Columns are seconds for each phase: `generate` (scenario coefficients, in
`__init__`), `solve` (N scenario LPs), `cluster` (KMeans tree), `tree`
(per-node re-solve), `quality` (feasibility scoring over M scenarios), `total`.

## Baseline (SCIP, serial, pandas) — pre-optimization

Apple Silicon, `main` @ PR #5. Reference point for the optimization phases.

| vars | con | N | M | generate | solve | cluster | tree | quality | total |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 5 | 100 | 100 | 0.013 | 0.164 | 0.185 | 0.108 | 0.808 | 1.277 |
| 20 | 50 | 100 | 100 | 0.016 | 0.644 | 0.016 | 1.177 | 0.895 | 2.747 |
| 50 | 200 | 100 | 100 | 0.047 | 3.091 | 0.295 | 7.369 | 4.447 | 15.249 |
| 20 | 50 | 500 | 500 | 0.119 | 3.326 | 0.374 | 5.880 | 20.335 | 30.034 |

Dominant costs: the SCIP LP solves (`solve` + `tree`) at larger sizes, and
`quality` at high M. `generate` is negligible.
