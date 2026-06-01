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

## After optimization (GLOP + vectorized quality + numpy solver + vectorized generation)

Same machine, serial (`n_jobs=1`).

| vars | con | N | M | generate | solve | cluster | tree | quality | total | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 5 | 100 | 100 | 0.001 | 0.008 | 0.124 | 0.010 | 0.003 | 0.145 | 8.8x |
| 20 | 50 | 100 | 100 | 0.011 | 0.053 | 0.098 | 0.111 | 0.006 | 0.280 | 9.8x |
| 50 | 200 | 100 | 100 | 0.061 | 0.408 | 0.357 | 0.718 | 0.210 | 1.754 | 8.7x |
| 20 | 50 | 500 | 500 | 0.017 | 0.237 | 0.281 | 0.402 | 0.091 | 1.027 | 29.2x |

What moved the needle (in order of impact):

1. **numpy solver build/eval** — biggest win; the per-coefficient pandas loops
   (`iterrows`/`SetCoefficient`/`float`) dominated `solve`/`tree`, not the LP solve.
2. **vectorized quality measure** — collapsed an O(results x scenarios) pandas
   loop into one batched matmul (up to ~160x on `quality` at high M).
3. **GLOP default** — faster per-LP simplex for the (continuous) LPs; the win is
   only visible once the pandas build overhead above is removed.
4. **vectorized generation** — minor; removed the list-of-DataFrames churn.

On top of the serial numbers, the scenario solves parallelize across cores via
`ProblemsBucket(n_jobs=-1)`: at 50x200, N=800 the `solve` phase drops from
~3.55s to ~1.08s with 4 threads (~3.3x). The API keeps `n_jobs=1` so its
process pool owns parallelism; the standalone CLI (`sirom/main.py`) uses all cores.
