# Frontier impact of the constraint-slack fix

The constraint-slack fix (PR #7) changed the per-constraint clustering feature
from `A_i·x − b_i·sum(x)` to the correct slack `A_i·x − b_i`. Since that feature
drives clustering, it changes which solutions land on the Pareto frontier. This
note records how much, measured with `frontier_diff.py` (same problem and
**replayed scenarios** before vs. after, so the difference is the formula alone).

## Results

Aggregated over seeds (`|ΔObj|` = shift in best objective achievable at a fixed
feasibility level; objective magnitudes are ~13 at 20×50 and ~32 at 50×150):

| instance | seeds | Jaccard overlap (mean [min–max]) | avg #before → #after | range match (obj, feas) | mean \|ΔObj\| | max \|ΔObj\| |
|---|---|---|---|---|---|---|
| 6 × 15 (N=120) | 1 | ~0.83 | 42 → 42 | yes | — | — |
| 20 × 50 (N=300) | 5 | **0.24** [0.17–0.33] | 20.8 → 21.2 | 5/5, 5/5 | 0.041 | 0.224 |
| 50 × 150 (N=400) | 5 | **0.09** [0.07–0.14] | 14.4 → 15.0 | 5/5, 5/5 | 0.086 | 0.402 |

### Robustness to constraint structure

The rows above all use the default generator (box rows + non-negative coupling,
mixed-sign objective, ±10% intervals). Repeating at 30×80 (3 seeds each) with
structurally different problems — `ΔObj%` = trade-off shift relative to the
objective magnitude:

| structure | Jaccard | range match | mean ΔObj% | max ΔObj% |
|---|---|---|---|---|
| mixed-sign constraint coefficients | 0.10 | 3/3, 3/3 | 0.8% | 3.0% |
| all-negative objective (large sum(x)) | 0.09 | 3/3, 3/3 | 0.2% | 0.8% |
| wide ±30% uncertainty intervals | 0.09 | 3/3, 3/3 | 1.3% | 4.3% |

Same picture as the default structure: low overlap, exact range match (18/18),
small trade-off shift — somewhat larger under wide uncertainty.

## Conclusion

- **Structural change is large, scales with problem size, and is seed-robust.**
  Frontier-point overlap collapses ~0.83 → 0.24 → 0.09 from 6 to 20 to 50
  variables, with narrow per-instance bands (0.17–0.33, 0.07–0.14 over 5 seeds).
  The error term `b·(sum(x) − 1)` grows with `sum(x)`, so more variables ⇒ more
  cluster-tree divergence ⇒ fewer shared frontier points.
- **The envelope is invariant** — objective and feasibility ranges matched
  exactly on *every* seed, *every* size, and *every* constraint structure
  tested (10/10 across sizes, 18/18 across structures).
- **The trade-off is stable on average; worst-case shifts are small and
  structure-dependent.** Mean shift stays a fraction of a percent to ~1.3% of
  the objective; the worst case is mixed-sign and modest (a few percent), and
  is largest under wide uncertainty intervals — but never reorders the frontier
  envelope.
- **Robust to constraint structure too** — low overlap (~0.1) and the stable
  envelope/trade-off hold for mixed-sign coefficients, a `sum(x)`-maximizing
  objective, and wide intervals, not just the default generator.

So the fix is correct and matters increasingly for reproducing *specific*
solution sets as problems scale, but pre-fix frontiers were not qualitatively
misleading — the achievable cost-vs-robustness frontier is essentially the same.

## Reproduce

```bash
python benchmarks/frontier_diff.py prep --dir /tmp/fd --vars 20 --con 50 --scenarios 300 --quality 500 --seed 5
python benchmarks/frontier_diff.py run  --dir /tmp/fd --out /tmp/fd/after.json
git checkout dd77b02 -- sirom/mini_ortools_solver.py   # pre-fix slack
python benchmarks/frontier_diff.py run  --dir /tmp/fd --out /tmp/fd/before.json
git checkout HEAD -- sirom/mini_ortools_solver.py      # restore
python benchmarks/frontier_diff.py diff /tmp/fd/before.json /tmp/fd/after.json
```
