# Frontier impact of the constraint-slack fix

The constraint-slack fix (PR #7) changed the per-constraint clustering feature
from `A_i·x − b_i·sum(x)` to the correct slack `A_i·x − b_i`. Since that feature
drives clustering, it changes which solutions land on the Pareto frontier. This
note records how much, measured with `frontier_diff.py` (same problem and
**replayed scenarios** before vs. after, so the difference is the formula alone).

## Results

Small instance (6 vars × 15 constraints, N=120): **35 / 42 points identical**,
same objective and feasibility ranges.

Larger instance (20 vars × 50 constraints, N=300), across 5 seeds:

| seed | #before | #after | shared | obj range match | feas range match |
|---|---|---|---|---|---|
| 1 | 21 | 19 | 7 | yes | yes |
| 2 | 26 | 26 | 13 | yes | yes |
| 3 | 13 | 14 | 4 | yes | yes |
| 4 | 26 | 27 | 12 | yes | yes |
| 5 | 18 | 20 | 6 | yes | yes |

Mean Jaccard overlap ≈ **0.24**. Best objective achievable at a given
feasibility level differed by only small, mixed-sign amounts (typically within
~±0.04 of an objective of magnitude 12–14; one outlier ~0.22 at seed 1).

XL instance (50 vars × 150 constraints, N=400): **2 of ~15 points identical**
(Jaccard ≈ 0.07), ranges still identical, trade-off deltas < 0.4% of an
objective of magnitude ~32.

### Overlap shrinks with problem size

| instance | variables | frontier overlap (Jaccard) |
|---|---|---|
| 6 × 15 | 6 | ~0.83 |
| 20 × 50 | 20 | ~0.24 |
| 50 × 150 | 50 | ~0.07 |

Monotonic: more variables ⇒ larger `sum(x)` ⇒ bigger `b·(sum(x) − 1)` error ⇒
the cluster trees, and thus the frontier's specific points, diverge more.

## Conclusion

- **Structural change is large and grows with problem size** — frontier-point
  overlap falls from ~83% (6 vars) to ~24% (20 vars) to ~7% (50 vars). The
  error term `b·(sum(x) − 1)` scales with `sum(x)`, so more variables ⇒ more
  divergence. Consistent across all 5 seeds at 20×50.
- **The decision-relevant envelope is stable** — objective and feasibility
  ranges matched exactly on every seed, and neither formula systematically
  dominates the cost-vs-robustness trade-off.

So the fix is correct and matters for reproducing *specific* solution sets
(especially at scale), but pre-fix frontiers were not qualitatively misleading.

## Reproduce

```bash
python benchmarks/frontier_diff.py prep --dir /tmp/fd --vars 20 --con 50 --scenarios 300 --quality 500 --seed 5
python benchmarks/frontier_diff.py run  --dir /tmp/fd --out /tmp/fd/after.json
git checkout dd77b02 -- sirom/mini_ortools_solver.py   # pre-fix slack
python benchmarks/frontier_diff.py run  --dir /tmp/fd --out /tmp/fd/before.json
git checkout HEAD -- sirom/mini_ortools_solver.py      # restore
python benchmarks/frontier_diff.py diff /tmp/fd/before.json /tmp/fd/after.json
```
