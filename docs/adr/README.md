# Architecture Decision Records

Short records of architectural decisions — especially decisions **not** to make a
change, so a future review (human or automated) doesn't re-propose something that
was already weighed and rejected.

| ADR | Decision |
|-----|----------|
| [0001](0001-retain-optimizationproblem-coefficient.md) | Keep `OptimizationProblem` + `Coefficient` separate; keep the DataFrame canonical form |
| [0002](0002-separate-portfolio-and-cvrp-builders.md) | Keep the `portfolio.py` and `cvrp.py` request builders separate |
| [0003](0003-results-cluster-count-not-stage-tags.md) | Count cluster nodes from the tree; don't add a stage-tag seam to results |
