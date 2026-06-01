"""Render the /example Pareto frontier to docs/frontier_example.png.

Used to (re)generate the README illustration. Requires matplotlib (in the
`dev` extra). The exact points vary run-to-run with the random sampling; the
shape — objective vs. robustness trade-off — is what matters.

    pip install -e ".[dev]"
    python benchmarks/plot_frontier.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sirom.api.examples import EXAMPLE_PROBLEM  # noqa: E402
from sirom.api.schemas import SolveRequest  # noqa: E402
from sirom.api.service import solve_problem  # noqa: E402

OUT = os.path.join("docs", "frontier_example.png")


def main() -> None:
    response = solve_problem(SolveRequest(**EXAMPLE_PROBLEM))
    points = sorted(
        (s.feasibility_probability, s.objective_value) for s in response.solutions
    )
    feasibility = [p[0] for p in points]
    objective = [p[1] for p in points]

    os.makedirs("docs", exist_ok=True)
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(feasibility, objective, "o-", color="#2b6cb0", markersize=5, linewidth=1.5)
    plt.xlabel("feasibility probability  (robustness →)")
    plt.ylabel("objective value  (min c·x ↓ is better)")
    plt.title(f"SIROM Pareto frontier — /example ({len(points)} candidates)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    print(f"wrote {OUT} ({len(points)} points)")


if __name__ == "__main__":
    main()
