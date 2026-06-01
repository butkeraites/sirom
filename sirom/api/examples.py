"""The canonical sample problem.

Single source of truth for the example used by the ``/example`` endpoint, the
Swagger "Try it out" prefill, and the test suite. It is a small interval LP
chosen to produce a *non-trivial* Pareto frontier (a real objective-vs-
robustness trade-off, not a single point), so the interactive demo is
illustrative. Scenario counts are low so it solves in well under a second.

The model maximizes ``3x + 4y`` (i.e. minimizes ``-3x - 4y``) subject to three
uncertain constraints plus ``x, y >= 0``. The first three constraint rows carry
interval uncertainty; the last two (``-x <= 0``, ``-y <= 0``) are fixed.
"""

from __future__ import annotations

from typing import Any, Dict

EXAMPLE_PROBLEM: Dict[str, Any] = {
    "objective": [-3.0, -4.0],
    "lb_A": [[1, 2], [-3, 1], [1, -1], [-1, 0], [0, -1]],
    "ub_A": [[1.3, 2.3], [-2.7, 1.2], [1.2, -0.8], [-1, 0], [0, -1]],
    "lb_b": [14, 0, 2, 0, 0],
    "ub_b": [16, 0, 3, 0, 0],
    "options": {
        "number_of_scenarios": 60,
        "quality_scenarios": 60,
        "clusters": 3,
        "include_log": False,
    },
}
"""A ready-to-POST body for ``POST /solve`` (yields a multi-point frontier)."""
