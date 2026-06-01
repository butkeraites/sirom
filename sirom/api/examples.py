"""The canonical sample problem.

Single source of truth for the example used by the ``/example`` endpoint, the
Swagger "Try it out" prefill, and the test suite. It is the small, known-good
problem from ``tests/test_batch_solver.py`` (objective ``c = [3, 1]`` over a
5x2 constraint set) with low scenario counts so it solves in well under a
second.
"""

from __future__ import annotations

from typing import Any, Dict

EXAMPLE_PROBLEM: Dict[str, Any] = {
    "objective": [3.0, 1.0],
    "lb_A": [[1, 1], [1, 0], [0, 1], [-1, 0], [0, -1]],
    "ub_A": [[2, 2], [2, 1], [1, 2], [-1, 0], [0, -1]],
    "lb_b": [2, 1, 2, 0, 0],
    "ub_b": [3, 2, 3, 0, 0],
    "options": {
        "number_of_scenarios": 50,
        "quality_scenarios": 50,
        "clusters": 3,
        "include_log": False,
    },
}
"""A ready-to-POST body for ``POST /solve``."""
