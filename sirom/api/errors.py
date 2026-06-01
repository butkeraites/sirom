"""Translate the algorithm's internal failure signals into user-facing text.

``ProblemsBucket`` reports problems by appending ``[ERROR] ...`` strings to its
``status`` list rather than raising. This module scans those strings and maps
the known ones to friendly messages, so API clients never see raw internal
diagnostics.
"""

from __future__ import annotations

from typing import List

# Ordered substring -> friendly message. The first matching entry wins, so put
# more specific phrases before generic ones.
_FRIENDLY: List[tuple[str, str]] = [
    (
        "Dimension inconsistency",
        "The objective, constraint matrix, and right-hand-side have mismatched "
        "dimensions.",
    ),
    (
        "Failed acquiring number of scenarios",
        "number_of_scenarios must be a non-negative integer.",
    ),
    ("Undefined number of scenarios", "number_of_scenarios is required."),
    ("Undefined objective", "The objective vector is missing or empty."),
    ("Undefined lb_constraint", "The lower-bound constraint matrix is missing."),
    ("Undefined ub_constraint", "The upper-bound constraint matrix is missing."),
    ("Undefined lb_rhs", "The lower-bound right-hand-side vector is missing."),
    ("Undefined ub_rhs", "The upper-bound right-hand-side vector is missing."),
    ("Failed acquiring", "One of the coefficient inputs could not be parsed."),
]


class SolveError(Exception):
    """Raised when a problem cannot be solved. Carries client-safe messages."""

    def __init__(self, messages: List[str]):
        self.messages = messages
        super().__init__("; ".join(messages))

    def __reduce__(self):
        # Ensure it round-trips through pickle (ProcessPoolExecutor) with the
        # message list intact rather than the joined string.
        return (SolveError, (self.messages,))


def has_errors(status: List[str]) -> bool:
    """True if any status entry signals a failure."""
    return any("[ERROR]" in entry for entry in status)


def friendly_messages(status: List[str]) -> List[str]:
    """Map the ``[ERROR]`` entries in ``status`` to friendly messages.

    Unrecognized errors fall back to their raw text (minus the ``[ERROR]``
    prefix) so nothing is silently swallowed. Duplicates are removed while
    preserving order.
    """
    out: List[str] = []
    for entry in status:
        if "[ERROR]" not in entry:
            continue
        message = next(
            (msg for needle, msg in _FRIENDLY if needle in entry),
            entry.replace("[ERROR]", "").strip(),
        )
        if message not in out:
            out.append(message)
    return out
