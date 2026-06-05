"""The single error-detection gate for the status-string convention.

Every validation class accumulates a ``status: list[str]`` of ``[OK]`` /
``[ERROR]`` / ``[INFO]`` messages and gates the next step on whether any error
was recorded. This is that gate, in one place, so the prose-match lives once.

It sits at the core level (not in ``api/``) so the algorithm modules can depend
on it without importing upward into the HTTP layer; ``api/errors.has_errors``
keeps its own copy to preserve that layering.
"""

from __future__ import annotations

from typing import List


def has_errors(status: List[str]) -> bool:
    """Whether any ``[ERROR]`` message has been recorded in ``status``."""
    return any("[ERROR]" in entry for entry in status)
