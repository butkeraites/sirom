"""HTTP interface for the SIROM robust-optimization algorithm.

This subpackage wraps :class:`sirom.batch_solver.ProblemsBucket` in a FastAPI
application so problems can be solved over HTTP without knowing the algorithm
internals. The public entry point is the ASGI app at ``sirom.api.app:app``.
"""
