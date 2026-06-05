"""Turn a (subsampled) CVRP instance into a robust LP for the SIROM API.

This is the heart of the demo. It builds an arc-based Capacitated Vehicle
Routing MILP with Miller–Tucker–Zemlin (MTZ) subtour elimination, injects
*interval uncertainty* into the constraints, submits the problem to the **SIROM
HTTP API** (it never imports SIROM), polls the job to completion, and decodes
each returned decision vector back into vehicle routes.

Model (depot = node 0, customers 1..n, K vehicles, capacity Q):

    minimize    Σ d_ij · x_ij                         (fixed Euclidean distance)
    subject to  Σ_i x_ij = 1         ∀ customer j      (in-degree)
                Σ_j x_ij = 1         ∀ customer i      (out-degree)
                Σ_j x_0j ≤ K                           (vehicles available)
                Σ_j x_0j = Σ_i x_i0                    (depot flow balance)
                u_i - u_j + Q·x_ij ≤ Q - d_j           (MTZ load / subtour elim)
                d_i ≤ u_i ≤ Q                          (load bounds)

Uncertainty (toggled per solve):

* **demand**      d_j ∈ [q_j(1-α), q_j(1+α)] — lives entirely in the RHS `b`.
* **travel_time** add a time-MTZ chain t_j ≥ t_i + τ_ij·x_ij - M(1-x_ij),
                  t_j ≤ MaxShift, with τ_ij ∈ [d_ij(1-α), d_ij(1+α)] — lives in
                  `A`; capacity uses nominal demand. MaxShift is synthesized.

SIROM models uncertainty per *coefficient* and samples each independently, so a
customer's demand appearing in several rows is perturbed independently per row.
The robustness score is therefore "fraction of perturbed scenarios in which the
fixed plan stays feasible" — a valid robustness measure, documented in the
README.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import httpx

from vrprep import Instance

SIROM_API_URL = os.getenv("SIROM_API_URL", "http://sirom-api:8000").rstrip("/")
SOLVE_TIMEOUT = float(os.getenv("VRP_SOLVE_TIMEOUT", "300"))
POLL_INTERVAL = float(os.getenv("VRP_POLL_INTERVAL", "1.0"))

# Mirrors sirom/api/schemas.py so we can fail fast with a friendly message
# instead of bouncing off the API's 422.
MAX_VARS = 200
MAX_CONSTRAINTS = 500
CELL_BUDGET = 5_000_000


class SolveError(Exception):
    """Raised when the SIROM API rejects or fails the problem."""


@dataclass
class Built:
    request: Dict          # SolveRequest-shaped JSON for the SIROM API
    arc_index: Dict[Tuple[int, int], int]  # (i, j) -> column in the decision vector
    n_nodes: int           # depot + customers
    vehicles: int          # K
    nodes: List[Dict]      # [{index, id, x, y, demand, depot}], index 0 = depot
    capacity: float        # Q
    max_shift: float       # synthesized per-route time limit (travel_time mode; else 0)


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def vehicles_needed(instance: Instance) -> int:
    """K = enough vehicles to carry the (nominal) total demand, ≥ 1, ≤ n."""
    total = sum(c.demand for c in instance.customers)
    k = max(1, math.ceil(total / instance.capacity)) if instance.capacity > 0 else 1
    return max(1, min(k, instance.n))


def _nodes_payload(instance: Instance) -> List[Dict]:
    nodes = [
        {
            "index": 0,
            "id": instance.depot_id,
            "x": instance.depot_cx,
            "y": instance.depot_cy,
            "demand": 0.0,
            "depot": True,
        }
    ]
    for i, c in enumerate(instance.customers, start=1):
        nodes.append(
            {"index": i, "id": c.id, "x": c.cx, "y": c.cy, "demand": c.demand, "depot": False}
        )
    return nodes


def build_request(
    instance: Instance,
    mode: str,
    alpha: float,
    n_scenarios: int,
    shift_factor: float = 1.1,
) -> Built:
    """Build a SolveRequest dict for the SIROM API and the arc→column index map."""
    if mode not in ("demand", "travel_time"):
        raise SolveError(f"Unknown uncertainty mode: {mode!r}.")
    if not (0.0 <= alpha < 1.0):
        raise SolveError("alpha must be in [0, 1).")

    n = instance.n
    nodes = _nodes_payload(instance)
    Q = instance.capacity
    K = vehicles_needed(instance)

    # Coordinates and (nominal) demands indexed by node index (0 = depot).
    xs = [nd["x"] for nd in nodes]
    ys = [nd["y"] for nd in nodes]
    dem = [nd["demand"] for nd in nodes]  # dem[0] = 0 (depot)

    # --- decision-variable layout -------------------------------------------
    # 1) arc binaries x_ij for all ordered pairs i != j over nodes {0..n}
    arcs: List[Tuple[int, int]] = [
        (i, j) for i in range(n + 1) for j in range(n + 1) if i != j
    ]
    arc_index = {ij: idx for idx, ij in enumerate(arcs)}
    n_arc = len(arcs)
    # 2) load vars u_i for customers 1..n
    u_off = n_arc
    # 3) time vars t_i for customers 1..n (travel_time mode only)
    time_mode = mode == "travel_time"
    t_off = u_off + n
    n_vars = t_off + (n if time_mode else 0)

    def u_col(i: int) -> int:
        return u_off + (i - 1)

    def t_col(i: int) -> int:
        return t_off + (i - 1)

    # --- objective: fixed Euclidean distance on arcs, 0 on u/t --------------
    objective = [0.0] * n_vars
    dist = {
        (i, j): _distance(xs[i], ys[i], xs[j], ys[j]) for (i, j) in arcs
    }
    for (i, j), idx in arc_index.items():
        objective[idx] = dist[(i, j)]

    integer_variables = list(range(n_arc))  # the arc binaries

    # Constraint rows accumulate as (coef_dict, lb, ub). Equalities are split
    # into two ≤ rows by the caller helpers below.
    lb_A: List[List[float]] = []
    ub_A: List[List[float]] = []
    lb_b: List[float] = []
    ub_b: List[float] = []

    def add_row(coefs: Dict[int, float], lo_b: float, hi_b: float,
                a_overrides: Dict[int, Tuple[float, float]] | None = None) -> None:
        """Append one ``Σ coef·x ≤ b`` row. ``a_overrides`` gives per-column
        (lb, ub) for uncertain A entries; everything else is fixed (lb == ub)."""
        lo_row = [0.0] * n_vars
        hi_row = [0.0] * n_vars
        for col, val in coefs.items():
            lo_row[col] = val
            hi_row[col] = val
        if a_overrides:
            for col, (lo, hi) in a_overrides.items():
                lo_row[col] = lo
                hi_row[col] = hi
        lb_A.append(lo_row)
        ub_A.append(hi_row)
        lb_b.append(lo_b)
        ub_b.append(hi_b)

    def add_eq(coefs: Dict[int, float], rhs: float) -> None:
        add_row(coefs, rhs, rhs)                       # Σ ≤ rhs
        add_row({c: -v for c, v in coefs.items()}, -rhs, -rhs)  # -Σ ≤ -rhs

    # in-degree = 1, out-degree = 1 for each customer
    for j in range(1, n + 1):
        add_eq({arc_index[(i, j)]: 1.0 for i in range(n + 1) if i != j}, 1.0)
    for i in range(1, n + 1):
        add_eq({arc_index[(i, j)]: 1.0 for j in range(n + 1) if j != i}, 1.0)

    # depot out-degree ≤ K
    add_row({arc_index[(0, j)]: 1.0 for j in range(1, n + 1)}, K, K)
    # depot flow balance: Σ_j x_0j - Σ_i x_i0 = 0
    balance = {}
    for j in range(1, n + 1):
        balance[arc_index[(0, j)]] = balance.get(arc_index[(0, j)], 0.0) + 1.0
    for i in range(1, n + 1):
        balance[arc_index[(i, 0)]] = balance.get(arc_index[(i, 0)], 0.0) - 1.0
    add_eq(balance, 0.0)

    # NOTE: no explicit x_ij ≤ 1 rows are needed. Every arc variable appears in
    # some customer's in- or out-degree equality (Σ = 1) of non-negative
    # integers, which already forces each arc to be 0/1. Dropping the (n+1)·n
    # cap rows keeps even travel-time mode under MAX_CONSTRAINTS.

    # --- capacity (MTZ) ------------------------------------------------------
    # u_i - u_j + Q·x_ij ≤ Q - d_j   for customer pairs i != j
    # demand mode: d_j uncertain → RHS interval. travel_time: nominal d_j.
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            coefs = {u_col(i): 1.0, u_col(j): -1.0, arc_index[(i, j)]: Q}
            if mode == "demand":
                # RHS = Q - d_j, d_j in [q(1-a), q(1+a)] -> b in [Q-q(1+a), Q-q(1-a)]
                lo_b = Q - dem[j] * (1 + alpha)
                hi_b = Q - dem[j] * (1 - alpha)
                add_row(coefs, lo_b, hi_b)
            else:
                add_row(coefs, Q - dem[j], Q - dem[j])

    # load bounds: u_i ≤ Q ; u_i ≥ d_i  (-u_i ≤ -d_i)
    for i in range(1, n + 1):
        add_row({u_col(i): 1.0}, Q, Q)
        if mode == "demand":
            add_row({u_col(i): -1.0}, -dem[i] * (1 + alpha), -dem[i] * (1 - alpha))
        else:
            add_row({u_col(i): -1.0}, -dem[i], -dem[i])

    # --- travel time (MTZ) ---------------------------------------------------
    max_shift = 0.0
    if time_mode:
        # Synthesize a per-route shift limit ≈ one vehicle's nominal route time:
        # depot out-and-back plus a hop per stop. The hop is the mean *nearest
        # neighbour* distance (a real route chains nearby stops); averaging over
        # all customer pairs badly overestimates and the limit never binds.
        depot_dists = [dist[(0, i)] for i in range(1, n + 1)]
        mean_depot = sum(depot_dists) / n if n else 0.0
        if n > 1:
            nn = [min(dist[(i, j)] for j in range(1, n + 1) if j != i) for i in range(1, n + 1)]
            mean_hop = sum(nn) / len(nn)
        else:
            mean_hop = 0.0
        per_route = math.ceil(n / K)
        max_shift = shift_factor * (2 * mean_depot + per_route * mean_hop)
        max_tau_ub = max((dist[(i, j)] * (1 + alpha) for (i, j) in arcs), default=0.0)
        big_m = max_shift + max_tau_ub + 1.0

        # t_j ≥ t_i + τ_ij·x_ij - M(1-x_ij)  ->  t_i - t_j + (τ_ij+M)·x_ij ≤ M
        # depot origin (i = 0): t_0 = 0  ->  -t_j + (τ_0j+M)·x_0j ≤ M
        for (i, j) in arcs:
            if j == 0:
                continue  # arrival "time" at the depot isn't tracked
            tau = dist[(i, j)]
            lo = tau * (1 - alpha) + big_m
            hi = tau * (1 + alpha) + big_m
            coefs: Dict[int, float] = {arc_index[(i, j)]: 0.0}  # placeholder; A override sets it
            if i != 0:
                coefs[t_col(i)] = 1.0
            coefs[t_col(j)] = -1.0
            add_row(coefs, big_m, big_m, a_overrides={arc_index[(i, j)]: (lo, hi)})

        # shift limit: t_j ≤ MaxShift
        for j in range(1, n + 1):
            add_row({t_col(j): 1.0}, max_shift, max_shift)

    request = {
        "objective": objective,
        "lb_A": lb_A,
        "ub_A": ub_A,
        "lb_b": lb_b,
        "ub_b": ub_b,
        "integer_variables": integer_variables,
        "options": {
            "number_of_scenarios": n_scenarios,
            "quality_scenarios": n_scenarios,
            "clusters": 3,
        },
    }

    n_cons = len(lb_A)
    if n_vars > MAX_VARS or n_cons > MAX_CONSTRAINTS or n_scenarios * n_vars * n_cons > CELL_BUDGET:
        raise SolveError(
            f"Problem too large for SIROM (vars={n_vars}/{MAX_VARS}, "
            f"constraints={n_cons}/{MAX_CONSTRAINTS}, "
            f"cells={n_scenarios * n_vars * n_cons}/{CELL_BUDGET}). Reduce the "
            "customer count or scenarios" + (", or switch off travel-time mode."
            if time_mode else ".")
        )

    return Built(
        request=request,
        arc_index=arc_index,
        n_nodes=n + 1,
        vehicles=K,
        nodes=nodes,
        capacity=Q,
        max_shift=max_shift,
    )


def solve_via_sirom(request: Dict) -> Dict:
    """Submit to the SIROM HTTP API and poll the job to completion.

    Returns the ``SolveResponse`` dict (``solutions``/``summary``/``warnings``).
    Raises :class:`SolveError` on rejection, failure, or timeout.
    """
    deadline = time.monotonic() + SOLVE_TIMEOUT
    try:
        with httpx.Client(base_url=SIROM_API_URL, timeout=30.0) as client:
            resp = client.post("/solve", json=request)
            if resp.status_code == 422:
                raise SolveError(f"SIROM rejected the problem: {resp.text}")
            resp.raise_for_status()
            job_id = resp.json()["job_id"]

            while True:
                job = client.get(f"/jobs/{job_id}")
                job.raise_for_status()
                body = job.json()
                state = body.get("status")
                if state == "succeeded":
                    return body["result"]
                if state == "failed":
                    errs = body.get("errors") or ["unknown error"]
                    raise SolveError("SIROM solve failed: " + "; ".join(errs))
                if time.monotonic() > deadline:
                    raise SolveError(
                        f"SIROM solve timed out after {SOLVE_TIMEOUT:.0f}s "
                        f"(last status: {state})."
                    )
                time.sleep(POLL_INTERVAL)
    except httpx.HTTPError as exc:
        raise SolveError(
            f"Could not reach the SIROM API at {SIROM_API_URL} ({type(exc).__name__})."
        ) from exc


def decode_routes(variables: List[float], built: Built) -> List[List[int]]:
    """Reconstruct vehicle routes (lists of node indices, depot-bookended).

    Reads the arc variables, builds a successor map, and walks out of the depot
    once per outgoing arc. Returns ``[]`` if the arc structure is degenerate.
    """
    used: List[Tuple[int, int]] = []
    for (i, j), col in built.arc_index.items():
        if col < len(variables) and variables[col] > 0.5:
            used.append((i, j))

    succ: Dict[int, List[int]] = {}
    for i, j in used:
        succ.setdefault(i, []).append(j)

    routes: List[List[int]] = []
    starts = list(succ.get(0, []))
    visited_arcs = set()
    for first in starts:
        route = [0, first]
        cur = first
        guard = 0
        while cur != 0 and guard <= built.n_nodes + 1:
            guard += 1
            nexts = [j for j in succ.get(cur, []) if (cur, j) not in visited_arcs]
            if not nexts:
                break
            nxt = nexts[0]
            visited_arcs.add((cur, nxt))
            route.append(nxt)
            cur = nxt
        if route[-1] == 0 and len(route) >= 3:
            routes.append(route)
    return routes


ROBUSTNESS_TRIALS = 3000


def _canonical_routes(routes: List[List[int]]) -> frozenset:
    """A direction-independent key for a set of routes.

    A route and its reverse are the same physical tour, so we normalise each
    route's interior to the lexicographically smaller of (seq, reversed seq).
    Two solutions that differ only in traversal direction or route order then
    share a key and collapse to one candidate.
    """
    keys = []
    for r in routes:
        interior = tuple(r[1:-1])
        keys.append(min(interior, interior[::-1]))
    return frozenset(keys)


def route_robustness(routes: List[List[int]], built: Built, mode: str, alpha: float) -> float:
    """Monte-Carlo robustness of a *routing* under interval uncertainty.

    Robustness is a property of the routes, not of SIROM's auxiliary MTZ load
    variables (which are free at the optimum and otherwise make equivalent
    routings score differently). We therefore re-score each decoded routing
    directly:

    * demand mode      — sample each customer's demand in [q(1-α), q(1+α)];
                         a routing is feasible iff every route's load ≤ capacity.
    * travel_time mode — sample each arc's time in [d(1-α), d(1+α)];
                         feasible iff every route's total time ≤ MaxShift.
    """
    nodes = built.nodes
    demand = [nd["demand"] for nd in nodes]

    def dist(i: int, j: int) -> float:
        return _distance(nodes[i]["x"], nodes[i]["y"], nodes[j]["x"], nodes[j]["y"])

    rng = random.Random(0xC0FFEE)
    Q = built.capacity
    shift = built.max_shift
    ok = 0
    for _ in range(ROBUSTNESS_TRIALS):
        feasible = True
        for r in routes:
            if mode == "demand":
                load = sum(
                    demand[n] * (1 + rng.uniform(-alpha, alpha)) for n in r if n != 0
                )
                if load > Q + 1e-9:
                    feasible = False
                    break
            else:
                t = sum(
                    dist(r[k], r[k + 1]) * (1 + rng.uniform(-alpha, alpha))
                    for k in range(len(r) - 1)
                )
                if t > shift + 1e-9:
                    feasible = False
                    break
        if feasible:
            ok += 1
    return ok / ROBUSTNESS_TRIALS


def _pareto(candidates: List[Dict]) -> List[Dict]:
    """Non-dominated set: minimise objective (distance), maximise robustness."""
    front = []
    for c in candidates:
        dominated = any(
            o is not c
            and o["objective"] <= c["objective"]
            and o["robustness"] >= c["robustness"]
            and (o["objective"] < c["objective"] or o["robustness"] > c["robustness"])
            for o in candidates
        )
        if not dominated:
            front.append(c)
    return front


def solve(
    instance: Instance,
    mode: str,
    alpha: float,
    n_scenarios: int,
    subsample_note: str | None,
    shift_factor: float = 1.1,
) -> Dict:
    """End-to-end: build → call SIROM → decode → score routes → frontier.

    SIROM generates the candidate routings (scenario solves + clustering); we
    score each routing's robustness directly on its routes (see
    :func:`route_robustness` for why we don't trust SIROM's vector-level
    feasibility here), dedup physically-equivalent routings, and return the
    non-dominated distance/robustness frontier.
    """
    built = build_request(instance, mode, alpha, n_scenarios, shift_factor=shift_factor)
    result = solve_via_sirom(built.request)

    by_key: Dict[frozenset, Dict] = {}
    for sol in result.get("solutions", []):
        routes = decode_routes(sol["variables"], built)
        if not routes:
            continue
        key = _canonical_routes(routes)
        if key in by_key:
            continue  # physically the same routing as one we already kept
        by_key[key] = {
            "objective": sol["objective_value"],
            "robustness": route_robustness(routes, built, mode, alpha),
            "vehicles": len(routes),
            "routes": routes,
        }

    candidates = _pareto(list(by_key.values()))
    # Most robust first, then cheapest distance.
    candidates.sort(key=lambda c: (-c["robustness"], c["objective"]))

    return {
        "instance": {
            "dataset": instance.dataset,
            "name": instance.name,
            "capacity": instance.capacity,
            "vehicles": built.vehicles,
            "nodes": built.nodes,
            "total_customers": instance.total_customers,
            "used_customers": instance.n,
        },
        "mode": mode,
        "alpha": alpha,
        "shift_factor": shift_factor,
        "max_shift": round(built.max_shift, 1) if mode == "travel_time" else None,
        "subsample_note": subsample_note,
        "candidates": candidates,
        "summary": result.get("summary", {}),
        "warnings": result.get("warnings", []),
    }
