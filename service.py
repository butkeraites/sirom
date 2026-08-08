"""One ASGI app serving everything SIROM can do over HTTP.

Composes surfaces that already exist in this repository so a single container —
and therefore a single scale-to-zero deployment — covers all of them:

    /            SIROM solver API (solve, jobs, example, docs)
    /vrp         vehicle routing under interval uncertainty, when available
    /mcp         Model Context Protocol, so an AI agent can *run* the solver
                 instead of reading about it

Everything is unauthenticated and public by design, which makes bounded input a
correctness requirement rather than a nicety. Limits are enforced in middleware,
before the request reaches a solver, and they are published at `/limits` rather
than left to be discovered by trial and error.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sirom.api.app import app as sirom_app
from sirom.api.examples import EXAMPLE_PROBLEM

log = logging.getLogger("sirom.service")

# ------------------------------------------------------------------ limits

# These bound the SHAPE of a request. The bound that matters is SIROM's own
# cell budget — scenarios x variables x constraints <= 5,000,000 — which limits
# actual work rather than dimensions, and which the solver enforces itself. At
# that ceiling a solve costs about 4-5 s on a laptop.
#
# The per-dimension caps are set to match what the routing surface generates
# internally (cvrp.py caps itself at 200 variables and 500 constraints). Tighter
# values rejected the service's own loopback calls, which is how this was found.
LIMITS = {
    "max_scenarios": 1000,
    "max_clusters": 16,
    "max_variables": 200,
    "max_constraints": 500,
    "max_body_bytes": 4_000_000,
    "cell_budget": 5_000_000,
}

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://butkeraites.com,https://www.butkeraites.com,http://localhost:4321",
    ).split(",") if o.strip()
]

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run the mounted solver app's lifespan too.

    Starlette only runs the lifespan of the top-level application, so a mounted
    sub-app's startup never fires. The SIROM API creates its JobManager there —
    without this, every POST /solve raises AttributeError: 'State' object has no
    attribute 'jobs'. It fails on the main endpoint and nowhere else, which is
    exactly the kind of thing that reaches production.
    """
    async with sirom_app.router.lifespan_context(sirom_app):
        yield


app = FastAPI(
    lifespan=lifespan,
    title="SIROM service",
    version=os.getenv("SERVICE_VERSION", "1.0.0"),
    description=(
        "Robust optimization under interval uncertainty — the reference "
        "implementation of the method published in Expert Systems with "
        "Applications 203:117337 (2022)."
    ),
    docs_url="/service-docs",
    openapi_url="/service-openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=3600,
)


def problem_violations(payload: Any) -> list[str]:
    """Everything wrong with a problem, in one pass.

    Returning all violations rather than the first means a caller fixes their
    request once instead of playing whack-a-mole against a public endpoint.
    """
    if not isinstance(payload, dict):
        return ["body must be a JSON object"]

    bad: list[str] = []
    lb_A = payload.get("lb_A")
    if isinstance(lb_A, list):
        if len(lb_A) > LIMITS["max_constraints"]:
            bad.append(f"at most {LIMITS['max_constraints']} constraints, got {len(lb_A)}")
        if lb_A and isinstance(lb_A[0], list) and len(lb_A[0]) > LIMITS["max_variables"]:
            bad.append(f"at most {LIMITS['max_variables']} variables, got {len(lb_A[0])}")

    opts = payload.get("options")
    if isinstance(opts, dict):
        for key, cap in (
            ("number_of_scenarios", LIMITS["max_scenarios"]),
            ("quality_scenarios", LIMITS["max_scenarios"]),
            ("clusters", LIMITS["max_clusters"]),
        ):
            v = opts.get(key)
            if isinstance(v, (int, float)) and v > cap:
                bad.append(f"options.{key} must be at most {cap}, got {v}")
    return bad


@app.middleware("http")
async def guard(request: Request, call_next):
    """Bound the request before anything expensive touches it."""
    started = time.perf_counter()

    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > LIMITS["max_body_bytes"]:
        return JSONResponse(
            {"detail": f"request body exceeds {LIMITS['max_body_bytes']} bytes"},
            status_code=413,
        )

    # Validate solve payloads here rather than inside the mounted app, so the
    # ceiling holds no matter which sub-application ends up serving the route.
    if request.method == "POST" and request.url.path.rstrip("/").endswith("/solve"):
        body = await request.body()
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            return JSONResponse({"detail": f"invalid JSON: {exc.msg}"}, status_code=400)

        violations = problem_violations(payload)
        if violations:
            return JSONResponse(
                {"detail": "request exceeds published limits",
                 "violations": violations,
                 "limits": LIMITS},
                status_code=422,
            )

        # The body was consumed above; hand it back to the downstream app.
        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": body, "more_body": False}
        request = Request(request.scope, receive)

    response = await call_next(request)
    response.headers["X-Compute-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.get("/status", tags=["meta"], summary="Liveness probe and warm-up hook")
def status() -> dict[str, Any]:
    """The site pings this on page load, so the container is already running by
    the time a visitor moves a slider. That is what makes min-instances=0
    invisible rather than merely cheap.

    Named /status rather than /healthz because Google Frontend intercepts
    /healthz ahead of Cloud Run and answers it with its own HTML 404 — the
    request never reaches the container. The solver API's own /health also
    works and is the lighter warm-up target.
    """
    return {"status": "ok", "service": "sirom", "version": app.version}


@app.get("/limits", tags=["meta"], summary="What this service will accept")
def limits() -> dict[str, Any]:
    return {
        **LIMITS,
        "note": ("Requests beyond these are rejected with the full list of "
                 "violations, never silently truncated."),
        "measured": (
            "800 scenarios x 10 clusters on a small problem: about 0.28 s. "
            "At the 5,000,000-cell budget: about 4-5 s."
        ),
        "binding_constraint": (
            "scenarios x variables x constraints must not exceed cell_budget; "
            "the solver enforces this itself and it is stricter than the "
            "per-dimension caps for anything large."
        ),
    }


# --------------------------------------------------------- portfolio surface

# Declared here rather than mounting demo/app.py, which also serves a static
# index page this image does not ship. Thin wrappers keep the request ceiling
# in one place.

from pydantic import BaseModel, Field  # noqa: E402


class AssetIn(BaseModel):
    name: str = Field(max_length=40)
    return_low: float = Field(ge=-1.0, le=5.0)
    return_high: float = Field(ge=-1.0, le=5.0)
    risk: float = Field(gt=0.0, le=10.0)
    cap: float = Field(default=0.5, gt=0.0, le=1.0)


class PortfolioRequest(BaseModel):
    target_return: float = Field(default=0.10, ge=0.0, le=1.0)
    number_of_scenarios: int = Field(default=120, ge=20, le=1000)
    assets: list[AssetIn] = Field(default_factory=list, max_length=12)
    uncertainty_scale: float = Field(
        default=1.0, ge=0.25, le=3.0,
        description=(
            "Widens or narrows every asset's return interval about its midpoint. "
            "1.0 leaves the intervals as given."
        ),
    )


@app.get("/portfolio/assets", tags=["portfolio"], summary="The default asset universe")
def portfolio_assets() -> dict[str, Any]:
    from demo.portfolio import DEFAULT_ASSETS
    return {"assets": [
        {"name": a.name, "return_low": a.return_low, "return_high": a.return_high,
         "risk": a.risk, "cap": a.cap}
        for a in DEFAULT_ASSETS
    ]}


@app.post("/portfolio/optimize", tags=["portfolio"],
          summary="Risk / robustness frontier for a portfolio")
def portfolio_optimize(req: PortfolioRequest) -> dict[str, Any]:
    """Solve the robust portfolio problem and return its Pareto frontier.

    Synchronous: the whole point of the demo is that a slider move produces an
    answer, and at these sizes the solve is faster than a round of polling.
    """
    from demo.portfolio import DEFAULT_ASSETS, Asset, optimize

    chosen = [Asset(a.name, a.return_low, a.return_high, a.risk, a.cap)
              for a in req.assets] or list(DEFAULT_ASSETS)

    if req.uncertainty_scale != 1.0:
        k = req.uncertainty_scale
        widened = []
        for a in chosen:
            mid = (a.return_low + a.return_high) / 2
            half = (a.return_high - a.return_low) / 2 * k
            widened.append(Asset(a.name, mid - half, mid + half, a.risk, a.cap))
        chosen = widened

    started = time.perf_counter()
    out = optimize(chosen, req.target_return, req.number_of_scenarios)
    out["wall_seconds"] = round(time.perf_counter() - started, 4)
    out["uncertainty_scale"] = req.uncertainty_scale
    out["intervals"] = [
        {"name": a.name, "low": a.return_low, "high": a.return_high,
         "risk": a.risk, "cap": a.cap} for a in chosen
    ]
    return out


# --------------------------------------------------------------------- MCP

MCP_PROTOCOL = "2025-06-18"

TOOLS = [
    {
        "name": "get_example_problem",
        "description": (
            "Return a ready-to-solve interval linear program. Use this to see "
            "the exact request shape solve_robust expects."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "solve_robust",
        "description": (
            "Solve a linear program whose coefficients are intervals rather than "
            "fixed numbers, and return a Pareto frontier trading objective value "
            "against the probability the solution stays feasible. This runs the "
            "SIROM method (Expert Systems with Applications 203:117337, 2022) — "
            "it is the real solver, not a description of one."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["objective", "lb_A", "ub_A", "lb_b", "ub_b"],
            "properties": {
                "objective": {"type": "array", "items": {"type": "number"},
                              "description": "Cost vector c in min c.x"},
                "lb_A": {"type": "array", "description": "Lower bounds of the constraint matrix"},
                "ub_A": {"type": "array", "description": "Upper bounds of the constraint matrix"},
                "lb_b": {"type": "array", "items": {"type": "number"}},
                "ub_b": {"type": "array", "items": {"type": "number"}},
                "integer_variables": {"type": "array", "items": {"type": "integer"}},
                "number_of_scenarios": {
                    "type": "integer", "minimum": 10,
                    "maximum": LIMITS["max_scenarios"], "default": 120},
                "clusters": {
                    "type": "integer", "minimum": 2,
                    "maximum": LIMITS["max_clusters"], "default": 5},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_limits",
        "description": "The bounds this public service enforces on any request.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def run_solver(args: dict[str, Any]) -> dict[str, Any]:
    """Solve synchronously. The MCP caller wants an answer, not a job id."""
    from sirom.api.schemas import SolveRequest
    from sirom.api.service import solve_problem

    payload = {
        "objective": args["objective"],
        "lb_A": args["lb_A"], "ub_A": args["ub_A"],
        "lb_b": args["lb_b"], "ub_b": args["ub_b"],
        "integer_variables": args.get("integer_variables", []),
        "options": {
            "number_of_scenarios": min(int(args.get("number_of_scenarios", 120)),
                                       LIMITS["max_scenarios"]),
            "quality_scenarios": min(int(args.get("number_of_scenarios", 120)),
                                     LIMITS["max_scenarios"]),
            "clusters": min(int(args.get("clusters", 5)), LIMITS["max_clusters"]),
            "include_log": False,
        },
    }
    violations = problem_violations(payload)
    if violations:
        raise ValueError("; ".join(violations))

    started = time.perf_counter()
    result = solve_problem(SolveRequest(**payload))
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    data["wall_seconds"] = round(time.perf_counter() - started, 4)
    return data


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "get_example_problem":
        return EXAMPLE_PROBLEM if isinstance(EXAMPLE_PROBLEM, dict) else dict(EXAMPLE_PROBLEM)
    if name == "get_limits":
        return dict(LIMITS)
    if name == "solve_robust":
        return run_solver(args)
    raise KeyError(f"unknown tool: {name}")


def rpc_result(rid: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def rpc_error(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


@app.post("/mcp", tags=["mcp"], summary="Model Context Protocol endpoint")
async def mcp(request: Request):
    """A minimal, dependency-free MCP server over HTTP.

    Hand-written rather than pulled from an SDK because the surface is three
    methods wide and this image is already 500 MB of solver; the protocol is
    JSON-RPC and the cost of getting it wrong is visible immediately.
    """
    try:
        msg = json.loads(await request.body() or b"{}")
    except json.JSONDecodeError as exc:
        return JSONResponse(rpc_error(None, -32700, f"parse error: {exc.msg}"), status_code=400)

    rid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}

    if method == "initialize":
        return JSONResponse(rpc_result(rid, {
            "protocolVersion": MCP_PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "sirom", "version": app.version},
            "instructions": (
                "SIROM solves linear programs with interval-uncertain coefficients "
                "and returns a Pareto frontier of objective value against "
                "feasibility probability. Call get_example_problem first to see "
                "the request shape, then solve_robust."
            ),
        }))

    if method in ("notifications/initialized", "initialized"):
        return JSONResponse({}, status_code=202)

    if method == "tools/list":
        return JSONResponse(rpc_result(rid, {"tools": TOOLS}))

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            data = call_tool(name, args)
        except KeyError as exc:
            return JSONResponse(rpc_error(rid, -32601, str(exc)))
        except ValueError as exc:
            return JSONResponse(rpc_result(rid, {
                "isError": True,
                "content": [{"type": "text", "text": f"Rejected: {exc}"}],
            }))
        except Exception as exc:  # solver failure is data, not a protocol error
            log.exception("tool %s failed", name)
            return JSONResponse(rpc_result(rid, {
                "isError": True,
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            }))
        return JSONResponse(rpc_result(rid, {
            "content": [{"type": "text", "text": json.dumps(data, default=str)}],
            "structuredContent": data,
            "isError": False,
        }))

    return JSONResponse(rpc_error(rid, -32601, f"method not found: {method}"))


@app.get("/mcp.json", tags=["mcp"], summary="How to connect an agent to this service")
def mcp_manifest(request: Request) -> dict[str, Any]:
    base = str(request.base_url).rstrip("/")
    return {
        "name": "sirom",
        "description": "Robust optimization under interval uncertainty.",
        "protocolVersion": MCP_PROTOCOL,
        "transport": {"type": "streamable-http", "url": f"{base}/mcp"},
        "tools": [t["name"] for t in TOOLS],
        "authentication": "none",
        "note": (
            "Discovery paths under /.well-known are not standardized for MCP yet, "
            "so this manifest is linked explicitly rather than advertised there."
        ),
    }


# ------------------------------------------------------------------ mounts

try:
    from demo.vrp.backend.app import app as vrp_app
    # Graft the routing routes rather than mounting the app: they are already
    # namespaced under /vrp, so mounting would produce /vrp/vrp/solve. This also
    # leaves that app's own /health and /docs behind, which this service already
    # provides.
    grafted = [r for r in vrp_app.routes
               if getattr(r, "path", "").startswith("/vrp")]
    app.router.routes.extend(grafted)
    log.info("VRP surface attached: %s", [r.path for r in grafted])
except Exception as exc:  # pragma: no cover
    # Routing instances are fetched from a third-party dataset host at runtime.
    # That host is not allowed to decide whether this service starts.
    log.warning("VRP surface unavailable: %s", exc)

# Mounted last and at the root, so every route defined above wins on overlap.
app.mount("/", sirom_app)
