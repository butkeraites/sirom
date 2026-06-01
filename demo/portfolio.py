"""Robust portfolio optimization on top of SIROM.

Frames portfolio construction as the robust LP SIROM solves:

    minimize    sum_i risk_i * x_i          (portfolio risk)
    subject to  sum_i x_i <= 1              (budget; cash allowed)
                sum_i r_i * x_i >= target   (return floor, r_i UNCERTAIN)
                x_i <= cap_i                (per-asset cap)
                x_i >= 0

Each asset's expected return ``r_i`` is an interval ``[low, high]`` — exactly
the kind of parametric uncertainty SIROM was built for. SIROM samples returns
within those intervals, finds the minimum-risk allocation that meets the target
in each scenario, clusters the results, and reports how often each allocation
actually meets the target. The output is a Pareto frontier trading **risk**
against the **probability of hitting the target return**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sirom.api.schemas import SolveRequest
from sirom.api.service import solve_problem


@dataclass
class Asset:
    name: str
    return_low: float      # low end of the expected annual return, e.g. 0.05 = 5%
    return_high: float     # high end of the expected annual return
    risk: float            # volatility proxy (>0); minimized in aggregate
    cap: float = 0.5       # maximum weight for this asset


DEFAULT_ASSETS: List[Asset] = [
    Asset("Tech Equity",     0.08, 0.24, 0.22, cap=0.30),
    Asset("Emerging Mkts",   0.06, 0.27, 0.26, cap=0.30),
    Asset("Real Estate",     0.05, 0.14, 0.11, cap=0.35),
    Asset("Corp. Bonds",     0.03, 0.07, 0.05, cap=0.40),
    Asset("Gold",            0.00, 0.17, 0.14, cap=0.30),
    Asset("T-Bills",         0.02, 0.04, 0.02, cap=0.40),
]


def build_request(
    assets: List[Asset], target_return: float, number_of_scenarios: int = 120
) -> SolveRequest:
    n = len(assets)

    def zeros() -> List[float]:
        return [0.0] * n

    objective = [a.risk for a in assets]
    lb_A: List[List[float]] = []
    ub_A: List[List[float]] = []
    lb_b: List[float] = []
    ub_b: List[float] = []

    # Budget: sum x <= 1  (fixed)
    lb_A.append([1.0] * n)
    ub_A.append([1.0] * n)
    lb_b.append(1.0)
    ub_b.append(1.0)

    # Return floor: -sum r_i x_i <= -target, with r_i in [low, high] so the
    # coefficient -r_i lies in [-high, -low]. This row carries the uncertainty.
    lb_A.append([-a.return_high for a in assets])
    ub_A.append([-a.return_low for a in assets])
    lb_b.append(-target_return)
    ub_b.append(-target_return)

    # Per-asset caps: x_i <= cap_i  (fixed)
    for i, asset in enumerate(assets):
        row = zeros()
        row[i] = 1.0
        lb_A.append(list(row))
        ub_A.append(list(row))
        lb_b.append(asset.cap)
        ub_b.append(asset.cap)

    return SolveRequest(
        objective=objective,
        lb_A=lb_A,
        ub_A=ub_A,
        lb_b=lb_b,
        ub_b=ub_b,
        options={
            "number_of_scenarios": number_of_scenarios,
            "quality_scenarios": number_of_scenarios,
            "clusters": 3,
        },
    )


def optimize(
    assets: List[Asset], target_return: float, number_of_scenarios: int = 120
) -> Dict:
    """Return the risk/robustness frontier of candidate portfolios."""
    request = build_request(assets, target_return, number_of_scenarios)
    response = solve_problem(request)

    mids = [(a.return_low + a.return_high) / 2 for a in assets]
    portfolios = []
    for sol in response.solutions:
        weights = [max(0.0, w) for w in sol.variables]
        invested = sum(weights)
        portfolios.append(
            {
                "weights": weights,
                "invested": invested,
                "cash": max(0.0, 1.0 - invested),
                "risk": sol.objective_value,
                "robustness": sol.feasibility_probability,
                "expected_return": sum(m * w for m, w in zip(mids, weights)),
            }
        )
    portfolios.sort(key=lambda p: p["robustness"])
    return {
        "assets": [a.name for a in assets],
        "target_return": target_return,
        "portfolios": portfolios,
        "summary": response.summary.model_dump(),
        "warnings": response.warnings,
    }
