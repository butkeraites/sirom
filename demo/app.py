"""FastAPI app for the SIROM robust-portfolio demo.

    pip install -e ".[api]"
    uvicorn demo.app:app --port 8800   ->   http://localhost:8800
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .portfolio import DEFAULT_ASSETS, Asset, optimize

app = FastAPI(title="SIROM — Robust Portfolio Demo")

_HERE = os.path.dirname(__file__)


class AssetIn(BaseModel):
    name: str
    return_low: float
    return_high: float
    risk: float = Field(gt=0)
    cap: float = Field(default=0.5, gt=0, le=1)


class OptimizeRequest(BaseModel):
    target_return: float = Field(default=0.10, ge=-0.5, le=1.0)
    number_of_scenarios: int = Field(default=120, ge=10, le=500)
    assets: Optional[List[AssetIn]] = None


def _default_assets_payload():
    return [
        {
            "name": a.name,
            "return_low": a.return_low,
            "return_high": a.return_high,
            "risk": a.risk,
            "cap": a.cap,
        }
        for a in DEFAULT_ASSETS
    ]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open(os.path.join(_HERE, "static", "index.html"), encoding="utf-8") as fh:
        return fh.read()


@app.get("/assets")
def assets() -> dict:
    return {"assets": _default_assets_payload()}


@app.post("/optimize")
def run_optimize(req: OptimizeRequest) -> dict:
    if req.assets:
        chosen = [
            Asset(a.name, a.return_low, a.return_high, a.risk, a.cap)
            for a in req.assets
        ]
    else:
        chosen = DEFAULT_ASSETS
    return optimize(chosen, req.target_return, req.number_of_scenarios)
