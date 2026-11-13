"""HTTP API over ``osler_api.runtime`` (frozen expert cells) and ``osler_api.m_runtime`` (online marginal M).

Cross-origin browser calls are allowed only from the origins listed in ``OSLER_API_ALLOWED_ORIGINS``
(comma-separated); with none listed, only same-origin and server-to-server calls work.  M's per-hospital state is
kept in SQLite at ``OSLER_API_STATE`` (default ``osler_api_state/m_state.sqlite3``).  Research-only.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from osler_api import m_runtime, runtime
from osler_api.m_store import Store

MAX_ANCHORS = 2000
KEY = r"^[A-Za-z0-9._:-]{1,64}$"


class Anchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time: datetime = Field(description="Anchor time; anchors are evenly spaced from the stay's first anchor on")
    values: dict[str, float | None] = Field(
        description="State at the anchor: `<variable>_t` (latest value), `<variable>_age_hr` (hours since it was "
                    "measured), `hist_*` treatment context; missing stays missing (null or omitted), never 0")


class ExpertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    anchors: list[Anchor] = Field(min_length=1, max_length=MAX_ANCHORS, description="6 hours apart")
    cells: list[str] | None = Field(default=None, description="Cells such as `map@6h`; all served cells when omitted")


class MRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hospital_id: str = Field(pattern=KEY, description="The hospital whose online state is used and updated")
    stay_id: str = Field(pattern=KEY, description="A pseudonymous stay key, for the hospital's warm-up count")
    anchors: list[Anchor] = Field(min_length=1, max_length=MAX_ANCHORS, description="3 hours apart")
    joint: bool = Field(default=False, description="Also return copula-coupled members of each block vector")


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    cell: str = Field(description="An issued cell of that forecast, such as `map@6h`")
    value: float = Field(description="The measured value")
    measured_at: datetime = Field(description="When it was measured; must be within the target's label tolerance")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.expert = runtime.load_model()
    app.state.m = m_runtime.load_m()
    app.state.store = Store(Path(os.environ.get("OSLER_API_STATE", runtime.ROOT / "osler_api_state" / "m_state.sqlite3")))
    yield


app = FastAPI(title="Osler forecast API", version="2.0.0", lifespan=lifespan,
              description="Research-only factual forecasts: the frozen expert cells, and the online marginal M with "
                          "per-hospital state.")
origins = [origin.strip() for origin in os.environ.get("OSLER_API_ALLOWED_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def timestamp(value: datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp


def anchors_of(body) -> list[dict]:
    return [{"time": timestamp(a.time), "values": a.values} for a in body.anchors]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/expert/cells")
def expert_cells(request: Request) -> dict:
    model = request.app.state.expert
    return {"cells": [{"cell": c.name, "target": c.target, "horizon_hours": c.horizon_hours, "recipe": c.recipe,
                       "flags": list(c.flags)} for c in model.cells.values()],
            "anchor_spacing_hours": runtime.ANCHOR_SPACING_HOURS, "input_columns": sorted(model.feature_columns),
            "model": model.metadata, "boundary": runtime.BOUNDARY}


@app.post("/v1/expert/forecast")
def expert_forecast(body: ExpertRequest, request: Request) -> dict:
    try:
        return runtime.forecast(request.app.state.expert, anchors_of(body), body.cells)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/m/cells")
def m_cells(request: Request) -> dict:
    model = request.app.state.m
    return {"cells": model.cells, "statuses": model.statuses, "anchor_spacing_hours": model.contract["anchor_spacing_hours"],
            "input_columns": model.contract["state_columns"], "warm_up_stays": m_runtime.WARMUP_STAYS,
            "model": model.metadata, "boundary": m_runtime.BOUNDARY}


@app.post("/v1/m/forecast")
def m_forecast(body: MRequest, request: Request) -> dict:
    try:
        return m_runtime.forecast(request.app.state.m, request.app.state.store, body.hospital_id, body.stay_id,
                                  anchors_of(body), joint=body.joint)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/v1/m/outcomes")
def m_outcome(body: OutcomeRequest, request: Request) -> dict:
    try:
        return m_runtime.post_outcome(request.app.state.m, request.app.state.store, body.forecast_id, body.cell,
                                      body.value, timestamp(body.measured_at))
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
