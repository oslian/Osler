"""FastAPI app for private, in-memory BUN@3h CSV forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from bun_forecast_demo.examples import example_catalog, example_path
from bun_forecast_demo.runtime import capabilities, predict_csv


APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "static" / "index.html"

app = FastAPI(
    title="Osler Personalized BUN 3h Forecast",
    version="1.1.0",
    description="Research-only factual BUN forecast from an uploaded trajectory.",
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")


@app.get("/api/examples")
def get_examples() -> dict[str, Any]:
    return {"examples": example_catalog()}


@app.get("/examples/{example_id}.csv")
def get_example_csv(example_id: str) -> FileResponse:
    try:
        path = example_path(example_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Unknown example") from exc
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/capabilities")
def get_capabilities() -> dict[str, Any]:
    return capabilities()


@app.post("/api/predict-csv")
async def post_predict_csv(request: Request) -> dict[str, Any]:
    try:
        return predict_csv(await request.body())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
