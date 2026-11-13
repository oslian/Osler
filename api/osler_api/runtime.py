"""Forecasts of the frozen, independently replicated expert cells for one ICU stay.

The bundles are the ones ``adaptive_cell_weights.py`` exported and verified
against the frozen fits.  Every bundle's SHA-256 must match the manifest, the
manifest must name the registry file that is loaded, and scikit-learn must be the
version the bundles were pickled with; otherwise loading fails.

Features are built as the frozen scorer builds them: previous-anchor values,
per-hour slopes and hours since the stay's first anchor come from
``history_features`` over the complete anchor stream the caller sends, then
``_feature_frame``.  Caller-supplied history columns are dropped and recomputed.

What this runtime does not do:
- online site adaptation (point correction, DtACI): it needs each hospital's own
  outcomes after a warm-up, so every point is the day-one (uncorrected) forecast
  and every interval is ``null``;
- the own-target residual, the online Mondrian marginal and the copula layer:
  they are candidates without a second external lockbox and are not exported.

Research runtime only: ``causal_claim_allowed`` and ``clinical_promotion_allowed``
are false.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from adaptive_cell_weights import predict
from osler_jepa.frozen_protocol import sha256
from target_specific_domain_conformal_audit import history_features

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "adaptive_cell_weights_20261102"
REGISTRY = ROOT / "adaptive_replication_registry_20261102.json"
ANCHOR_SPACING_HOURS = 6.0           # every MIMIC transition file the cells were fitted on has 6-hour anchors
SPACING_TOLERANCE_HOURS = 1.0 / 60.0
DERIVED_PREFIXES = ("lag1__", "slope1h__", "future_", "_source_")
DERIVED_COLUMNS = {"hours_since_onset", "t", "anchor_time", "stay_id", "subject_id"}
BOUNDARY = {
    "research_only": True,
    "causal_claim_allowed": False,
    "clinical_promotion_allowed": False,
    "online_site_adaptation": "not_applied",
    "interval_status": "not_available_without_site_online_state",
    "point_form": "median of the five MIMIC training samples' day-one forecasts; the median itself was not scored in any lockbox",
}


@dataclass(frozen=True)
class Cell:
    name: str
    target: str
    horizon_hours: int
    recipe: str
    flags: tuple[str, ...]
    bundles: tuple[dict, ...]


@dataclass(frozen=True)
class Model:
    cells: dict[str, Cell]
    feature_columns: frozenset[str]
    metadata: dict


def load_model(weights: Path = WEIGHTS, registry: Path = REGISTRY) -> Model:
    manifest = json.loads((weights / "MANIFEST.json").read_text(encoding="utf-8"))
    registry_sha = sha256(registry)
    if registry_sha != manifest["registry_sha256"]:
        raise RuntimeError(f"{registry.name} does not match the manifest's registry")
    if sklearn.__version__ != manifest["versions"]["scikit-learn"]:
        raise RuntimeError(f"bundles need scikit-learn {manifest['versions']['scikit-learn']}, found {sklearn.__version__}")
    entries = {entry["cell"]: entry for entry in json.loads(registry.read_text(encoding="utf-8"))["cells"]
               if entry["task_type"] == "continuous_value" and entry["status"] == "independently_replicated"}
    grouped: dict[str, list[dict]] = {}
    for item in manifest["bundles"]:
        path = weights / item["file"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"{item['file']} does not match its manifest checksum")
        grouped.setdefault(item["cell"], []).append(joblib.load(path))
    if set(grouped) != set(entries):
        raise RuntimeError("the manifest's cells differ from the registry's replicated continuous cells")
    cells, columns = {}, set()
    for name, bundles in sorted(grouped.items()):
        entry = entries[name]
        cells[name] = Cell(name=name, target=entry["target"], horizon_hours=int(entry["horizon_hours"]),
                           recipe=entry["recipe"], flags=tuple(entry.get("flags", ())),
                           bundles=tuple(sorted(bundles, key=lambda bundle: bundle["sample"])))
        for bundle in bundles:
            columns.add(f"{bundle['target']}_t")
            if "transform" in bundle:
                columns.update(bundle["transform"].columns)
    metadata = {"registry": registry.name, "registry_sha256": registry_sha, "weights": weights.name,
                "schema": manifest["schema"], "training_database": "MIMIC-IV v3.1"}
    return Model(cells=cells, feature_columns=frozenset(columns), metadata=metadata)


def stay_frame(anchors: list[dict], spacing_hours: float = ANCHOR_SPACING_HOURS) -> tuple[pd.DataFrame, list[str]]:
    """The stay's anchor stream with recomputed histories, and the names of the derived inputs it dropped.

    ``anchors`` are ``{"time": Timestamp, "values": {column: float | None}}`` from the stay's first
    anchor on, one every ``spacing_hours``.
    """

    if not anchors:
        raise ValueError("at least one anchor is required")
    times = pd.to_datetime([anchor["time"] for anchor in anchors], utc=True).tz_localize(None)
    if times.has_duplicates:
        raise ValueError("anchor times must be unique")
    order = np.argsort(times.to_numpy(), kind="stable")
    gaps = np.diff(times.to_numpy()[order]).astype("timedelta64[s]").astype(np.float64) / 3600.0
    if np.any(np.abs(gaps - spacing_hours) > SPACING_TOLERANCE_HOURS):
        raise ValueError(f"consecutive anchors must be {spacing_hours:g} hours apart, as in the training data")
    values = pd.DataFrame([anchors[i]["values"] for i in order], dtype=np.float64)
    derived = [c for c in values.columns if c.startswith(DERIVED_PREFIXES) or c in DERIVED_COLUMNS]
    values = values.drop(columns=derived)
    keys = pd.DataFrame({"stay_id": "request", "anchor_time": times[order]})
    frame = pd.concat([keys, values], axis=1)
    return pd.concat([frame, history_features(frame)], axis=1), derived


def forecast(model: Model, anchors: list[dict], cells: list[str] | None = None) -> dict:
    """Forecasts at the stay's latest anchor, for ``cells`` (all served cells when ``None``)."""

    names = sorted(model.cells) if cells is None else list(dict.fromkeys(cells))
    unknown = [name for name in names if name not in model.cells]
    if unknown:
        raise ValueError(f"unknown cells: {unknown}")
    frame, derived = stay_frame(anchors)
    last = frame.iloc[[-1]].reset_index(drop=True)          # carries its histories, so ``predict`` does not recompute them
    anchor_time = last["anchor_time"].iloc[0]
    results = []
    for name in names:
        cell = model.cells[name]
        current = float(pd.to_numeric(last.get(f"{cell.target}_t", pd.Series([np.nan])), errors="coerce").iloc[0])
        result = {"cell": name, "target": cell.target, "horizon_hours": cell.horizon_hours, "recipe": cell.recipe,
                  "forecast_time": (anchor_time + pd.Timedelta(hours=cell.horizon_hours)).isoformat(),
                  "current_value": current if np.isfinite(current) else None, "flags": list(cell.flags),
                  "interval": None}
        if not np.isfinite(current):
            results.append({**result, "status": "current_value_missing", "point": None, "sample_points": None})
            continue
        samples = [float(predict(bundle, last)[0]) for bundle in cell.bundles]
        results.append({**result, "status": "ok", "point": float(np.median(samples)), "sample_points": samples})
    supplied = {column for anchor in anchors for column in anchor["values"]}
    ignored = sorted(supplied - model.feature_columns - set(derived))
    return {"anchor_time": anchor_time.isoformat(), "anchors_received": len(anchors), "forecasts": results,
            "ignored_columns": ignored, "recomputed_columns": sorted(derived), "model": model.metadata,
            "boundary": BOUNDARY}
