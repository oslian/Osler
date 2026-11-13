"""Fitted weights of the independently replicated adaptive cells (protocol v10).

The frozen scorer (``site_adaptive_external_audit.py``) refits each cell's
recipe on five MIMIC-IV training samples and keeps only predictions.  This
module refits the same recipe on the same samples with the same seeds through
the frozen helpers (``_split_fit_calibration``, ``_fit_transform``, the frozen
estimator settings), stores the fitted estimators, and accepts a bundle only if
it reproduces the frozen fit's calibration predictions to float32 precision
relative to the largest prediction (features are float32; a urine-output
forecast of thousands of mL can differ by one float32 step, 2**-12).  A hurdle bundle keeps
the components of the policy the frozen fit selected.

Each bundle also holds the MIMIC-side starting state of the online interval:
the point-correction scale, the v2 quintile interval scale and the conformal
calibration scores (residual / scale ratios, no identifiers).  Not included:
the calibration covariates behind the day-one covariate-shift weights (they are
row-level MIMIC features) and any site state, which starts at deployment.

``predict`` needs a frame prepared like the scorer's (``normalize_cohort``:
complete-stream histories and contract columns).  Research runtime only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from mimic_to_eicu_49_cell_external_audit import feature_subset, fit_continuous_recipe, normalize_cohort, stable_seed
from multi_sample_external_audit import mimic_sample
from osler_jepa.frozen_protocol import sha256
from osler_jepa.interval_scale import fit_interval_scale
from osler_jepa.patient_split import selection_patients
from site_adaptive_external_audit import SCALE_SCHEME
from target_specific_domain_conformal_audit import (
    TARGET_CHANGE_THRESHOLDS,
    _exclude_sibling_features,
    _feature_frame,
    _fit_transform,
    _safe_float,
    _split_fit_calibration,
)

RIDGE_RECIPES = ("measurement_pure_ridge", "observed_treatment_context_ridge", "plain_ridge")
ROBUST_MODELS = {"compact": (7, 50, 10.0), "interaction": (15, 75, 15.0)}
POLICIES = (
    (re.compile(r"robust_(compact|interaction)_delta_([\d.]+)$"), "robust"),
    (re.compile(r"ridge_[\d.]+_robust_(compact|interaction)_([\d.]+)$"), "blend"),
    (re.compile(r"soft_p_x_delta_([\d.]+)$"), "soft"),
    (re.compile(r"hard_p([\d.]+)_x_delta_([\d.]+)$"), "hard"),
)
SAMPLE_ROWS = 15000
TOLERANCE = 1e-6                     # float32 features, relative to the largest prediction


def parse_policy(name: str) -> tuple[str, tuple]:
    for pattern, kind in POLICIES:
        match = pattern.match(name)
        if match:
            return kind, match.groups()
    raise ValueError(f"unknown hurdle policy {name!r}")


def fit_rows_of(frame: pd.DataFrame, train_rows: np.ndarray, target: str, seed: int) -> np.ndarray:
    fit_rows, _ = _split_fit_calibration(frame, train_rows, seed)
    valid = np.isfinite(_safe_float(frame, f"{target}_t")) & np.isfinite(_safe_float(frame, f"future_{target}"))
    return fit_rows[valid[fit_rows]]


def fit_predictor(frame: pd.DataFrame, train_rows: np.ndarray, *, target: str, recipe: str, seed: int,
                  policy: str | None) -> dict:
    """The fitted base predictor of one recipe, with the frozen estimator settings."""

    if recipe == "persistence":
        return {"recipe": recipe, "target": target}
    features = _feature_frame(frame)
    features = feature_subset(features, recipe, target) if recipe in RIDGE_RECIPES else _exclude_sibling_features(features, target)
    rows = fit_rows_of(frame, train_rows, target, seed)
    transform = _fit_transform(features, rows)
    x_fit = transform.apply(features.iloc[rows])
    delta = _safe_float(frame, f"future_{target}")[rows] - _safe_float(frame, f"{target}_t")[rows]
    if recipe in RIDGE_RECIPES:
        limit = max(float(np.quantile(np.abs(delta), 0.99)), 1e-6)
        return {"recipe": recipe, "target": target, "transform": transform, "limit": limit,
                "models": {"ridge": Ridge(alpha=25.0).fit(x_fit, delta)}}
    if recipe != "target_specific_change_hurdle":
        raise ValueError(f"unknown recipe {recipe!r}")
    threshold = float(TARGET_CHANGE_THRESHOLDS[target])
    kind, args = parse_policy(policy)
    models = {}
    if kind in ("soft", "hard"):
        changed = np.abs(delta) >= threshold
        models["classifier"] = LogisticRegression(C=0.25, max_iter=300, solver="lbfgs").fit(x_fit, changed.astype(np.int8))
        models["delta"] = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=120, max_leaf_nodes=7, min_samples_leaf=50,
            l2_regularization=10.0, random_state=seed).fit(x_fit[np.flatnonzero(changed)], delta[changed])
    else:
        name = args[0]
        leaves, minimum_leaf, regularization = ROBUST_MODELS[name]
        models[name] = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_iter=120, max_leaf_nodes=leaves, min_samples_leaf=minimum_leaf,
            l2_regularization=regularization, random_state=seed + 17 + leaves).fit(x_fit, delta)
        if kind == "blend":
            models["ridge"] = Ridge(alpha=25.0).fit(x_fit, delta)
    limit = max(float(np.quantile(np.abs(delta), 0.99)), threshold)
    return {"recipe": recipe, "target": target, "transform": transform, "limit": limit, "policy": policy, "models": models}


def predict(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Forecast of every row (NaN where the current value is missing), as the frozen fit computes it."""

    current = _safe_float(frame, f"{bundle['target']}_t")
    if bundle["recipe"] == "persistence":
        return current
    x = bundle["transform"].apply(_feature_frame(frame))
    models, limit = bundle["models"], bundle["limit"]
    if bundle["recipe"] in RIDGE_RECIPES:
        return current + np.clip(models["ridge"].predict(x), -limit, limit)
    kind, args = parse_policy(bundle["policy"])
    if kind == "robust":
        return current + float(args[1]) * np.clip(models[args[0]].predict(x), -limit, limit)
    if kind == "blend":
        weight = float(args[1])
        return (current + (1.0 - weight) * np.clip(models["ridge"].predict(x), -limit, limit)
                + weight * np.clip(models[args[0]].predict(x), -limit, limit))
    probability = models["classifier"].predict_proba(x)[:, 1]
    delta = np.clip(models["delta"].predict(x), -limit, limit)
    if kind == "soft":
        return current + float(args[0]) * probability * delta
    return current + float(args[1]) * (probability >= float(args[0])) * delta


def reproduces(reproduced: np.ndarray, frozen: np.ndarray) -> bool:
    return bool(np.max(np.abs(reproduced - frozen), initial=0.0) <= TOLERANCE * max(1.0, float(np.max(np.abs(frozen), initial=0.0))))


def interval_start(frame: pd.DataFrame, fitted: dict, target: str) -> dict:
    """The MIMIC-side starting state of the online interval, as ``simulate_sites`` derives it."""

    future, current = _safe_float(frame, f"future_{target}"), _safe_float(frame, f"{target}_t")
    change = np.abs(current - _safe_float(frame, f"lag1__{target}_t"))
    held_out = fitted["calibration_rows"]
    residual = np.abs(fitted["calibration_prediction"] - future[held_out])
    choosing = selection_patients(frame.iloc[held_out]["subject_id"].to_numpy())
    selected, conformal = held_out[choosing], held_out[~choosing]
    scale = fit_interval_scale(SCALE_SCHEME, current[selected], change[selected], residual[choosing],
                               frame.iloc[selected]["subject_id"].to_numpy())
    return {"point_scale": float(np.median(residual[choosing])), "interval_scale": scale,
            "calibration_scores": residual[~choosing] / scale(current[conformal], change[conformal])}


def export_cell(full: pd.DataFrame, *, cell: str, recipe: str, sample: int) -> dict:
    target, horizon = cell.rsplit("@", 1)
    horizon = int(horizon.rstrip("h"))
    frame = mimic_sample(full, target=target, horizon=horizon, sample=sample, rows=SAMPLE_ROWS)
    frame = frame.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
    train = np.arange(len(frame))
    seed = stable_seed(cell, "mimic_to_eicu")
    fitted = fit_continuous_recipe(frame, train, train, target=target, source=recipe, seed=seed)
    bundle = fit_predictor(frame, train, target=target, recipe=recipe, seed=seed, policy=fitted.get("mimic_selected_policy"))
    rows = fitted["calibration_rows"]
    reproduced, frozen = predict(bundle, frame)[rows], fitted["calibration_prediction"]
    difference = float(np.max(np.abs(reproduced - frozen)))
    if not reproduces(reproduced, frozen):
        raise RuntimeError(f"{cell} sample {sample}: refit differs from the frozen fit by {difference}")
    return {**bundle, "cell": cell, "sample": sample, "seed": seed, "horizon_hours": horizon,
            "interval": interval_start(frame, fitted, target), "calibration_max_abs_difference": difference}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("adaptive_replication_registry_20261028.json"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("adaptive_cell_weights_20261028"))
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    template = registry["deployment_form_arguments"]["mimic_template"]
    cells = [entry for entry in registry["cells"] if entry["status"] == "independently_replicated"]
    args.output.mkdir(exist_ok=False)
    entries, raw, loaded = [], None, None
    for entry in sorted(cells, key=lambda item: (item["horizon_hours"], item["cell"])):
        if entry["horizon_hours"] != loaded:
            loaded = entry["horizon_hours"]
            raw = pd.read_parquet(template.format(horizon=loaded))
        full, _ = normalize_cohort(raw, database="MIMIC-IV", target=entry["target"], horizon=loaded, max_rows=0)
        for sample in range(args.samples):
            bundle = export_cell(full, cell=entry["cell"], recipe=entry["recipe"], sample=sample)
            path = args.output / f"{entry['cell'].replace('@', '_')}_s{sample}.joblib"
            joblib.dump(bundle, path, compress=3)
            entries.append({"cell": entry["cell"], "sample": sample, "recipe": entry["recipe"], "policy": bundle.get("policy"),
                            "file": path.name, "sha256": sha256(path),
                            "calibration_max_abs_difference": bundle["calibration_max_abs_difference"]})
            print(json.dumps(entries[-1]), flush=True)
    manifest = {
        "schema": "adaptive_cell_weights.v1",
        "registry": str(args.registry),
        "registry_sha256": sha256(args.registry),
        "protocol_sha256": registry["protocol_sha256"],
        "training_data_sha256": {template.format(horizon=h): sha256(Path(template.format(horizon=h)))
                                 for h in sorted({entry["horizon_hours"] for entry in cells})},
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "joblib": joblib.__version__},
        "loader": "joblib.load(path) -> dict; adaptive_cell_weights.predict(bundle, frame)",
        "not_included": ["day-one covariate-shift calibration covariates (row-level MIMIC features)", "site online state"],
        "bundles": entries,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
