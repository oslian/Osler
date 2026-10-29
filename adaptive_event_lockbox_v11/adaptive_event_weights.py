"""Fitted weights of the independently replicated event cells (registry source protocol v11).

The frozen event scorer (``site_adaptive_event_audit.py``) fits the static
hypoxemia model (``_fit_hypoxemia``: logistic regression C=0.20 on the
measurement-only features, shrunk toward the fit prevalence by a factor chosen
on held-out MIMIC patients) on five MIMIC-IV training samples and keeps only
predictions.  Its feature transform keeps only columns with support in the
MIMIC fit rows, so the model does not depend on the deployment site.  This
module refits the same model on the same samples with the same seeds through
the frozen helpers, stores the fitted transform, estimator, fit prevalence and
shrinkage, and accepts a bundle only if it reproduces the frozen fit's
probabilities on every MIMIC row of the sample.

Each bundle also holds the frozen online Platt scaling configuration; the site
state starts at (1, 0) on deployment and is updated with
``osler_jepa.online_platt.online_platt_scaling`` as outcomes resolve.

``predict`` needs a frame prepared like the scorer's (``normalize_cohort``).
Research runtime only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from adaptive_cell_weights import SAMPLE_ROWS, reproduces
from mimic_to_eicu_49_cell_external_audit import feature_subset, normalize_cohort, stable_seed
from multi_sample_external_audit import mimic_sample
from osler_jepa.frozen_protocol import sha256
from site_adaptive_event_audit import frozen_event_config
from site_adaptive_event_dev import THRESHOLD, static_model
from target_specific_domain_conformal_audit import _feature_frame, _fit_transform, _safe_float, _split_fit_calibration


def fit_static(frame: pd.DataFrame, train_rows: np.ndarray, *, seed: int) -> dict:
    """The fitted static hypoxemia model, with the frozen estimator settings and shrinkage grid."""

    fit_rows, calibration_rows = _split_fit_calibration(frame, train_rows, seed)
    valid = np.isfinite(_safe_float(frame, "o2sat_t")) & np.isfinite(_safe_float(frame, "future_o2sat"))
    fit_rows, calibration_rows = fit_rows[valid[fit_rows]], calibration_rows[valid[calibration_rows]]
    features = feature_subset(_feature_frame(frame), "measurement_pure_ridge", "o2sat")
    transform = _fit_transform(features, fit_rows)
    event = _safe_float(frame, "future_o2sat") <= THRESHOLD
    model = LogisticRegression(C=0.20, max_iter=300, solver="lbfgs").fit(
        transform.apply(features.iloc[fit_rows]), event[fit_rows].astype(np.int8))
    prevalence = float(np.mean(event[fit_rows]))
    raw = model.predict_proba(transform.apply(features.iloc[calibration_rows]))[:, 1]
    observed = event[calibration_rows].astype(np.float64)
    shrinkage = min(np.linspace(0.0, 1.0, 21),
                    key=lambda alpha: float(np.mean((prevalence + alpha * (raw - prevalence) - observed) ** 2)))
    return {"threshold": THRESHOLD, "transform": transform, "model": model, "fit_prevalence": prevalence,
            "shrinkage": float(shrinkage)}


def predict(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Static probability of SpO2 <= threshold at t + h for every row, as the frozen fit computes it."""

    raw = bundle["model"].predict_proba(bundle["transform"].apply(_feature_frame(frame)))[:, 1]
    return bundle["fit_prevalence"] + bundle["shrinkage"] * (raw - bundle["fit_prevalence"])


def export_cell(full: pd.DataFrame, *, cell: str, sample: int, online_platt: dict) -> dict:
    target, horizon = cell.rsplit("@", 1)
    horizon = int(horizon.rstrip("h"))
    frame = mimic_sample(full, target=target, horizon=horizon, sample=sample, rows=SAMPLE_ROWS)
    frame = frame.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
    train = np.arange(len(frame))
    seed = stable_seed(cell, "mimic_to_eicu")
    frozen = static_model(frame, train, train, cell)["probability"]
    bundle = fit_static(frame, train, seed=seed)
    reproduced = predict(bundle, frame)[frozen.index.to_numpy()]
    difference = float(np.max(np.abs(reproduced - frozen.to_numpy())))
    if not reproduces(reproduced, frozen.to_numpy()):
        raise RuntimeError(f"{cell} sample {sample}: refit differs from the frozen fit by {difference}")
    return {**bundle, "cell": cell, "target": target, "sample": sample, "seed": seed, "horizon_hours": horizon,
            "online_platt": online_platt, "mimic_max_abs_difference": difference}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("adaptive_replication_registry_20261029.json"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("adaptive_event_weights_20261029"))
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    source = next(item for item in registry["sources"] if item["task_type"] == "binary_event_probability")
    form = source["deployment_form_arguments"]
    config = frozen_event_config(Path(form["event_config"]))
    online_platt = {"gamma": config.gamma, "rho": config.rho, "radius": config.radius, "initial": [1.0, 0.0]}
    cells = [entry for entry in registry["cells"]
             if entry["status"] == "independently_replicated" and entry["protocol"] == source["protocol"]]
    args.output.mkdir(exist_ok=False)
    entries = []
    for entry in sorted(cells, key=lambda item: (item["horizon_hours"], item["cell"])):
        horizon = entry["horizon_hours"]
        full, _ = normalize_cohort(pd.read_parquet(form["mimic_template"].format(horizon=horizon)), database="MIMIC-IV",
                                   target=entry["target"], horizon=horizon, max_rows=0)
        for sample in range(args.samples):
            bundle = export_cell(full, cell=entry["cell"], sample=sample, online_platt=online_platt)
            path = args.output / f"{entry['cell'].replace('@', '_')}_s{sample}.joblib"
            joblib.dump(bundle, path, compress=3)
            entries.append({"cell": entry["cell"], "sample": sample, "file": path.name, "sha256": sha256(path),
                            "fit_prevalence": bundle["fit_prevalence"], "shrinkage": bundle["shrinkage"],
                            "mimic_max_abs_difference": bundle["mimic_max_abs_difference"]})
            print(json.dumps(entries[-1]), flush=True)
    manifest = {
        "schema": "adaptive_event_weights.v1",
        "registry": str(args.registry),
        "registry_sha256": sha256(args.registry),
        "protocol_sha256": source["protocol_sha256"],
        "training_data_sha256": {form["mimic_template"].format(horizon=h): sha256(Path(form["mimic_template"].format(horizon=h)))
                                 for h in sorted({entry["horizon_hours"] for entry in cells})},
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "joblib": joblib.__version__},
        "loader": "joblib.load(path) -> dict; adaptive_event_weights.predict(bundle, frame) -> static probability; "
                  "osler_jepa.online_platt.online_platt_scaling(..., config=OnlinePlattConfig(**gamma/rho/radius)) "
                  "-> site-recalibrated probability",
        "not_included": ["site online Platt state (starts at (1, 0) on deployment)"],
        "bundles": entries,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
