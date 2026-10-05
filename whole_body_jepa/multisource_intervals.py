"""Patient-disjoint calibration for frozen multi-source point heads.

One outcome-independent anchor per calibration patient supplies a conformal
score. Coverage of all repeated windows is tested empirically, not guaranteed
for every patient or every hospital under distribution shift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEVEL = 0.90
BOUNDS = (0.87, 0.93)
SEEDS = (7, 19, 31, 43, 59, 71, 83)
MIN_PATIENTS = 100
BOUND_MODEL_FILES = (
    "whole_body_joint_jepa.pt",
    "multisource_shared_jepa_head_bundle.joblib",
    "FEATURE_CONTRACT.json",
)


def stable_hash(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def subject_digest(subjects) -> str:
    return hashlib.sha256("\n".join(sorted(set(map(str, subjects)))).encode()).hexdigest()


def patient_split(subjects, seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(subjects).astype(str)
    selected = {v for v in np.unique(values) if stable_hash("interval_role", seed, v) % 2 == 0}
    calibration = np.isin(values, list(selected))
    return calibration, ~calibration


def representative_rows(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Select one anchor per patient without inspecting values or residuals."""
    keys = frame[["subject_id", "stay_id", "t"]].astype(str)
    if keys.duplicated().any():
        raise ValueError("duplicate patient/stay/anchor calibration rows")
    candidates = pd.DataFrame({
        "subject": keys["subject_id"].to_numpy(),
        "hash": [stable_hash("interval_anchor", seed, *row) for row in keys.itertuples(index=False, name=None)],
        "row": np.arange(len(frame)),
    })
    return candidates.sort_values(["subject", "hash"]).drop_duplicates("subject")["row"].to_numpy()


def conformal_radius(scores, level: float = LEVEL) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("calibration scores must be finite nonnegative values")
    if not 0 < level < 1:
        raise ValueError("invalid coverage level")
    rank = int(np.ceil((len(scores) + 1) * level))
    if rank > len(scores):
        raise ValueError("insufficient patients for a finite conformal radius")
    return float(np.partition(scores, rank - 1)[rank - 1])


def fit_radius(frame: pd.DataFrame, seed: int) -> float:
    if frame["subject_id"].nunique() < MIN_PATIENTS:
        raise ValueError("insufficient calibration patients")
    selected = representative_rows(frame, seed)
    return conformal_radius(frame["residual"].to_numpy()[selected])


def coverage(frame: pd.DataFrame, radius: float) -> dict:
    if frame.empty or frame["subject_id"].nunique() < MIN_PATIENTS:
        return {"available": False, "pass": False, "reason": "insufficient_test_patients"}
    residual = frame["residual"].to_numpy(dtype=float)
    if not np.isfinite(residual).all() or not np.isfinite(radius) or radius < 0:
        return {"available": False, "pass": False, "reason": "invalid_interval_or_residual"}
    hits = pd.DataFrame({"subject": frame["subject_id"].to_numpy(), "hit": residual <= radius})
    row = float(hits["hit"].mean())
    patient = float(hits.groupby("subject")["hit"].mean().mean())
    return {"available": True, "rows": len(frame), "patients": int(frame["subject_id"].nunique()),
            "row_coverage": row, "patient_equal_coverage": patient,
            "pass": BOUNDS[0] <= row <= BOUNDS[1] and BOUNDS[0] <= patient <= BOUNDS[1]}


def domain_coverage(frame: pd.DataFrame, radius: float) -> dict:
    units = frame["careunit"].fillna("unknown").astype(str)
    folds, retained, unsupported = [], [], []
    for unit in sorted(units.unique()):
        selected = frame.loc[units.eq(unit)]
        if unit.lower() in {"unknown", "nan", "none", ""} or len(selected) < 100 or selected["subject_id"].nunique() < 30:
            unsupported.append(unit)
            continue
        hits = pd.Series(selected["residual"].to_numpy() <= radius, index=selected["subject_id"])
        folds.append({"careunit": unit, "rows": len(selected),
                      "patients": int(selected["subject_id"].nunique()),
                      "row_coverage": float(hits.mean()),
                      "patient_equal_coverage": float(hits.groupby(level=0).mean().mean())})
        retained.append(selected)
    if len(folds) < 2:
        return {"available": False, "pass": False, "reason": "fewer_than_two_supported_careunits", "folds": folds}
    pooled = coverage(pd.concat(retained), radius)
    equal = float(np.mean([f["patient_equal_coverage"] for f in folds]))
    return {"available": True, "pass": pooled["pass"] and BOUNDS[0] <= equal <= BOUNDS[1],
            "pooled": pooled, "careunit_equal_coverage": equal, "folds": folds,
            "unsupported_careunits": unsupported,
            "scope": "pooled_and_equal_supported_careunits_not_every_subgroup"}


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return {"available": False, "pass": False, "groups": len(values)}
    rng = np.random.default_rng(seed)
    estimates = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(1000)])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"available": True, "groups": len(values), "delta_mae": float(values.mean()),
            "ci95": [float(low), float(high)], "pass": bool(high < 0 and values.mean() < 0)}


def point_gates(frame: pd.DataFrame, cell: str) -> dict:
    delta = frame["residual"] - (frame["truth"] - frame["current"]).abs()
    work = frame[["subject_id", "careunit", "t"]].copy()
    work["delta"] = delta.to_numpy()
    patient = _bootstrap(work.groupby("subject_id")["delta"].mean().to_numpy(), stable_hash(cell, "point_patient"))
    known = ~work["careunit"].fillna("unknown").astype(str).str.lower().isin({"unknown", "nan", "none", ""})
    unit_values = work.loc[known].groupby(["careunit", "subject_id"])["delta"].mean().groupby(level=0).mean()
    unit = _bootstrap(unit_values.to_numpy(), stable_hash(cell, "point_careunit"))
    cutoff = pd.to_datetime(work["t"]).quantile(0.8)
    late = work.loc[pd.to_datetime(work["t"]) >= cutoff]
    time = _bootstrap(late.groupby("subject_id")["delta"].mean().to_numpy(), stable_hash(cell, "point_time"))
    return {"patient": patient, "careunit": unit, "late_time": time,
            "pass": patient["pass"] and unit["pass"] and time["pass"]}


def audit_intervals(mimic: pd.DataFrame, external: pd.DataFrame, cell: str, horizon: int) -> dict:
    if set(mimic["subject_id"]) & set(external["subject_id"]):
        raise ValueError("calibration and external patient identities overlap")
    support = []
    for seed in SEEDS:
        cal, test = patient_split(mimic["subject_id"], seed)
        support.append({"seed": seed, "calibration_patients": int(mimic.loc[cal, "subject_id"].nunique()),
                        "test_patients": int(mimic.loc[test, "subject_id"].nunique())})
    if any(min(s["calibration_patients"], s["test_patients"]) < MIN_PATIENTS for s in support):
        return {"radius": None, "mean_width": None, "median_width": None, "level": LEVEL,
                "patient_splits": support, "nwicu_patient": {"available": False, "pass": False},
                "corrected_nwicu_point_gate": point_gates(external, cell),
                "runtime_interval_eligible": False,
                "failure_reasons": ["insufficient_unused_mimic_calibration_patients"]}
    splits = []
    primary_radius = None
    primary_test = None
    for seed in SEEDS:
        cal, test = patient_split(mimic["subject_id"], seed)
        radius = fit_radius(mimic.loc[cal], seed)
        score = coverage(mimic.loc[test], radius)
        splits.append({"seed": seed, "radius": radius, "calibration_patients": int(mimic.loc[cal, "subject_id"].nunique()), **score})
        if seed == SEEDS[0]:
            primary_radius, primary_test = radius, mimic.loc[test]
    assert primary_radius is not None and primary_test is not None
    unit = domain_coverage(primary_test, primary_radius)

    # This is a chronological stress test, with patients and possible future
    # label windows purged across the boundary, not a new tuned calibrator.
    times = pd.to_datetime(mimic["t"])
    cutoff = times.quantile(0.7)
    late = mimic.loc[times >= cutoff]
    early = mimic.loc[(times + pd.to_timedelta(horizon + 2, unit="h") < cutoff)
                      & ~mimic["subject_id"].isin(late["subject_id"])]
    if early["subject_id"].nunique() >= MIN_PATIENTS:
        time_radius = fit_radius(early, SEEDS[0])
        time = {**coverage(late, time_radius), "radius": time_radius,
                "patient_overlap": 0, "future_window_purge_hours": horizon + 2}
    else:
        time = {"available": False, "pass": False, "reason": "insufficient_purged_early_patients"}

    ext_patient = coverage(external, primary_radius)
    ext_unit = domain_coverage(external, primary_radius)
    ext_times = pd.to_datetime(external["t"])
    ext_late = coverage(external.loc[ext_times >= ext_times.quantile(0.8)], primary_radius)
    points = point_gates(external, cell)
    reasons = []
    for name, passed in [("mimic_patient_splits", all(s["pass"] for s in splits)),
                         ("mimic_careunit", unit["pass"]), ("mimic_time", time["pass"]),
                         ("nwicu_patient", ext_patient["pass"]), ("nwicu_careunit", ext_unit["pass"]),
                         ("nwicu_time", ext_late["pass"]), ("nwicu_point", points["pass"])]:
        if not passed:
            reasons.append(name)
    return {"radius": primary_radius, "mean_width": 2 * primary_radius,
            "median_width": 2 * primary_radius, "level": LEVEL,
            "patient_splits": splits, "mimic_careunit": unit, "mimic_forward_time": time,
            "nwicu_patient": ext_patient, "nwicu_careunit": ext_unit,
            "nwicu_late_time": ext_late, "corrected_nwicu_point_gate": points,
            "runtime_interval_eligible": not reasons, "failure_reasons": reasons}


def load_calibrator(path: Path, model_dir: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "whole_body_jepa.multisource_interval_calibrator.v1":
        raise ValueError("unsupported multi-source calibrator")
    if set(payload.get("release_binding", {})) != set(BOUND_MODEL_FILES):
        raise ValueError("incomplete or unexpected calibrator model binding")
    for name, digest in payload["release_binding"].items():
        if sha256(model_dir / name) != digest:
            raise ValueError(f"calibrator model binding mismatch: {name}")
    return payload


def prediction_interval(calibrator: dict, cell: str, point: float | None) -> dict:
    row = calibrator["cells"].get(cell, {})
    if point is None or not np.isfinite(point):
        return {"lower": None, "upper": None, "interval_status": "missing_point"}
    if not row.get("runtime_interval_eligible"):
        return {"lower": None, "upper": None, "interval_status": "needs_calibration_audit"}
    radius = row["radius"]
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("invalid calibrated radius")
    return {"lower": float(point - radius), "upper": float(point + radius),
            "interval_status": "multisource_patient_split_90_empirically_validated"}
