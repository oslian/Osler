"""Externally validate every MIMIC development cell on eICU.

Each source recipe is fit and calibrated on MIMIC-IV only, then applied
unchanged to eICU. eICU labels are read only for final scoring. This is a
factual independent-database test, not a causal or clinical evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from osler_jepa.treatment_units import treatment_feature_actions
from target_specific_domain_conformal_audit import (
    add_history_features,
    TARGET_CHANGE_THRESHOLDS,
    _exclude_sibling_features,
    _feature_frame,
    _fit_hurdle,
    _fit_hypoxemia,
    _fit_transform,
    _patient_bootstrap,
    _safe_float,
    _split_fit_calibration,
)
from teacher_anchored_joint_jepa_audit import _patient_cluster_conformal_quantile


MIN_ROWS = 100
MIN_SUBJECTS = 30
MIN_HOSPITALS = 10
BOOTSTRAP_SAMPLES = 2000
COVERAGE_BOUNDS = (0.87, 0.93)
PREFERRED_MODULE = {
    "glucose": "endocrine_stress",
    "glucose_poc": "endocrine_stress",
    "heart_rate": "cardiovascular_instability",
    "map": "cardiovascular_instability",
    "respiratory_rate": "respiratory",
    "o2sat": "respiratory",
    "hypoxemia_event": "respiratory",
    "platelets": "coagulopathy_heme",
    "hemoglobin": "coagulopathy_heme",
    "hematocrit": "coagulopathy_heme",
    "potassium": "electrolyte_acid_base",
    "sodium": "electrolyte_acid_base",
    "chloride": "electrolyte_acid_base",
    "bicarbonate": "electrolyte_acid_base",
    "urine_output": "aki",
    "creatinine": "aki",
    "bun": "aki",
}
FALLBACK_MODULES = ("aki", "sepsis", "hepatic_failure", "coagulopathy_heme")
DATABASE_PREFIX = {"MIMIC-IV": "MIMIC", "eICU": "EICU", "NWICU": "NWICU", "SICdb": "SICDB", "HiRID": "HIRID", "Zigong": "ZIGONG", "Emory": "EMORY"}


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def module_candidates(target: str) -> tuple[str, ...]:
    target = "o2sat" if target == "hypoxemia_event" else target
    return tuple(dict.fromkeys((PREFERRED_MODULE.get(target, "sepsis"), *FALLBACK_MODULES)))


def mimic_path(project_root: Path, horizon: int) -> Path:
    name = f"mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe.parquet"
    temporary = Path("/private/tmp") / name
    return temporary if temporary.exists() else project_root / name


def eicu_path(root: Path, module: str, horizon: int) -> Path:
    return root / f"eicu_{module}_transitions_{horizon}h.parquet"


def cap_by_subject(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.copy()
    counts = frame.groupby("subject_id", sort=False).size()
    subjects = counts.index.astype(str).to_numpy(dtype=object)
    np.random.default_rng(seed).shuffle(subjects)
    count_map = {str(key): int(value) for key, value in counts.items()}
    keep: list[str] = []
    total = 0
    for subject in subjects:
        keep.append(str(subject))
        total += count_map[str(subject)]
        if total >= max_rows:
            break
    return frame[frame["subject_id"].astype(str).isin(set(keep))].copy()


def normalize_cohort(
    frame: pd.DataFrame,
    *,
    database: str,
    target: str,
    horizon: int,
    max_rows: int,
) -> tuple[pd.DataFrame, dict]:
    target = "o2sat" if target == "hypoxemia_event" else target
    current_column = f"{target}_t"
    future_column = f"{target}_tp{horizon}"
    if current_column not in frame or future_column not in frame:
        return frame.iloc[0:0].copy(), {"reason": "target_columns_missing"}
    view = frame.copy()
    prefix = DATABASE_PREFIX[database]
    view["subject_id"] = prefix + ":" + view["subject_id"].astype(str)
    view["stay_id"] = prefix + ":" + view["stay_id"].astype(str)
    view["anchor_time"] = pd.to_datetime(view["t"], errors="coerce")
    # Histories and the stay clock come from the complete anchor stream, before any
    # selection on this target's future label.
    view = add_history_features(view)
    offset_column = f"{future_column}_offset_hr"
    view[f"future_{target}_available_hr"] = (
        float(horizon) + pd.to_numeric(view[offset_column], errors="coerce").abs()
        if offset_column in view else np.nan
    )
    quality_column = f"future_label_quality_{target}"
    if quality_column in view:
        quality = view[quality_column].astype(str).str.lower()
        view = view[quality.isin({"exact", "near"})].copy()
    current = pd.to_numeric(view[current_column], errors="coerce")
    future = pd.to_numeric(view[future_column], errors="coerce")
    view = view[np.isfinite(current) & np.isfinite(future)].copy()
    original_rows = len(view)
    careunit_column = "unittype" if database == "eICU" else "first_careunit"
    view["careunit"] = view.get(
        careunit_column, pd.Series("unknown", index=view.index)
    ).fillna("unknown").astype(str)
    if database == "MIMIC-IV":
        view["hospitalid"] = "MIMIC-IV"
    else:
        view["hospitalid"] = view.get(
            "hospitalid", pd.Series("unknown", index=view.index)
        ).fillna("unknown").astype(str)
    view[f"future_{target}"] = pd.to_numeric(view[future_column], errors="coerce")
    view.sort_values(["subject_id", "anchor_time"], inplace=True)
    view.reset_index(drop=True, inplace=True)
    view = cap_by_subject(
        view,
        max_rows,
        stable_seed(database, target, horizon),
    ).reset_index(drop=True)
    return view, {
        "original_rows": int(original_rows),
        "sampled_rows": int(len(view)),
        "subjects": int(view["subject_id"].nunique()),
        "hospitals": int(view["hospitalid"].nunique()),
        "quality_filter": "exact_or_near_when_available",
    }


def choose_eicu_cohort(
    root: Path,
    *,
    target: str,
    horizon: int,
    max_rows: int,
) -> tuple[pd.DataFrame, dict]:
    attempts = []
    for module in module_candidates(target):
        path = eicu_path(root, module, horizon)
        if not path.exists():
            attempts.append({"module": module, "reason": "file_missing"})
            continue
        frame, support = normalize_cohort(
            pd.read_parquet(path),
            database="eICU",
            target=target,
            horizon=horizon,
            max_rows=max_rows,
        )
        attempts.append({"module": module, **support})
        if (
            len(frame) >= MIN_ROWS
            and frame["subject_id"].nunique() >= MIN_SUBJECTS
            and frame["hospitalid"].nunique() >= MIN_HOSPITALS
        ):
            return frame, {
                "module": module,
                "path": str(path),
                "support": support,
                "attempts": attempts,
                "selection_policy": "fixed_target_module_then_fixed_fallback_order",
                "performance_used_for_selection": False,
            }
    return pd.DataFrame(), {
        "module": None,
        "path": None,
        "attempts": attempts,
        "selection_policy": "fixed_target_module_then_fixed_fallback_order",
        "performance_used_for_selection": False,
    }


def feature_subset(features: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    features = _exclude_sibling_features(features, target)
    if source == "measurement_pure_ridge":
        return features[[column for column in features if not column.startswith("hist_")]]
    if source == "observed_treatment_context_ridge":
        permitted = treatment_feature_actions(target)
        columns = [
            column
            for column in features
            if not column.startswith("hist_")
            or any(
                column == f"hist_{action}" or column.startswith(f"hist_{action}_")
                for action in permitted
            )
        ]
        return features[columns]
    return features


def fit_ridge_source(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    target: str,
    source: str,
    seed: int,
    require_test_labels: bool = True,
) -> dict:
    fit_rows, calibration_rows = _split_fit_calibration(frame, train_rows, seed)
    features = feature_subset(features, source, target)
    current = _safe_float(frame, f"{target}_t")
    future = _safe_float(frame, f"future_{target}")
    valid = np.isfinite(current) & np.isfinite(future)
    fit_rows = fit_rows[valid[fit_rows]]
    calibration_rows = calibration_rows[valid[calibration_rows]]
    test_rows = test_rows[(valid if require_test_labels else np.isfinite(current))[test_rows]]
    transform = _fit_transform(features, fit_rows)
    x_fit = transform.apply(features.iloc[fit_rows])
    x_cal = transform.apply(features.iloc[calibration_rows])
    x_test = transform.apply(features.iloc[test_rows])
    delta_fit = future[fit_rows] - current[fit_rows]
    limit = max(float(np.quantile(np.abs(delta_fit), 0.99)), 1e-6)
    model = Ridge(alpha=25.0).fit(x_fit, delta_fit)
    cal_delta = np.clip(model.predict(x_cal), -limit, limit)
    test_delta = np.clip(model.predict(x_test), -limit, limit)
    return {
        "test_rows": test_rows,
        "test_prediction": current[test_rows] + test_delta,
        "calibration_rows": calibration_rows,
        "calibration_prediction": current[calibration_rows] + cal_delta,
        "feature_count": int(len(transform.columns)),
    }


def hospital_bootstrap(
    frame: pd.DataFrame,
    rows: np.ndarray,
    candidate_loss: np.ndarray,
    comparator_loss: np.ndarray,
    *,
    seed: int,
) -> dict:
    selected = frame.iloc[np.asarray(rows, dtype=np.int64)]
    patient = pd.DataFrame(
        {
            "hospitalid": selected["hospitalid"].astype(str).to_numpy(),
            "subject_id": selected["subject_id"].astype(str).to_numpy(),
            "delta": np.asarray(candidate_loss) - np.asarray(comparator_loss),
        }
    ).groupby(["hospitalid", "subject_id"], as_index=False)["delta"].mean()
    values = patient.groupby("hospitalid")["delta"].mean().to_numpy(dtype=np.float64)
    point = float(values.mean()) if len(values) else None
    if len(values) < MIN_HOSPITALS:
        return {
            "n_hospitals": int(len(values)),
            "hospital_equal_delta": point,
            "bootstrap_low": None,
            "bootstrap_high": None,
            "pass": False,
        }
    rng = np.random.default_rng(seed)
    sampled = values[
        rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    ].mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "n_hospitals": int(len(values)),
        "hospital_equal_delta": point,
        "bootstrap_low": float(low),
        "bootstrap_high": float(high),
        "pass": bool(high < 0.0),
    }


def group_equal_coverage(
    frame: pd.DataFrame,
    rows: np.ndarray,
    covered: np.ndarray,
    group_column: str,
) -> float:
    groups = frame.iloc[np.asarray(rows, dtype=np.int64)][group_column].astype(str).to_numpy()
    values = np.asarray(covered, dtype=np.float64)
    return float(np.mean([np.mean(values[groups == group]) for group in np.unique(groups)]))


def direction_metrics(
    observed: np.ndarray,
    candidate: np.ndarray,
    current: np.ndarray,
    target: str,
) -> dict:
    threshold = float(TARGET_CHANGE_THRESHOLDS.get(target, 0.0))
    true_delta = observed - current
    predicted_delta = candidate - current
    changed = np.abs(true_delta) >= threshold
    if int(changed.sum()) < 30:
        return {
            "changed_rows": int(changed.sum()),
            "direction_accuracy": None,
            "delta_correlation": None,
        }
    direction = float(np.mean(np.sign(true_delta[changed]) == np.sign(predicted_delta[changed])))
    correlation = float(np.corrcoef(true_delta[changed], predicted_delta[changed])[0, 1])
    return {
        "changed_rows": int(changed.sum()),
        "direction_accuracy": direction,
        "delta_correlation": correlation if np.isfinite(correlation) else None,
    }


def paired_comparison(
    frame: pd.DataFrame,
    rows: np.ndarray,
    candidate_loss: np.ndarray,
    comparator_loss: np.ndarray,
    *,
    seed: int,
) -> dict:
    return {
        "patient": _patient_bootstrap(
            frame, rows, candidate_loss, comparator_loss, seed=seed
        ),
        "hospital": hospital_bootstrap(
            frame, rows, candidate_loss, comparator_loss, seed=seed + 1
        ),
    }


def continuous_metrics(
    frame: pd.DataFrame,
    rows: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    calibration_rows: np.ndarray,
    calibration_prediction: np.ndarray,
    *,
    target: str,
    seed: int,
    comparator_name: str,
) -> dict:
    observed = _safe_float(frame, f"future_{target}")[rows]
    current = _safe_float(frame, f"{target}_t")[rows]
    candidate_loss = np.abs(candidate - observed)
    persistence_loss = np.abs(current - observed)
    comparator_loss = np.abs(comparator - observed)
    calibration_observed = _safe_float(frame, f"future_{target}")[calibration_rows]
    q90, rank, calibration = _patient_cluster_conformal_quantile(
        frame,
        calibration_rows,
        np.abs(calibration_prediction - calibration_observed),
        coverage=0.90,
    )
    covered = candidate_loss <= q90
    interval = {
        "nominal": 0.90,
        "half_width": float(q90),
        "calibration_rank": int(rank),
        "row_coverage": float(np.mean(covered)),
        "patient_equal_coverage": group_equal_coverage(frame, rows, covered, "subject_id"),
        "hospital_equal_coverage": group_equal_coverage(frame, rows, covered, "hospitalid"),
        "calibration": calibration,
    }
    interval["pass"] = bool(
        COVERAGE_BOUNDS[0] <= interval["patient_equal_coverage"] <= COVERAGE_BOUNDS[1]
        and COVERAGE_BOUNDS[0] <= interval["hospital_equal_coverage"] <= COVERAGE_BOUNDS[1]
    )
    versus_persistence = paired_comparison(
        frame, rows, candidate_loss, persistence_loss, seed=seed
    )
    versus_comparator = paired_comparison(
        frame, rows, candidate_loss, comparator_loss, seed=seed + 2
    )
    point_pass = bool(
        versus_persistence["patient"]["pass"]
        and versus_persistence["hospital"]["pass"]
        and versus_comparator["patient"]["pass"]
        and versus_comparator["hospital"]["pass"]
    )
    return {
        "rows": int(len(rows)),
        "subjects": int(frame.iloc[rows]["subject_id"].nunique()),
        "hospitals": int(frame.iloc[rows]["hospitalid"].nunique()),
        "candidate_mae": float(candidate_loss.mean()),
        "persistence_mae": float(persistence_loss.mean()),
        f"{comparator_name}_mae": float(comparator_loss.mean()),
        "versus_persistence": versus_persistence,
        f"versus_{comparator_name}": versus_comparator,
        "direction": direction_metrics(observed, candidate, current, target),
        "interval": interval,
        "point_pass": point_pass,
        "interval_pass": interval["pass"],
        "full_external_pass": bool(point_pass and interval["pass"]),
    }


def fit_continuous_recipe(
    frame: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    target: str,
    source: str,
    seed: int,
    require_test_labels: bool = True,
) -> dict:
    """Fit one frozen source recipe on train rows and predict test rows.

    ``require_test_labels=False`` returns predictions for every test row with a
    current value (forecast-time use, e.g. JEPA anchors); evaluation keeps the
    default and scores labelled rows only.

    Comparators: the change hurdle is compared with plain ridge (all features);
    treatment-context ridge and plain ridge with measurement-pure ridge; the
    measurement-pure ridge with persistence.  ``persistence`` is the safety
    fallback: it predicts the current value and fits nothing; its calibration
    rows are the same held-out training patients the ridge recipes use.
    """

    if source == "persistence":
        current = _safe_float(frame, f"{target}_t")
        valid = np.isfinite(current) & np.isfinite(_safe_float(frame, f"future_{target}"))
        _, calibration_rows = _split_fit_calibration(frame, train_rows, seed)
        calibration_rows = calibration_rows[valid[calibration_rows]]
        test_rows = test_rows[(valid if require_test_labels else np.isfinite(current))[test_rows]]
        return {
            "test_rows": test_rows,
            "test_prediction": current[test_rows],
            "comparator": current[test_rows],
            "comparator_name": "persistence_matched_baseline",
            "calibration_rows": calibration_rows,
            "calibration_prediction": current[calibration_rows],
        }
    features = _feature_frame(frame)
    if source == "target_specific_change_hurdle":
        rows, candidate, comparator, cal_rows, cal_candidate, policy = _fit_hurdle(
            frame,
            features,
            train_rows,
            test_rows,
            seed=seed,
            target=target,
            change_threshold=float(TARGET_CHANGE_THRESHOLDS[target]),
            require_test_labels=require_test_labels,
        )
        return {
            "test_rows": rows,
            "test_prediction": candidate,
            "comparator": comparator,
            "comparator_name": "plain_ridge",
            "calibration_rows": cal_rows,
            "calibration_prediction": cal_candidate,
            "mimic_selected_policy": policy,
        }
    fitted = fit_ridge_source(
        frame,
        features,
        train_rows,
        test_rows,
        target=target,
        source=source,
        seed=seed,
        require_test_labels=require_test_labels,
    )
    if source in ("observed_treatment_context_ridge", "plain_ridge"):
        baseline = fit_ridge_source(
            frame,
            features,
            train_rows,
            test_rows,
            target=target,
            source="measurement_pure_ridge",
            seed=seed,
            require_test_labels=require_test_labels,
        )
        comparator = baseline["test_prediction"]
        comparator_name = "measurement_pure_ridge"
    else:
        comparator = _safe_float(frame, f"{target}_t")[fitted["test_rows"]]
        comparator_name = "persistence_matched_baseline"
    return {**fitted, "comparator": comparator, "comparator_name": comparator_name}


def evaluate_continuous(
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
    *,
    target: str,
    source: str,
    seed: int,
) -> dict:
    frame = pd.concat([mimic, eicu], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    train_rows = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("MIMIC:"))
    test_rows = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("EICU:"))
    fitted = fit_continuous_recipe(
        frame,
        train_rows,
        test_rows,
        target=target,
        source=source,
        seed=seed,
    )
    report = continuous_metrics(
        frame,
        fitted["test_rows"],
        fitted["test_prediction"],
        fitted["comparator"],
        fitted["calibration_rows"],
        fitted["calibration_prediction"],
        target=target,
        seed=seed,
        comparator_name=fitted["comparator_name"],
    )
    if "mimic_selected_policy" in fitted:
        report["mimic_selected_policy"] = fitted["mimic_selected_policy"]
    if "feature_count" in fitted:
        report["feature_count"] = fitted["feature_count"]
    return report


def evaluate_event(
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
    *,
    seed: int,
    external_prefix: str = "EICU:",
) -> dict:
    frame = pd.concat([mimic, eicu], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    train_rows = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("MIMIC:"))
    test_rows = np.flatnonzero(frame["subject_id"].astype(str).str.startswith(external_prefix))
    features = feature_subset(_feature_frame(frame), "measurement_pure_ridge", "o2sat")
    rows, candidate, persistence, prevalence, shrinkage = _fit_hypoxemia(
        frame,
        features,
        train_rows,
        test_rows,
        seed=seed,
        threshold=92.0,
    )
    observed = (_safe_float(frame, "future_o2sat")[rows] <= 92.0).astype(float)
    candidate_loss = (candidate - observed) ** 2
    persistence_loss = (persistence - observed) ** 2
    prevalence_loss = (prevalence - observed) ** 2
    comparisons = {
        "persistence": paired_comparison(
            frame, rows, candidate_loss, persistence_loss, seed=seed
        ),
        "mimic_prevalence": paired_comparison(
            frame, rows, candidate_loss, prevalence_loss, seed=seed + 2
        ),
    }
    point_pass = all(
        report[axis]["pass"]
        for report in comparisons.values()
        for axis in ("patient", "hospital")
    )
    return {
        "rows": int(len(rows)),
        "subjects": int(frame.iloc[rows]["subject_id"].nunique()),
        "hospitals": int(frame.iloc[rows]["hospitalid"].nunique()),
        "event_prevalence": float(observed.mean()),
        "candidate_brier": float(candidate_loss.mean()),
        "persistence_brier": float(persistence_loss.mean()),
        "mimic_prevalence_brier": float(prevalence_loss.mean()),
        "comparisons": comparisons,
        "mimic_calibrated_shrinkage": shrinkage,
        "point_pass": bool(point_pass),
        "interval_pass": None,
        "full_external_pass": bool(point_pass),
    }


def failure_reasons(metrics: dict, task_type: str) -> list[str]:
    reasons = []
    if not metrics.get("point_pass"):
        comparisons = metrics.get("comparisons", {}) if task_type == "binary_event_probability" else {
            key.removeprefix("versus_"): value
            for key, value in metrics.items()
            if key.startswith("versus_") and isinstance(value, dict)
        }
        for comparator, report in comparisons.items():
            for axis in ("patient", "hospital"):
                if axis in report and not report[axis].get("pass"):
                    reasons.append(f"does_not_beat_{comparator}_{axis}")
    if task_type != "binary_event_probability" and not metrics.get("interval_pass"):
        interval = metrics.get("interval", {})
        patient = interval.get("patient_equal_coverage")
        hospital = interval.get("hospital_equal_coverage")
        if patient is None or not COVERAGE_BOUNDS[0] <= patient <= COVERAGE_BOUNDS[1]:
            reasons.append("patient_interval_miscalibration")
        if hospital is None or not COVERAGE_BOUNDS[0] <= hospital <= COVERAGE_BOUNDS[1]:
            reasons.append("hospital_interval_miscalibration")
    return sorted(set(reasons))


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# MIMIC-IV to eICU 49-Cell External Validation",
        "",
        f"- Catalog: `{report['catalog_cells']}`",
        f"- Evaluable: `{report['evaluable_cells']}`",
        f"- Point pass: `{report['point_pass_cells']}`",
        f"- Continuous interval pass: `{report['continuous_interval_pass_cells']}`",
        f"- Full external pass: `{report['full_external_pass_cells']}`",
        f"- Not evaluable: `{report['not_evaluable_cells']}`",
        "",
        "| Cell | MIMIC source | eICU module | Rows / hospitals | Point | Interval | Full | Failure |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["cells"]:
        support = row.get("external_cohort", {}).get("support", {})
        lines.append(
            f"| {row['cell']} | {row['source_model']} | {row.get('eicu_module') or '-'} | "
            f"{support.get('sampled_rows', 0)} / {support.get('hospitals', 0)} | "
            f"{row.get('point_pass')} | {row.get('interval_pass')} | "
            f"{row.get('full_external_pass')} | {', '.join(row.get('failure_reasons', [])) or '-'} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "All model fitting, source selection, clipping, and conformal calibration use MIMIC-IV only. eICU labels are used only for final scoring; no eICU rescue or retuning is allowed.",
        "",
        "eICU is an independent database for these MIMIC-fitted recipes, but it is not a pristine project-wide never-seen lockbox because earlier Osler experiments used eICU.",
        "",
        "All outputs remain factual research-only: `causal_claim_allowed: false`, `clinical_promotion_allowed: false`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = sorted(
        development["cells"],
        key=lambda row: (int(row["horizon_hours"]), str(row["target"])),
    )
    if args.cells:
        selected = set(filter(None, map(str.strip, args.cells.split(","))))
        contracts = [row for row in contracts if row["cell"] in selected]
    reports = []
    mimic_cache: dict[int, pd.DataFrame] = {}
    for contract in contracts:
        cell = str(contract["cell"])
        target = str(contract["target"])
        horizon = int(contract["horizon_hours"])
        source = str(contract["source_model"])
        print(f"external validating {cell} ({source})", flush=True)
        if horizon not in mimic_cache:
            mimic_cache[horizon] = pd.read_parquet(mimic_path(args.project_root, horizon))
        mimic, mimic_support = normalize_cohort(
            mimic_cache[horizon],
            database="MIMIC-IV",
            target=target,
            horizon=horizon,
            max_rows=args.max_rows_per_database,
        )
        eicu, eicu_info = choose_eicu_cohort(
            args.eicu_root,
            target=target,
            horizon=horizon,
            max_rows=args.max_rows_per_database,
        )
        base = {
            "cell": cell,
            "target": target,
            "horizon_hours": horizon,
            "task_type": contract["task_type"],
            "source_model": source,
            "mimic_support": mimic_support,
            "external_cohort": eicu_info,
            "eicu_module": eicu_info.get("module"),
            "model_selection_database": "MIMIC-IV",
            "external_scoring_database": "eICU",
            "eicu_labels_used_for_model_selection": False,
        }
        if mimic.empty or eicu.empty:
            reports.append({
                **base,
                "evaluable": False,
                "point_pass": False,
                "interval_pass": None,
                "full_external_pass": False,
                "failure_reasons": ["insufficient_cross_database_support"],
            })
            continue
        try:
            seed = stable_seed(cell, "mimic_to_eicu")
            metrics = (
                evaluate_event(mimic, eicu, seed=seed)
                if contract["task_type"] == "binary_event_probability"
                else evaluate_continuous(
                    mimic,
                    eicu,
                    target=target,
                    source=source,
                    seed=seed,
                )
            )
        except Exception as error:
            reports.append({
                **base,
                "evaluable": False,
                "point_pass": False,
                "interval_pass": None,
                "full_external_pass": False,
                "failure_reasons": [f"evaluation_error:{type(error).__name__}:{error}"],
            })
            continue
        reports.append({
            **base,
            "evaluable": True,
            "metrics": metrics,
            "point_pass": bool(metrics["point_pass"]),
            "interval_pass": metrics["interval_pass"],
            "full_external_pass": bool(metrics["full_external_pass"]),
            "failure_reasons": failure_reasons(metrics, contract["task_type"]),
        })
        print(json.dumps({
            "cell": cell,
            "point": metrics["point_pass"],
            "interval": metrics["interval_pass"],
            "full": metrics["full_external_pass"],
        }), flush=True)
    by_horizon = defaultdict(lambda: {"cells": 0, "evaluable": 0, "full_pass": 0})
    for row in reports:
        item = by_horizon[str(row["horizon_hours"])]
        item["cells"] += 1
        item["evaluable"] += int(row["evaluable"])
        item["full_pass"] += int(row["full_external_pass"])
    failure_counts = Counter(
        reason for row in reports for reason in row.get("failure_reasons", [])
    )
    output = {
        "schema": "mimic_to_eicu_49_cell_external_audit.v1",
        "comparison_type": "mimic_fitted_recipe_frozen_eicu_transfer",
        "catalog_cells": len(reports),
        "evaluable_cells": sum(row["evaluable"] for row in reports),
        "point_pass_cells": sum(row["point_pass"] for row in reports),
        "continuous_interval_pass_cells": sum(row.get("interval_pass") is True for row in reports),
        "full_external_pass_cells": sum(row["full_external_pass"] for row in reports),
        "not_evaluable_cells": sum(not row["evaluable"] for row in reports),
        "by_horizon": dict(sorted(by_horizon.items(), key=lambda item: int(item[0]))),
        "failure_reason_counts": dict(failure_counts.most_common()),
        "cells": reports,
        "gate": {
            "point": "patient and hospital paired-bootstrap CI upper < 0 versus persistence and matched comparator",
            "interval": "MIMIC-only patient-cluster conformal; eICU patient/hospital equal coverage both in [0.87, 0.93]",
        },
        "external_database_independent_of_model_selection": True,
        "pristine_project_wide_lockbox": False,
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown(output, args.findings)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-report",
        type=Path,
        default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_future_label_repaired"))
    parser.add_argument("--max-rows-per-database", type=int, default=15000)
    parser.add_argument("--cells", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "output": str(args.output),
        "findings": str(args.findings),
        "catalog_cells": report["catalog_cells"],
        "evaluable_cells": report["evaluable_cells"],
        "point_pass_cells": report["point_pass_cells"],
        "full_external_pass_cells": report["full_external_pass_cells"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
