"""Diagnose whether MIMIC-to-eICU interval failures follow the measurement process.

Two diagnostics, both factual research-only:

1. eICU hospital-level association.  The frozen MIMIC recipe and MIMIC-only
   conformal width are reproduced exactly.  eICU labels are read only to score
   coverage, never to fit, select, or calibrate anything.  Each hospital's
   coverage is compared with its measurement-process covariates (input age and
   realized label time) and with case-mix covariates.  MIMIC calibration
   residuals are then importance-reweighted to each hospital's covariate mix,
   using eICU inputs only, to test which covariate family predicts the
   hospital's coverage.

2. MIMIC-only intervention.  A recipe is fit on MIMIC training patients and
   scored on held-out MIMIC patients whose measurement process is changed while
   the patients stay fixed: the target lab input is rebuilt from raw MIMIC lab
   times on sparser ordering schedules, and lab labels are re-read at earlier or
   later realized offsets.  No eICU data enters this diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mimic_to_eicu_49_cell_external_audit import (
    choose_eicu_cohort,
    fit_continuous_recipe,
    group_equal_coverage,
    mimic_path,
    normalize_cohort,
    stable_seed,
)
from mimiciv_observation_transition_extract import LAB_ITEMIDS, clean_measurements
from target_specific_domain_conformal_audit import _safe_float
from teacher_anchored_joint_jepa_audit import _patient_cluster_conformal_quantile


HORIZON = 6
LOOKBACK_HOURS = 6.0
MIMIC_LABEL_TOLERANCE_HOURS = 2.0
WIDE_LABEL_TOLERANCE_HOURS = 3.5
LAB_TARGETS = ("creatinine", "potassium", "hemoglobin", "hematocrit", "glucose")
MIN_HOSPITAL_SUBJECTS = 20
MIN_HOSPITAL_ROWS = 100
MIN_BIN_ROWS = 30
THINNING_SPACINGS_HOURS = (0.0, 4.0, 8.0, 12.0)
LABEL_OFFSET_BINS = (
    ("early_2.5_to_4.5h", 2.5, 4.5),
    ("near_4.5_to_7.5h", 4.5, 7.5),
    ("late_7.5_to_9.5h", 7.5, 9.5),
)


# ---------------------------------------------------------------------------
# Raw MIMIC lab event times (needed only for lab label-timing interventions)
# ---------------------------------------------------------------------------


def extract_mimic_lab_times(mimic_dir: Path, output: Path, chunksize: int) -> dict:
    """Cache cleaned ICU-stay lab events for the lab targets of this diagnostic."""

    item_to_var = {
        int(itemid): var
        for var in LAB_TARGETS
        for itemid in LAB_ITEMIDS[var]
    }
    stays = pd.read_csv(
        mimic_dir / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    pieces = []
    scanned = 0
    for chunk in pd.read_csv(
        mimic_dir / "hosp" / "labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        dtype={"hadm_id": "float64", "itemid": "int64", "valuenum": "float64"},
        chunksize=chunksize,
    ):
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(item_to_var) & chunk["valuenum"].notna()]
        chunk = chunk[chunk["hadm_id"].notna()]
        if chunk.empty:
            continue
        chunk = chunk.assign(
            hadm_id=chunk["hadm_id"].astype("int64"),
            charttime=pd.to_datetime(chunk["charttime"]),
            var=chunk["itemid"].map(item_to_var),
        )
        joined = chunk.merge(stays, on="hadm_id", how="inner")
        joined = joined[
            (joined["charttime"] >= joined["intime"])
            & (joined["charttime"] <= joined["outtime"])
        ]
        pieces.append(joined[["subject_id", "stay_id", "charttime", "var", "valuenum"]])
        print(f"scanned {scanned:,} lab rows; kept {sum(map(len, pieces)):,}", flush=True)
    events = clean_measurements(pd.concat(pieces, ignore_index=True))
    events = events[["stay_id", "charttime", "var", "valuenum"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output, index=False)
    summary = {
        "scanned_lab_rows": int(scanned),
        "cached_rows": int(len(events)),
        "stays": int(events["stay_id"].nunique()),
        "rows_by_var": {str(k): int(v) for k, v in events["var"].value_counts().items()},
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def development_cells(report_path: Path) -> list[dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return sorted(
        (
            row
            for row in report["cells"]
            if int(row["horizon_hours"]) == HORIZON
            and row["task_type"] != "binary_event_probability"
        ),
        key=lambda row: str(row["target"]),
    )


def raw_stay_ids(frame: pd.DataFrame) -> np.ndarray:
    text = frame["stay_id"].astype(str).str.split(":", n=1).str[-1]
    return pd.to_numeric(text, errors="coerce").to_numpy(dtype=np.float64)


def anchor_hours(frame: pd.DataFrame) -> np.ndarray:
    """Anchor times in hours, independent of the stored datetime resolution."""

    times = pd.to_datetime(frame["anchor_time"]).to_numpy().astype("datetime64[ns]")
    return times.astype(np.int64) / 3.6e12


def patient_equal(subjects: np.ndarray, values: np.ndarray) -> float:
    table = pd.DataFrame({"s": subjects, "v": np.asarray(values, dtype=np.float64)})
    return float(table.groupby("s")["v"].mean().mean())


def conformal_width(frame: pd.DataFrame, fitted: dict, target: str) -> float:
    observed = _safe_float(frame, f"future_{target}")[fitted["calibration_rows"]]
    width, _, _ = _patient_cluster_conformal_quantile(
        frame,
        fitted["calibration_rows"],
        np.abs(fitted["calibration_prediction"] - observed),
        coverage=0.90,
    )
    return float(width)


def paired_coverage_shift(
    subjects: np.ndarray,
    reference: np.ndarray,
    changed: np.ndarray,
    *,
    seed: int,
    samples: int = 1000,
) -> dict:
    """Patient-equal coverage change with a patient-cluster bootstrap CI."""

    table = pd.DataFrame({
        "s": subjects,
        "a": np.asarray(reference, dtype=np.float64),
        "b": np.asarray(changed, dtype=np.float64),
    }).groupby("s")[["a", "b"]].mean()
    delta = (table["b"] - table["a"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "patients": int(len(table)),
        "reference_coverage": float(table["a"].mean()),
        "changed_coverage": float(table["b"].mean()),
        "coverage_shift": float(delta.mean()),
        "shift_ci95": [float(low), float(high)],
    }


def lab_labels_in_window(
    frame: pd.DataFrame,
    rows: np.ndarray,
    events: pd.DataFrame,
    target: str,
    low_hours: float,
    high_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lab value nearest to anchor + horizon inside an offset window."""

    values = np.full(len(rows), np.nan)
    offsets = np.full(len(rows), np.nan)
    lab = events.loc[events["var"] == target, ["stay_id", "charttime", "valuenum"]]
    queries = pd.DataFrame({
        "position": np.arange(len(rows)),
        "stay_id": raw_stay_ids(frame.iloc[rows]),
        "anchor": pd.to_datetime(frame.iloc[rows]["anchor_time"]).to_numpy(),
    })
    queries = queries[np.isfinite(queries["stay_id"])].astype({"stay_id": "int64"})
    joined = queries.merge(lab, on="stay_id", how="inner")
    offset = (joined["charttime"] - joined["anchor"]).dt.total_seconds() / 3600.0
    joined = joined.assign(offset=offset, distance=np.abs(offset - HORIZON))
    joined = joined[(joined["offset"] >= low_hours) & (joined["offset"] <= high_hours)]
    joined = joined.sort_values(["position", "distance", "charttime"]).drop_duplicates("position")
    values[joined["position"].to_numpy()] = joined["valuenum"].to_numpy()
    offsets[joined["position"].to_numpy()] = joined["offset"].to_numpy()
    return values, offsets


# ---------------------------------------------------------------------------
# Diagnostic 1: eICU hospital-level association
# ---------------------------------------------------------------------------


def _quantile_bins(reference: np.ndarray, values: np.ndarray, count: int = 5) -> np.ndarray:
    finite = reference[np.isfinite(reference)]
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, count + 1)[1:-1]))
    bins = np.digitize(values, edges)
    return np.where(np.isfinite(values), bins, -1)


def _offset_bins(values: np.ndarray) -> np.ndarray:
    edges = np.array([4.5, 5.5, 6.5, 7.5])
    bins = np.digitize(values, edges)
    return np.where(np.isfinite(values), bins, -1)


def _reweighted_coverage(
    calibration_bins: np.ndarray,
    calibration_covered: np.ndarray,
    hospital_bins: np.ndarray,
) -> float:
    """MIMIC calibration coverage reweighted to one hospital's bin mix."""

    overall = float(np.mean(calibration_covered))
    total = 0.0
    for value in np.unique(hospital_bins):
        share = float(np.mean(hospital_bins == value))
        mask = calibration_bins == value
        rate = (
            float(np.mean(calibration_covered[mask]))
            if int(mask.sum()) >= MIN_BIN_ROWS
            else overall
        )
        total += share * rate
    return total


def hospital_association(
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
    *,
    cell: str,
    target: str,
    source: str,
    lab_events: pd.DataFrame | None,
    stored_interval: dict | None,
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
        seed=stable_seed(cell, "mimic_to_eicu"),
    )
    width = conformal_width(frame, fitted, target)
    rows = fitted["test_rows"]
    observed = _safe_float(frame, f"future_{target}")[rows]
    covered = np.abs(fitted["test_prediction"] - observed) <= width
    reproduction = {
        "patient_equal_coverage": group_equal_coverage(frame, rows, covered, "subject_id"),
        "hospital_equal_coverage": group_equal_coverage(frame, rows, covered, "hospitalid"),
        "half_width": width,
    }
    if stored_interval:
        reproduction["matches_frozen_audit"] = bool(
            abs(reproduction["patient_equal_coverage"] - stored_interval["patient_equal_coverage"]) < 1e-9
            and abs(reproduction["hospital_equal_coverage"] - stored_interval["hospital_equal_coverage"]) < 1e-9
        )

    cal_rows = fitted["calibration_rows"]
    cal_observed = _safe_float(frame, f"future_{target}")[cal_rows]
    cal_covered = (np.abs(fitted["calibration_prediction"] - cal_observed) <= width).astype(float)
    cal_age = _safe_float(frame, f"{target}_age_hr")[cal_rows]
    cal_value = _safe_float(frame, f"{target}_t")[cal_rows]
    cal_offset = np.full(len(cal_rows), np.nan)
    label_offset_available = lab_events is not None and target in LAB_TARGETS
    offset_match_rate = None
    if label_offset_available:
        values, cal_offset = lab_labels_in_window(
            frame,
            cal_rows,
            lab_events,
            target,
            HORIZON - MIMIC_LABEL_TOLERANCE_HOURS,
            HORIZON + MIMIC_LABEL_TOLERANCE_HOURS,
        )
        offset_match_rate = float(np.mean(np.isclose(values, cal_observed, atol=1e-6)))

    test = frame.iloc[rows]
    test_age = _safe_float(frame, f"{target}_age_hr")[rows]
    test_value = _safe_float(frame, f"{target}_t")[rows]
    test_offset = _safe_float(frame, f"future_delta_t_hr_{target}")[rows]

    families = {
        "process_input_age": (
            _quantile_bins(cal_age, cal_age),
            _quantile_bins(cal_age, test_age),
        ),
        "case_mix_current_value": (
            _quantile_bins(cal_value, cal_value),
            _quantile_bins(cal_value, test_value),
        ),
    }
    if label_offset_available:
        families["process_label_offset"] = (_offset_bins(cal_offset), _offset_bins(test_offset))
        families["process_age_and_label_offset"] = (
            families["process_input_age"][0] * 10 + families["process_label_offset"][0],
            families["process_input_age"][1] * 10 + families["process_label_offset"][1],
        )

    hospital_ids = test["hospitalid"].astype(str).to_numpy()
    subjects = test["subject_id"].astype(str).to_numpy()
    hospitals = []
    for hospital in np.unique(hospital_ids):
        mask = hospital_ids == hospital
        if int(mask.sum()) < MIN_HOSPITAL_ROWS or len(np.unique(subjects[mask])) < MIN_HOSPITAL_SUBJECTS:
            continue
        item = {
            "hospital": hospital,
            "rows": int(mask.sum()),
            "subjects": int(len(np.unique(subjects[mask]))),
            "row_coverage": float(np.mean(covered[mask])),
            "patient_equal_coverage": patient_equal(subjects[mask], covered[mask]),
            "median_input_age_hr": float(np.nanmedian(test_age[mask])),
            "median_abs_label_offset_from_horizon_hr": float(np.nanmedian(np.abs(test_offset[mask] - HORIZON))),
            "share_label_outside_mimic_window": float(np.nanmean(
                np.abs(test_offset[mask] - HORIZON) > MIMIC_LABEL_TOLERANCE_HOURS
            )),
            "median_current_value": float(np.nanmedian(test_value[mask])),
        }
        for family, (cal_bins, test_bins) in families.items():
            item[f"predicted_coverage__{family}"] = _reweighted_coverage(
                cal_bins, cal_covered, test_bins[mask]
            )
        hospitals.append(item)

    table = pd.DataFrame(hospitals)
    summary: dict = {"hospitals_analyzed": int(len(table))}
    if len(table) >= 5:
        observed_cov = table["row_coverage"].to_numpy()
        noise = float(np.mean(observed_cov * (1 - observed_cov) / table["rows"].to_numpy()))
        variance = float(np.var(observed_cov, ddof=1))
        summary["between_hospital_coverage_sd"] = float(np.sqrt(variance))
        summary["binomial_noise_sd"] = float(np.sqrt(noise))
        summary["excess_variance_fraction"] = float(max(0.0, 1 - noise / variance)) if variance > 0 else 0.0
        correlations = {}
        for column in (
            "median_input_age_hr",
            "median_abs_label_offset_from_horizon_hr",
            "share_label_outside_mimic_window",
            "median_current_value",
        ):
            values = table[column].to_numpy(dtype=float)
            if np.isfinite(values).sum() >= 5 and np.nanstd(values) > 0:
                rho, p_value = spearmanr(values, observed_cov, nan_policy="omit")
                correlations[column] = {"spearman_rho": float(rho), "p_value": float(p_value)}
        summary["coverage_correlations"] = correlations
        null_error = np.sum((observed_cov - float(np.mean(cal_covered))) ** 2)
        explained = {}
        for family in families:
            predicted = table[f"predicted_coverage__{family}"].to_numpy()
            rho, p_value = spearmanr(predicted, observed_cov)
            explained[family] = {
                "spearman_rho_predicted_vs_observed": float(rho) if np.isfinite(rho) else None,
                "p_value": float(p_value) if np.isfinite(p_value) else None,
                "error_reduction_vs_mimic_constant": float(
                    1 - np.sum((observed_cov - predicted) ** 2) / null_error
                ) if null_error > 0 else None,
                "predicted_range": [float(predicted.min()), float(predicted.max())],
            }
        summary["reweighting"] = explained
        summary["observed_range"] = [float(observed_cov.min()), float(observed_cov.max())]

    pooled = {}
    for family, (cal_bins, test_bins) in families.items():
        if "and" in family:
            continue
        per_bin = []
        for value in np.unique(np.concatenate([cal_bins, test_bins])):
            if value < 0:
                continue
            cal_mask = cal_bins == value
            test_mask = test_bins == value
            per_bin.append({
                "bin": int(value),
                "mimic_rows": int(cal_mask.sum()),
                "mimic_coverage": float(np.mean(cal_covered[cal_mask])) if cal_mask.sum() >= MIN_BIN_ROWS else None,
                "eicu_rows": int(test_mask.sum()),
                "eicu_coverage": float(np.mean(covered[test_mask])) if test_mask.sum() >= MIN_BIN_ROWS else None,
            })
        pooled[family] = per_bin

    return {
        "reproduction": reproduction,
        "label_offset_available_in_mimic": bool(label_offset_available),
        "mimic_label_reconstruction_match_rate": offset_match_rate,
        "mimic_calibration_coverage": float(np.mean(cal_covered)),
        "mimic_median_input_age_hr": float(np.nanmedian(cal_age)),
        "eicu_median_input_age_hr": float(np.nanmedian(test_age)),
        "mimic_label_offset_quantiles_10_50_90": (
            [float(x) for x in np.nanquantile(cal_offset, (0.1, 0.5, 0.9))]
            if label_offset_available else None
        ),
        "eicu_label_offset_quantiles_10_50_90": [
            float(x) for x in np.nanquantile(test_offset, (0.1, 0.5, 0.9))
        ],
        "summary": summary,
        "pooled_bins": pooled,
        "hospitals": hospitals,
    }


# ---------------------------------------------------------------------------
# Diagnostic 2: MIMIC-only measurement-process intervention
# ---------------------------------------------------------------------------


def _greedy_spacing_keep(stays: np.ndarray, times: np.ndarray, spacing: float) -> np.ndarray:
    keep = np.zeros(len(times), dtype=bool)
    last_stay = None
    last_time = -np.inf
    for index in range(len(times)):
        if stays[index] != last_stay:
            last_stay = stays[index]
            keep[index] = True
            last_time = times[index]
        elif times[index] - last_time >= spacing:
            keep[index] = True
            last_time = times[index]
    return keep


def thin_lab_input(
    frame: pd.DataFrame,
    rows: np.ndarray,
    lab_events: pd.DataFrame,
    target: str,
    spacing_hours: float,
) -> pd.DataFrame:
    """Rebuild the target lab as-of input for ``rows`` from raw lab times.

    Raw events are thinned per stay to a minimum spacing (``0`` keeps every
    event and reproduces the original extraction), then read back with the
    same backward as-of rule and look-back used to build the transitions.
    Thinning removes measurements; it never invents one.
    """

    output = frame.copy()
    queries = pd.DataFrame({
        "position": np.arange(len(rows)),
        "stay_id": raw_stay_ids(frame.iloc[rows]),
        "time": pd.to_datetime(frame.iloc[rows]["anchor_time"]).to_numpy().astype("datetime64[ns]"),
    })
    queries = queries[np.isfinite(queries["stay_id"])].astype({"stay_id": "int64"})
    lab = lab_events.loc[
        (lab_events["var"] == target) & lab_events["stay_id"].isin(set(queries["stay_id"])),
        ["stay_id", "charttime", "valuenum"],
    ].copy()
    lab["charttime"] = lab["charttime"].to_numpy().astype("datetime64[ns]")
    lab = lab.sort_values(["stay_id", "charttime"], kind="stable")
    if spacing_hours > 0:
        hours = lab["charttime"].to_numpy().astype(np.int64) / 3.6e12
        lab = lab[_greedy_spacing_keep(lab["stay_id"].to_numpy(), hours, spacing_hours)]
    merged = pd.merge_asof(
        queries.sort_values("time"),
        lab.sort_values("charttime"),
        left_on="time",
        right_on="charttime",
        by="stay_id",
        direction="backward",
        tolerance=pd.Timedelta(hours=LOOKBACK_HOURS),
    ).sort_values("position")
    values = np.full(len(rows), np.nan)
    ages = np.full(len(rows), np.nan)
    positions = merged["position"].to_numpy()
    values[positions] = merged["valuenum"].to_numpy(dtype=np.float64)
    ages[positions] = ((merged["time"] - merged["charttime"]).dt.total_seconds() / 3600.0).to_numpy()
    output.loc[output.index[rows], f"{target}_t"] = values
    output.loc[output.index[rows], f"{target}_age_hr"] = ages
    return output


def _fit_and_score(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray, **kwargs) -> dict:
    fitted = fit_continuous_recipe(frame, train, test, **kwargs)
    target = kwargs["target"]
    width = conformal_width(frame, fitted, target)
    rows = fitted["test_rows"]
    error = np.abs(fitted["test_prediction"] - _safe_float(frame, f"future_{target}")[rows])
    return {"rows": rows, "error": error, "width": width}


def mimic_intervention(
    mimic: pd.DataFrame,
    *,
    cell: str,
    target: str,
    source: str,
    lab_events: pd.DataFrame | None,
) -> dict:
    frame = mimic.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
    subjects = frame["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    rng = np.random.default_rng(stable_seed(cell, "process_intervention_split"))
    held_out = set(rng.choice(unique, size=int(round(0.3 * len(unique))), replace=False))
    is_test = np.array([subject in held_out for subject in subjects])
    train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    fit_kwargs = {"target": target, "source": source, "seed": stable_seed(cell, "process_intervention")}
    base = _fit_and_score(frame, train, test, **fit_kwargs)
    base_covered = pd.Series(base["error"] <= base["width"], index=base["rows"])
    base_error = pd.Series(base["error"], index=base["rows"])
    result: dict = {
        "train_subjects": int(len(unique) - len(held_out)),
        "test_subjects": int(len(held_out)),
        "half_width": base["width"],
        "baseline_test_coverage": patient_equal(subjects[base["rows"]], base_covered.to_numpy()),
        "input_thinning": [],
        "label_offset": {},
    }

    if lab_events is not None and target in LAB_TARGETS:
        original = _safe_float(frame, f"{target}_t")
        rebuilt = thin_lab_input(frame, test, lab_events, target, 0.0)
        result["raw_rebuild_match_rate"] = float(np.mean(np.isclose(
            _safe_float(rebuilt, f"{target}_t")[test], original[test], equal_nan=True
        )))
        for spacing in THINNING_SPACINGS_HOURS[1:]:
            thinned = thin_lab_input(frame, test, lab_events, target, spacing)
            scored = _fit_and_score(thinned, train, test, **fit_kwargs)
            if not np.isclose(scored["width"], base["width"]):
                raise RuntimeError("thinning test inputs changed the MIMIC-trained interval")
            rows = np.intersect1d(scored["rows"], base["rows"])
            changed = pd.Series(scored["error"] <= scored["width"], index=scored["rows"])[rows]
            changed_error = pd.Series(scored["error"], index=scored["rows"])[rows]
            result["input_thinning"].append({
                "spacing_hours": spacing,
                "rows_retained": int(len(rows)),
                "rows_retained_share": float(len(rows) / len(base["rows"])),
                "share_rows_with_changed_input": float(np.mean(~np.isclose(
                    _safe_float(thinned, f"{target}_t")[rows], original[rows]
                ))),
                "median_target_age_hr_before": float(np.nanmedian(_safe_float(frame, f"{target}_age_hr")[rows])),
                "median_target_age_hr_after": float(np.nanmedian(_safe_float(thinned, f"{target}_age_hr")[rows])),
                "mae_before": float(base_error[rows].mean()),
                "mae_after": float(changed_error.mean()),
                **paired_coverage_shift(
                    subjects[rows],
                    base_covered[rows].to_numpy(),
                    changed.to_numpy(),
                    seed=stable_seed(cell, "thin", spacing),
                ),
            })

    if lab_events is not None and target in LAB_TARGETS:
        observed = _safe_float(frame, f"future_{target}")
        for name, low, high in LABEL_OFFSET_BINS:
            values, offsets = lab_labels_in_window(frame, test, lab_events, target, low, high)
            relabeled = frame.copy()
            column = relabeled.columns.get_loc(f"future_{target}")
            relabeled.iloc[test, column] = values
            scored = _fit_and_score(relabeled, train, test, **fit_kwargs)
            if not np.isclose(scored["width"], base["width"]):
                raise RuntimeError("relabeling test rows changed the MIMIC-trained interval")
            rows = np.intersect1d(scored["rows"], base["rows"])
            if len(rows) < MIN_BIN_ROWS:
                result["label_offset"][name] = {"rows": int(len(rows)), "reason": "insufficient_rows"}
                continue
            changed = pd.Series(scored["error"] <= scored["width"], index=scored["rows"])[rows]
            position = {row: index for index, row in enumerate(test)}
            row_offsets = offsets[[position[row] for row in rows]]
            result["label_offset"][name] = {
                "rows": int(len(rows)),
                "median_label_offset_hr": float(np.median(row_offsets)),
                "share_label_identical_to_standard": float(np.mean(
                    np.isclose(relabeled.iloc[rows, column].to_numpy(dtype=float), observed[rows])
                )),
                **paired_coverage_shift(
                    subjects[rows],
                    base_covered[rows].to_numpy(),
                    changed.to_numpy(),
                    seed=stable_seed(cell, name),
                ),
            }
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Measurement-Process Shift Diagnostic (MIMIC-IV to eICU, 6h cells)",
        "",
        "Generated by `measurement_process_shift_diagnostic.py`. Aggregate-only; no row-level data.",
        "",
        "## Diagnostic 1: eICU hospital coverage vs measurement process",
        "",
        "Frozen MIMIC recipe and MIMIC-only conformal width, reproduced exactly. eICU labels score coverage only.",
        "`error_reduction` = share of between-hospital coverage error (vs a constant MIMIC coverage) removed by",
        "reweighting MIMIC calibration residuals to each hospital's covariate mix (eICU inputs only).",
        "",
        "| Cell | Reproduced | Hospitals | Coverage range | Excess var. | Input age ρ | Label offset ρ | Case-mix ρ | Reweight: age | Reweight: label offset | Reweight: age+offset | Reweight: case mix |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["cells"]:
        d1 = cell["diagnostic_1"]
        summary = d1["summary"]
        correlations = summary.get("coverage_correlations", {})
        reweighting = summary.get("reweighting", {})

        def rho(key):
            item = correlations.get(key)
            return f"{item['spearman_rho']:.2f} (p={item['p_value']:.2g})" if item else "-"

        def reduction(key):
            item = reweighting.get(key)
            return _fmt(item["error_reduction_vs_mimic_constant"], 2) if item else "-"

        observed_range = summary.get("observed_range")
        lines.append(
            f"| {cell['cell']} | {d1['reproduction'].get('matches_frozen_audit', '-')} | "
            f"{summary['hospitals_analyzed']} | "
            f"{'-' if not observed_range else f'{observed_range[0]:.2f}–{observed_range[1]:.2f}'} | "
            f"{_fmt(summary.get('excess_variance_fraction'), 2)} | "
            f"{rho('median_input_age_hr')} | {rho('median_abs_label_offset_from_horizon_hr')} | "
            f"{rho('median_current_value')} | {reduction('process_input_age')} | "
            f"{reduction('process_label_offset')} | {reduction('process_age_and_label_offset')} | "
            f"{reduction('case_mix_current_value')} |"
        )
    lines.extend([
        "",
        "Label timing (hours from anchor to the realized future label, 10/50/90th percentiles):",
        "",
        "| Cell | MIMIC (reconstructed) | eICU | MIMIC label reconstruction match |",
        "|---|---|---|---:|",
    ])
    for cell in report["cells"]:
        d1 = cell["diagnostic_1"]
        mimic_q = d1["mimic_label_offset_quantiles_10_50_90"]
        eicu_q = d1["eicu_label_offset_quantiles_10_50_90"]
        lines.append(
            f"| {cell['cell']} | {'-' if not mimic_q else ' / '.join(f'{x:.2f}' for x in mimic_q)} | "
            f"{' / '.join(f'{x:.2f}' for x in eicu_q)} | {_fmt(d1['mimic_label_reconstruction_match_rate'])} |"
        )
    lines.extend([
        "",
        "## Diagnostic 2: MIMIC-only process intervention",
        "",
        "Recipe and conformal width fit on 70% of MIMIC patients; the other 30% are scored with their",
        "measurement process changed and patients held fixed. Shift = patient-equal coverage change on",
        "the same rows, with a patient-cluster bootstrap 95% CI. Lab targets only: inputs are rebuilt",
        "from raw MIMIC lab times, so vital-sign and urine cells have no intervention rows.",
        "",
        "| Cell | Baseline cov. | Intervention | Rows kept | Median age before→after | Coverage shift [95% CI] |",
        "|---|---:|---|---:|---|---|",
    ])
    for cell in report["cells"]:
        d2 = cell["diagnostic_2"]
        for item in d2["input_thinning"]:
            lines.append(
                f"| {cell['cell']} | {d2['baseline_test_coverage']:.3f} | raw lab ordering ≥{item['spacing_hours']:g}h apart "
                f"({item['share_rows_with_changed_input']:.0%} inputs changed) | "
                f"{item['rows_retained_share']:.0%} | {item['median_target_age_hr_before']:.2f}→{item['median_target_age_hr_after']:.2f} | "
                f"{item['coverage_shift']:+.3f} [{item['shift_ci95'][0]:+.3f}, {item['shift_ci95'][1]:+.3f}] |"
            )
        for name, item in d2["label_offset"].items():
            if "coverage_shift" not in item:
                continue
            lines.append(
                f"| {cell['cell']} | {d2['baseline_test_coverage']:.3f} | label {name} (median {item['median_label_offset_hr']:.1f}h) | "
                f"{item['rows']} rows | - | "
                f"{item['coverage_shift']:+.3f} [{item['shift_ci95'][0]:+.3f}, {item['shift_ci95'][1]:+.3f}] |"
            )
    lines.extend([
        "",
        "## Boundary",
        "",
        "Diagnostic only. Nothing here is fit, selected, or calibrated on eICU labels, and no cell is promoted.",
        "`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    cells = development_cells(args.development_report)
    if args.cells:
        selected = set(filter(None, map(str.strip, args.cells.split(","))))
        cells = [row for row in cells if row["cell"] in selected]
    frozen = json.loads(args.frozen_audit.read_text(encoding="utf-8"))
    stored = {
        row["cell"]: row.get("metrics", {}).get("interval")
        for row in frozen["cells"]
    }
    lab_events = pd.read_parquet(args.lab_cache) if args.lab_cache.exists() else None
    raw = pd.read_parquet(mimic_path(args.project_root, HORIZON))
    reports = []
    for contract in cells:
        cell, target, source = contract["cell"], contract["target"], contract["source_model"]
        print(f"diagnosing {cell} ({source})", flush=True)
        mimic, _ = normalize_cohort(
            raw, database="MIMIC-IV", target=target, horizon=HORIZON,
            max_rows=args.max_rows_per_database,
        )
        eicu, _ = choose_eicu_cohort(
            args.eicu_root, target=target, horizon=HORIZON,
            max_rows=args.max_rows_per_database,
        )
        intervention_mimic, _ = normalize_cohort(
            raw, database="MIMIC-IV", target=target, horizon=HORIZON,
            max_rows=args.intervention_rows,
        )
        reports.append({
            "cell": cell,
            "target": target,
            "source_model": source,
            "diagnostic_1": hospital_association(
                mimic, eicu, cell=cell, target=target, source=source,
                lab_events=lab_events, stored_interval=stored.get(cell),
            ),
            "diagnostic_2": mimic_intervention(
                intervention_mimic, cell=cell, target=target, source=source,
                lab_events=lab_events,
            ),
        })
    output = {
        "schema": "measurement_process_shift_diagnostic.v1",
        "horizon_hours": HORIZON,
        "lab_label_timing_available": lab_events is not None,
        "eicu_labels_used_for_fitting_or_selection": False,
        "cells": reports,
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown(output, args.findings)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-mimic-lab-times")
    extract.add_argument("--mimic-dir", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--chunksize", type=int, default=5_000_000)
    diagnose = commands.add_parser("run")
    diagnose.add_argument(
        "--development-report",
        type=Path,
        default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"),
    )
    diagnose.add_argument(
        "--frozen-audit",
        type=Path,
        default=Path("mimic_to_eicu_49_cell_external_audit_20260816.json"),
    )
    diagnose.add_argument("--project-root", type=Path, default=Path("."))
    diagnose.add_argument("--eicu-root", type=Path, default=Path("eicu_future_label_repaired"))
    diagnose.add_argument(
        "--lab-cache",
        type=Path,
        default=Path("mimiciv_lab_event_times_process_diagnostic.parquet"),
    )
    diagnose.add_argument("--max-rows-per-database", type=int, default=15000)
    diagnose.add_argument("--intervention-rows", type=int, default=30000)
    diagnose.add_argument("--cells", default="")
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--findings", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract-mimic-lab-times":
        extract_mimic_lab_times(args.mimic_dir, args.output, args.chunksize)
        return
    run(args)


if __name__ == "__main__":
    main()
