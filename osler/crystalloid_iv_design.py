"""Design-stage diagnostics for the balanced-versus-saline preference IV (no outcomes read).

Reads only ``eicu_crystalloid_iv_design.parquet`` (eligibility, treatment,
instrument, baseline covariates).  Reports instrument strength (first-stage F
with the frozen covariate set), treatment uptake by instrument quintile, and
baseline balance between stays at high- and low-preference hospitals (above vs
below the median instrument).  These decide, before outcomes are extracted,
whether the design is worth freezing.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osler_jepa.instrumental_variable import first_stage_f, standardized_mean_difference


PATIENT_COVARIATES = ("age", "admissionweight", "baseline_chloride", "baseline_sodium", "baseline_bicarbonate",
                      "baseline_creatinine", "baseline_bun", "baseline_glucose", "baseline_potassium",
                      "baseline_platelets", "baseline_hemoglobin", "typed_volume_ml")
CATEGORICAL = ("gender", "unittype", "unitadmitsource", "numbedscategory", "teachingstatus", "region", "admission_group")
TOP_DIAGNOSES = 20


def covariate_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Frozen adjustment set: numeric baselines (median-imputed + missing indicator) and one-hot categories."""

    frame = frame.copy()
    top = frame["apacheadmissiondx"].value_counts().index[:TOP_DIAGNOSES]
    frame["admission_group"] = frame["apacheadmissiondx"].where(frame["apacheadmissiondx"].isin(top), "other")
    columns, names = [], []
    for name in PATIENT_COVARIATES:
        if name == "typed_volume_ml":
            continue                               # volume is part of the treatment strategy, not a baseline covariate
        values = pd.to_numeric(frame[name], errors="coerce")
        columns.append(values.fillna(values.median()).to_numpy())
        names.append(name)
        if values.isna().any():
            columns.append(values.isna().astype(float).to_numpy())
            names.append(f"{name}_missing")
    for name in CATEGORICAL:
        dummies = pd.get_dummies(frame[name].fillna("unknown").astype(str), prefix=name, drop_first=True, dtype=float)
        columns.extend(dummies.to_numpy().T)
        names.extend(dummies.columns)
    return np.column_stack(columns), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=Path("eicu_crystalloid_iv_design.parquet"))
    parser.add_argument("--output", type=Path, default=Path("crystalloid_iv_design_report.json"))
    args = parser.parse_args()
    frame = pd.read_parquet(args.design)
    assert not any(column.startswith("day1_") for column in frame.columns), "design table must not contain outcomes"
    x, names = covariate_matrix(frame)
    d = frame["treated_balanced"].to_numpy(dtype=float)
    z = frame["instrument_hospital_preference"].to_numpy(dtype=float)
    high = z > np.median(z)
    quintile = pd.qcut(z, 5, labels=False, duplicates="drop")
    report = {
        "stays": int(len(frame)),
        "hospitals": int(frame["hospitalid"].nunique()),
        "treated_balanced_share": round(float(d.mean()), 4),
        "hospital_balanced_share_quantiles": np.round(frame.groupby("hospitalid")["treated_balanced"].mean().quantile([0, .1, .25, .5, .75, .9, 1]).to_numpy(), 3).tolist(),
        "first_stage_f_unadjusted": round(first_stage_f(d, z), 1),
        "first_stage_f_adjusted": round(first_stage_f(d, z, x), 1),
        "uptake_by_instrument_quintile": pd.Series(d).groupby(quintile).mean().round(3).tolist(),
        "balance_high_vs_low_preference_smd": {
            name: round(standardized_mean_difference(pd.to_numeric(frame[name], errors="coerce"), high), 3)
            for name in PATIENT_COVARIATES
        },
        "balance_treated_vs_untreated_smd": {
            name: round(standardized_mean_difference(pd.to_numeric(frame[name], errors="coerce"), d.astype(bool)), 3)
            for name in PATIENT_COVARIATES
        },
        "adjustment_columns": len(names),
        "outcomes_read": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
