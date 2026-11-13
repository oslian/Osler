"""Build Zigong Fourth People's Hospital ICU transitions in the MIMIC-IV contract.

Source: "Critical care database comprising patients with infection at Zigong
Fourth People's Hospital" v1.1 (PhysioNet, credentialed).  Times are hours
from hospital admission; ICU stays are transfer segments whose department is
an ICU.  Anchors, as-of state, labels and plausibility cleaning reuse
``mimiciv_observation_transition_extract`` so the code path matches MIMIC-IV.

Only the laboratory table is used.  The nursing chart (vital signs, bedside
glucose, urine volume) is excluded: its ChartTime has day-level resolution
(median spacing between distinct times 24 h, 23% missing, 6% before
admission), so a charted value cannot be guaranteed to precede a forecast
anchor and hourly labels cannot be formed.  Vital-sign and urine targets are
therefore not evaluable on Zigong, and vital-sign inputs are absent (imputed
with MIMIC training medians by the frozen recipes).

Harmonized laboratory definitions (see MEASUREMENT_DEFINITION_INVENTORY_20261015.md):

- glucose = chemistry laboratory glucose; blood-gas glucose -> ``glucose_bg``;
- laboratory units converted to MIMIC-IV units (factors in ``LAB_ITEMS``);
- not mapped: BNP (MIMIC uses NT-proBNP) and CK-MB (activity vs mass);
- treatment context is not extracted (prescriptions only, no administration
  records), so treatment-defined recipes fall back by
  ``osler_jepa.recipe_availability``.

Factual observation data only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import mimiciv_observation_transition_extract as contract


BASE_TIME = pd.Timestamp("2100-01-01")
GLUCOSE_MMOL_TO_MGDL = 18.016
ICU_DEPARTMENT = re.compile(r"ICU|intensive care", re.IGNORECASE)
CHEMISTRY_GLUCOSE_CATEGORIES = {"Biochemical 1", "Biochemistry 2", "blood sugar"}
BLOOD_GAS_CATEGORIES = {"Blood gas analysis combination", "EICU blood gas analysis",
                        "Blood gas analysis in emergency department"}
CBC_CATEGORIES = {"Blood analysis (five categories)", "Blood analysis + CRP"}

# Item -> (MIMIC variable, multiplicative factor to MIMIC-IV units); category-specific items are resolved in map_lab_item.
LAB_ITEMS: dict[str, tuple[str, float]] = {
    "Creatinine (enzymatic method) (CRE)": ("creatinine", 1 / 88.42),
    "Urea (urea)": ("bun", 2.801),
    "Platelet (PLT)": ("platelets", 1.0),
    "White blood cell (WBC)": ("wbc", 1.0),
    "Potassium (k)": ("potassium", 1.0),
    "Potassium ion (K +)": ("potassium", 1.0),
    "Sodium (NA)": ("sodium", 1.0),
    "Sodium ion (Na +)": ("sodium", 1.0),
    "Chlorine (CL)": ("chloride", 1.0),
    "Chloride ion (Cl -)": ("chloride", 1.0),
    "Carbon dioxide (CO2)": ("bicarbonate", 1.0),
    "Measured bicarbonate (hco3act)": ("bicarbonate", 1.0),
    "HCO3 - measured bicarbonate (HCO3 -)": ("bicarbonate", 1.0),
    "Anion gap": ("anion_gap", 1.0),
    "Anion gap (angap)": ("anion_gap", 1.0),
    "Calcium (CA)": ("calcium", 4.008),
    "Free calcium (Ca + +)": ("ionized_calcium", 1.0),
    "Calcium ion concentration (caca2 +)": ("ionized_calcium", 1.0),
    "Serum magnesium (mg)": ("magnesium", 2.431),
    "Inorganic phosphorus (P)": ("phosphate", 3.097),
    "Lactic acid (LAC)": ("lactate", 1.0),
    "PH (PH)": ("ph", 1.0),
    "Carbon dioxide partial pressure (PCO2)": ("paco2", 1.0),
    "Arterial partial pressure of carbon dioxide (PCO2)": ("paco2", 1.0),
    "Oxygen partial pressure (PO2)": ("pao2", 1.0),
    "Arterial oxygen partial pressure (PO2)": ("pao2", 1.0),
    "Albumin (ALB)": ("albumin", 0.1),
    "Total protein (TP)": ("total_protein", 0.1),
    "Total bilirubin (TBIL)": ("bilirubin", 1 / 17.1),
    "Direct bilirubin (DBIL)": ("bilirubin_direct", 1 / 17.1),
    "Aspartate aminotransferase (AST)": ("ast", 1.0),
    "Alanine aminotransferase (ALT)": ("alt", 1.0),
    "Alkaline phosphatase (ALP)": ("alkaline_phos", 1.0),
    "Lactate dehydrogenase (LDH)": ("ldh", 1.0),
    "Creatine kinase (CK)": ("cpk", 1.0),
    "International normalized ratio (INR)": ("inr", 1.0),
    "Activated partial thromboplastin time (APTT)": ("ptt", 1.0),
    "Fibrinogen (FIB)": ("fibrinogen", 100.0),
    "C-reactive protein (CRP)": ("crp", 1.0),
    "Uric acid (UA)": ("uric_acid", 1 / 59.48),
    "Triglyceride (TG)": ("triglycerides", 88.57),
    "High sensitivity troponin I (tn-i)": ("troponin_i", 1.0),
    "Myoglobin (myo)": ("myoglobin", 1.0),
}
def map_lab_item(category: str, item: str, unit: object) -> tuple[str, float] | None:
    if item in {"Glucose (Glu)", "Glucose concentration (cglu)"}:
        if category in CHEMISTRY_GLUCOSE_CATEGORIES:
            return "glucose", GLUCOSE_MMOL_TO_MGDL
        if category in BLOOD_GAS_CATEGORIES:
            return "glucose_bg", GLUCOSE_MMOL_TO_MGDL
        return None
    if item == "Hemoglobin (Hgb)" and category in CBC_CATEGORIES:
        return "hemoglobin", 0.1
    if item == "Total hemoglobin (THB)" and category in BLOOD_GAS_CATEGORIES:
        return "hemoglobin", 0.1
    if item == "Hematocrit (HCT)" and category in CBC_CATEGORIES:
        return "hematocrit", 100.0          # reported as a fraction (reference 0.40--0.50)
    if item in {"Hematocrit (HCT)", "Hct(Hct)"} and category in BLOOD_GAS_CATEGORIES:
        return "hematocrit", 1.0            # reported in percent
    return LAB_ITEMS.get(item)


def icu_stays(root: Path, horizon_hours: float) -> pd.DataFrame:
    transfers = pd.read_csv(root / "dtTransfer.csv", usecols=["PATIENT_ID", "INP_NO", "StartTime", "StopTime", "TransferDept"])
    transfers = transfers[transfers["TransferDept"].fillna("").str.contains(ICU_DEPARTMENT)]
    transfers = transfers.dropna(subset=["StartTime", "StopTime"])
    transfers = transfers[transfers["StopTime"] > transfers["StartTime"] + horizon_hours].copy()
    transfers = transfers.sort_values(["INP_NO", "StartTime"]).reset_index(drop=True)
    transfers["stay_id"] = transfers["INP_NO"].astype(str) + "_" + transfers.groupby("INP_NO").cumcount().astype(str)
    return pd.DataFrame({
        "subject_id": transfers["PATIENT_ID"].astype(str),
        "INP_NO": transfers["INP_NO"].astype(str),
        "stay_id": transfers["stay_id"],
        "first_careunit": transfers["TransferDept"].astype(str),
        "last_careunit": transfers["TransferDept"].astype(str),
        "start_hr": transfers["StartTime"].astype(float),
        "stop_hr": transfers["StopTime"].astype(float),
        "intime": BASE_TIME + pd.to_timedelta(transfers["StartTime"].astype(float), unit="h"),
        "outtime": BASE_TIME + pd.to_timedelta(transfers["StopTime"].astype(float), unit="h"),
    })


def attach_stays(events: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """Keep events inside an ICU segment and label them with that segment's stay."""

    joined = events.merge(stays[["INP_NO", "subject_id", "stay_id", "start_hr", "stop_hr"]], on="INP_NO", how="inner")
    joined = joined[(joined["hours"] >= joined["start_hr"]) & (joined["hours"] <= joined["stop_hr"])]
    joined = joined.assign(charttime=BASE_TIME + pd.to_timedelta(joined["hours"], unit="h"))
    return joined


def load_labs(root: Path, stays: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(root / "dtLab.csv", usecols=["INP_NO", "LabTime", "Category", "Item", "LabValue", "Unit"],
                             dtype={"INP_NO": str, "Category": str, "Item": str, "Unit": str}, chunksize=500_000):
        keys = list(zip(chunk["Category"].fillna(""), chunk["Item"].fillna(""), chunk["Unit"]))
        mapped = [map_lab_item(*key) for key in keys]
        chunk = chunk.assign(
            var=[m[0] if m else None for m in mapped],
            factor=[m[1] if m else np.nan for m in mapped],
        ).dropna(subset=["var"])
        chunk = chunk.assign(
            valuenum=pd.to_numeric(chunk["LabValue"], errors="coerce") * chunk["factor"],
            hours=pd.to_numeric(chunk["LabTime"], errors="coerce"),
        ).dropna(subset=["valuenum", "hours"])
        pieces.append(attach_stays(chunk[["INP_NO", "hours", "var", "valuenum"]], stays))
    return pd.concat(pieces, ignore_index=True)[["subject_id", "stay_id", "charttime", "var", "valuenum"]]


def build(root: Path, horizon_hours: float, anchor_step_hours: float) -> tuple[pd.DataFrame, dict]:
    contract.DELTA_H = float(horizon_hours)
    contract.ANCHOR_STEP_H = float(anchor_step_hours)
    contract.TARGET_TOL_H = float(min(2.0, horizon_hours / 2.0))
    suffix = f"_tp{int(horizon_hours)}" if float(horizon_hours).is_integer() else f"_tp{horizon_hours:g}"
    stays = icu_stays(root, horizon_hours)
    measurements = contract.clean_measurements(load_labs(root, stays))
    anchors = contract.build_anchors(stays)
    state_t = contract.assemble_state(anchors, measurements, "_t", "t", "backward", contract.LOOKBACK_H)
    state_f = contract.assemble_state(anchors, measurements, suffix, "t_plus", "nearest", contract.TARGET_TOL_H)
    frame = pd.concat(
        [anchors[["subject_id", "stay_id", "first_careunit", "last_careunit", "t", "t_plus"]],
         state_t.drop(columns=["stay_id", "t"]), state_f.drop(columns=["stay_id", "t_plus"])],
        axis=1,
    )
    # Every anchor is kept, labelled or not (see mimiciv_observation_transition_extract).
    frame = frame.reset_index(drop=True)
    frame["hospitalid"] = "Zigong"
    report = {
        "artifact": "Zigong Fourth People's Hospital infection ICU transitions (MIMIC-IV contract)",
        "delta_h": float(horizon_hours),
        "anchor_step_h": float(anchor_step_hours),
        "lookback_h": contract.LOOKBACK_H,
        "target_tolerance_h": contract.TARGET_TOL_H,
        "episode_max_h": contract.EPISODE_MAX_H,
        "icu_segments": int(len(stays)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "rows": int(len(frame)),
        "careunits": int(frame["first_careunit"].nunique()),
        "pair_counts": contract.pair_counts(frame, suffix),
        "treatment_context_available": False,
        "nursing_chart_used": False,
        "patient_ids_included_in_report": False,
        "causal_claim_allowed": False,
    }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="DataTables directory")
    parser.add_argument("--horizon-hours", type=float, required=True)
    parser.add_argument("--anchor-step-hours", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frame, report = build(args.root, args.horizon_hours, args.anchor_step_hours)
    frame.to_parquet(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "subjects", "stays")}))


if __name__ == "__main__":
    main()
