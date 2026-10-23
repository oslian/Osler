"""Build Emory University Hospital ICU transitions in the MIMIC-IV contract.

Source: PhysioNet/Computing in Cardiology Challenge 2019 (open access),
training set B.  The Challenge paper (Reyna et al., Crit Care Med 2020;
``physionet_challenge_2019_ccm_manuscript.pdf``) identifies hospital system A
as Beth Israel Deaconess (overlaps MIMIC, not used) and hospital system B as
Emory University Hospital.  One pipe-delimited file per ICU patient; one row
per hour of ICU stay (``ICULOS`` = hours since ICU admission), multiple
measurements within an hour summarized by their median, and LOINC codes for
the same parameter merged (e.g. serum and arterial hemoglobin).

An hourly row is timed at the end of its hour (ICU admission + ``ICULOS``
hours), so a value never precedes its measurement.  Anchors, as-of state,
labels and plausibility cleaning reuse ``mimiciv_observation_transition_extract``.

Mapping to MIMIC-IV variables and units (non-target input magnitudes were
inspected before the protocol freeze; no target value was read):

- FiO2 is a fraction -> percent (x100);
- ``Calcium`` mixes ionized calcium (mmol/L, values < 3) and total calcium
  (mg/dL, values >= 3) -> ``ionized_calcium`` / ``calcium``;
- lactate is mmol/L and magnesium mg/dL despite the Challenge table's labels;
- ``Unit1`` / ``Unit2`` -> care unit MICU / SICU (else ``unknown``);
- bicarbonate: the Challenge ``HCO3`` column is sparse; where an hour has pH
  and PaCO2 but no ``HCO3``, the calculated bicarbonate
  (``osler_jepa.derived_measurements``, as MIMIC-IV itemid 50803) is added;
- not mapped (no MIMIC contract variable): SBP, DBP, EtCO2, BaseExcess, SaO2;
- absent at this site: sodium, urine output, point-of-care and blood-gas
  glucose, treatments (treatment-defined recipes fall back by
  ``osler_jepa.recipe_availability``).

Glucose is the Challenge's "serum glucose"; hemoglobin and hematocrit merge
serum and arterial measurements (as MIMIC's hemoglobin and hematocrit
itemids do).  Factual observation data only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mimiciv_observation_transition_extract as contract
from osler_jepa.derived_measurements import calculated_bicarbonate


BASE_TIME = pd.Timestamp("2100-01-01")
IONIZED_CALCIUM_MAX = 3.0

# Challenge column -> (MIMIC variable, multiplicative factor to MIMIC-IV units).
COLUMNS: dict[str, tuple[str, float]] = {
    "HR": ("heart_rate", 1.0),
    "O2Sat": ("o2sat", 1.0),
    "Temp": ("temperature", 1.0),
    "MAP": ("map", 1.0),
    "Resp": ("respiratory_rate", 1.0),
    "HCO3": ("bicarbonate", 1.0),
    "FiO2": ("fio2", 100.0),
    "pH": ("ph", 1.0),
    "PaCO2": ("paco2", 1.0),
    "AST": ("ast", 1.0),
    "BUN": ("bun", 1.0),
    "Alkalinephos": ("alkaline_phos", 1.0),
    "Chloride": ("chloride", 1.0),
    "Creatinine": ("creatinine", 1.0),
    "Bilirubin_direct": ("bilirubin_direct", 1.0),
    "Glucose": ("glucose", 1.0),
    "Lactate": ("lactate", 1.0),
    "Magnesium": ("magnesium", 1.0),
    "Phosphate": ("phosphate", 1.0),
    "Potassium": ("potassium", 1.0),
    "Bilirubin_total": ("bilirubin", 1.0),
    "TroponinI": ("troponin_i", 1.0),
    "Hct": ("hematocrit", 1.0),
    "Hgb": ("hemoglobin", 1.0),
    "PTT": ("ptt", 1.0),
    "WBC": ("wbc", 1.0),
    "Fibrinogen": ("fibrinogen", 1.0),
    "Platelets": ("platelets", 1.0),
}


def read_patient(path: Path) -> tuple[dict, pd.DataFrame]:
    """One file -> (stay record, long measurements)."""

    table = pd.read_csv(path, sep="|")
    stay_id = path.stem
    hours = pd.to_numeric(table["ICULOS"], errors="coerce")
    micu = pd.to_numeric(table["Unit1"], errors="coerce").eq(1).any()
    sicu = pd.to_numeric(table["Unit2"], errors="coerce").eq(1).any()
    careunit = "MICU" if micu and not sicu else "SICU" if sicu and not micu else "unknown"
    stay = {
        "subject_id": stay_id,
        "stay_id": stay_id,
        "first_careunit": careunit,
        "last_careunit": careunit,
        "intime": BASE_TIME,
        "outtime": BASE_TIME + pd.Timedelta(hours=float(hours.max())),
    }
    pieces = []
    for column, (var, factor) in COLUMNS.items():
        values = pd.to_numeric(table[column], errors="coerce") * factor
        keep = values.notna() & hours.notna()
        pieces.append(pd.DataFrame({"hours": hours[keep], "var": var, "valuenum": values[keep]}))
    ph, paco2 = pd.to_numeric(table["pH"], errors="coerce"), pd.to_numeric(table["PaCO2"], errors="coerce")
    derive = pd.to_numeric(table["HCO3"], errors="coerce").isna() & ph.notna() & paco2.notna() & hours.notna()
    pieces.append(pd.DataFrame({"hours": hours[derive], "var": "bicarbonate",
                                "valuenum": calculated_bicarbonate(ph[derive], paco2[derive])}))
    calcium = pd.to_numeric(table["Calcium"], errors="coerce")
    keep = calcium.notna() & hours.notna()
    pieces.append(pd.DataFrame({
        "hours": hours[keep],
        "var": np.where(calcium[keep] < IONIZED_CALCIUM_MAX, "ionized_calcium", "calcium"),
        "valuenum": calcium[keep],
    }))
    events = pd.concat(pieces, ignore_index=True)
    events = events.assign(
        subject_id=stay_id,
        stay_id=stay_id,
        charttime=BASE_TIME + pd.to_timedelta(events["hours"], unit="h"),
    )
    return stay, events[["subject_id", "stay_id", "charttime", "var", "valuenum"]]


def build(root: Path, horizon_hours: float, anchor_step_hours: float) -> tuple[pd.DataFrame, dict]:
    contract.DELTA_H = float(horizon_hours)
    contract.ANCHOR_STEP_H = float(anchor_step_hours)
    contract.TARGET_TOL_H = float(min(2.0, horizon_hours / 2.0))
    suffix = f"_tp{int(horizon_hours)}" if float(horizon_hours).is_integer() else f"_tp{horizon_hours:g}"
    records, events = zip(*(read_patient(path) for path in sorted(root.glob("*.psv"))))
    stays = pd.DataFrame(records)
    stays = stays[stays["outtime"] > stays["intime"] + pd.Timedelta(hours=horizon_hours)].reset_index(drop=True)
    measurements = contract.clean_measurements(pd.concat(events, ignore_index=True))
    measurements = measurements[measurements["stay_id"].isin(set(stays["stay_id"]))]
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
    frame["hospitalid"] = "Emory"
    report = {
        "artifact": "Emory University Hospital ICU transitions (PhysioNet Challenge 2019 training set B, MIMIC-IV contract)",
        "delta_h": float(horizon_hours),
        "anchor_step_h": float(anchor_step_hours),
        "lookback_h": contract.LOOKBACK_H,
        "target_tolerance_h": contract.TARGET_TOL_H,
        "episode_max_h": contract.EPISODE_MAX_H,
        "icu_stays": int(len(stays)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "rows": int(len(frame)),
        "careunits": frame["first_careunit"].value_counts().to_dict(),
        "pair_counts": contract.pair_counts(frame, suffix),
        "treatment_context_available": False,
        "patient_ids_included_in_report": False,
        "causal_claim_allowed": False,
    }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="training_setB directory")
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
