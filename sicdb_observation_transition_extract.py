"""Build SICdb 1.0.8 observation transitions in the MIMIC-IV v5 contract (lockbox, protocol v10).

Source: Salzburg Intensive Care database 1.0.8 (PhysioNet, credentialed), files
``cases``, ``laboratory``, ``data_float_h`` and the dictionary ``d_references``.
Anchors, as-of state, labels (with their offsets) and plausibility cleaning reuse
``mimiciv_observation_transition_extract`` so the code path matches MIMIC-IV.

Variable mapping is declared from the data dictionary only (ReferenceValue,
ReferenceUnit and LOINC in ``d_references``), following the harmonized
definitions of protocol v10:

- ICU stay = one case: in-time = ``ICUOffset`` (first transfer to an intensive or
  intermediate care ward; a case may start with surgery), out-time =
  ``TimeOfStay``; subject = ``PatientID``; a patient's cases are placed on one
  timeline by ``OffsetAfterFirstAdmission`` (required for every case); this
  timeline has no origin shared across patients, and the deployment replay
  orders stays by a simulated clock (``osler_jepa.deployment_clock``: a random
  arrival order, so one patient's later case may arrive first) with histories
  built within a stay;
- care unit (the interval gate's and the unit-level adaptation's axis) = the
  ward at ICU admission, fixed for the whole stay (later transfers are not
  followed): the ``unitlog`` entry of one of the four wards at exactly
  ``ICUOffset``; when several are logged in that second, the last one (entry ids
  increase with time, so this is the state after that second, as-of semantics);
  cases with no such entry are excluded and counted.  ``cases.HospitalUnit`` is
  the LAST unit and is kept only as retrospective ``last_careunit``;
- laboratory values from the central laboratory (ZL) and the blood-gas analyzers
  (BGA, BG), as MIMIC-IV pools chemistry and blood-gas items for electrolytes,
  haemoglobin, haematocrit and bicarbonate; venous, capillary and
  temperature-corrected blood-gas items, standardized bicarbonate and items with
  an undocumented method suffix (CKR, HPM) are not mapped;
- glucose = central-laboratory glucose (348); blood-gas glucose -> ``glucose_bg``;
  SICdb has no point-of-care (capillary) glucose item, so ``glucose_poc`` cells
  are not evaluable;
- BUN = urea x 28.0134/60.056 (urea and BUN both in mg/dL); phosphate mmol/L x
  3.097; calcium mmol/L x 4.008; magnesium mmol/L x 2.431; CRP mg/dL x 10;
  hs-troponin T ng/L x 0.001; other mapped units already match MIMIC-IV;
- vital signs and urine from ``data_float_h``, which holds hourly aggregates
  (the mean of a monitored signal, the sum of a volume) at the start offset of
  the hour: each value is timed at the END of its hour, when it is complete, so
  no state uses minutes after its anchor, and only hours lying wholly inside
  the stay are used; urine output = hourly sum (mL/h; MIMIC instead divides a
  charted volume by its preceding collection interval);
- measured values are validated (``clean_measurements``) first; simultaneous
  values of one variable from several items (e.g. central-laboratory and
  blood-gas potassium at the same time) are averaged, so nothing depends on
  row order (the averaged values need not come from one specimen); then
  calculated bicarbonate from same-time validated (averaged) pH and PaCO2 where
  no validated bicarbonate exists (``osler_jepa.derived_measurements``; same-time
  pH and PaCO2 are not proven to come from one sample), validated again;
- MAP = the measured arterial or non-invasive mean only (protocol v10: "MAP =
  measured mean pressure"); no (SBP + 2 DBP)/3 surrogate;
- FiO2 and PEEP are not mapped: their unit is not declared in the dictionary;
- treatment context is not extracted (``osler_jepa.recipe_availability`` falls
  treatment-defined recipes back to measurement-pure ones).

Disclosed differences: laboratory values are usable at their sample time
(SICdb has no result time; MIMIC is treated the same way); an hourly label
describes the hour ending at its time, not a point measurement, so a nearest
label up to ``min(2, h/2)`` h before anchor + h can cover minutes before the
anchor at h = 1 (its availability is anchor + h + |offset|, a conservative
deadline, not its measurement time); blood-gas pH,
PaCO2 and PaO2 are the arterial items; the cohort includes intermediate-care
wards (INIC, INID).

Before the protocol-v10 addendum (``lockbox_addendum.py``) only the support
counts in the report may be inspected, never target values.  Factual
observation data only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import mimiciv_observation_transition_extract as contract
from osler_jepa.derived_measurements import calculated_bicarbonate


BASE_TIME = pd.Timestamp("2100-01-01")
HOURLY_AGGREGATE_S = 3600

# ReferenceGlobalID -> (MIMIC variable, multiplicative factor to MIMIC-IV units)
LAB_ITEMS: dict[int, tuple[str, float]] = {
    348: ("glucose", 1.0),                                       # Glukose (ZL), mg/dl, Glucose SerPl
    331: ("glucose_bg", 1.0), 656: ("glucose_bg", 1.0),          # Glucose (BG) (ZL), Glukose (BGA), mg/dl
    538: ("ph", 1.0), 688: ("ph", 1.0),                          # pH (ZL), PH (BGA), arterial
    443: ("paco2", 1.0), 687: ("paco2", 1.0),                    # PCO2 (ZL), PCO2 (BGA), mmHg
    444: ("pao2", 1.0), 689: ("pao2", 1.0),                      # PO2 (ZL), PO2 (BGA), mmHg
    451: ("bicarbonate", 1.0), 456: ("bicarbonate", 1.0), 666: ("bicarbonate", 1.0),   # actual HCO3, mmol/L
    672: ("anion_gap", 1.0),                                     # Anionen Gap (BGA), mmol/L
    450: ("chloride", 1.0), 683: ("chloride", 1.0),              # mmol/L
    463: ("potassium", 1.0), 453: ("potassium", 1.0), 685: ("potassium", 1.0),
    469: ("sodium", 1.0), 455: ("sodium", 1.0), 686: ("sodium", 1.0),
    457: ("calcium", 4.008),                                     # Calcium (ZL), mmol/L -> mg/dL
    452: ("ionized_calcium", 1.0), 655: ("ionized_calcium", 1.0),  # mmol/L
    468: ("magnesium", 2.431),                                   # mmol/L -> mg/dL
    471: ("phosphate", 3.097),                                   # mmol/L -> mg/dL
    465: ("lactate", 1.0), 454: ("lactate", 1.0), 657: ("lactate", 1.0),
    367: ("creatinine", 1.0), 368: ("creatinine", 1.0),          # mg/dl
    355: ("bun", 28.0134 / 60.056),                              # Harnstoff (urea) mg/dl -> urea nitrogen mg/dL
    301: ("wbc", 1.0),                                           # dictionary unit "g/L": giga-count per litre (10^9/L = K/uL)
    289: ("hemoglobin", 1.0), 288: ("hemoglobin", 1.0), 658: ("hemoglobin", 1.0),   # g/dl
    217: ("hematocrit", 1.0), 183: ("hematocrit", 1.0), 682: ("hematocrit", 1.0),   # %
    314: ("platelets", 1.0), 315: ("platelets", 1.0),            # "g/L" = 10^9/L = K/uL (EDTA and citrate tubes)
    333: ("bilirubin", 1.0), 332: ("bilirubin_direct", 1.0),     # mg/dl
    3128: ("inr", 1.0), 597: ("ptt", 1.0),
    344: ("fibrinogen", 1.0),                                    # mg/dl
    287: ("albumin", 1.0),                                       # Serum-Albumin g/dl
    616: ("ast", 1.0), 617: ("alt", 1.0), 609: ("alkaline_phos", 1.0), 623: ("lipase", 1.0), 608: ("amylase", 1.0),
    622: ("ldh", 1.0), 270: ("troponin_i", 1.0), 481: ("troponin_t", 0.001), 253: ("ck_mb", 1.0),
    476: ("serum_osmolality", 1.0), 479: ("tsh", 1.0), 259: ("ferritin", 1.0), 354: ("uric_acid", 1.0),
    351: ("crp", 10.0),                                          # mg/dl -> mg/L
}
# data_float_h DataID -> MIMIC variable
SIGNAL_ITEMS: dict[int, str] = {
    707: "heart_rate",                                           # HeartRateECG
    719: "respiratory_rate", 2274: "respiratory_rate", 2280: "respiratory_rate",
    710: "o2sat",                                                # SPO2 %
    709: "temperature",                                          # Temperature degrees C
    703: "map", 706: "map",                                      # arterial and non-invasive mean pressure
    713: "minute_volume", 2019: "minute_volume",                 # L
    725: "urine_output",                                         # Urine (c), hourly sum in mL
}
ICU_WARDS = ("CWIN", "INBD", "INIC", "INID")


def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(config={"memory_limit": "6GB", "threads": 4})


def load_stays(root: Path, horizon_hours: float) -> tuple[pd.DataFrame, dict]:
    """One stay per case, on its patient's timeline (``OffsetAfterFirstAdmission``), with the unit at ICU admission.

    ``first_careunit`` is the ward of the last ``unitlog`` entry (by id, which
    increases with time) among the four wards logged at exactly ``ICUOffset``:
    known when the stay starts.  Cases without one are dropped.  ``last_careunit``
    is ``cases.HospitalUnit`` (the last unit), kept as retrospective metadata only.
    Also returns the case counts behind that definition.
    """

    cases = pd.read_csv(root / "cases.csv.gz", usecols=["CaseID", "PatientID", "ICUOffset", "TimeOfStay", "HospitalUnit",
                                                       "OffsetAfterFirstAdmission"])
    names = pd.read_csv(root / "d_references.csv.gz", usecols=["ReferenceGlobalID", "ReferenceValue"]).set_index("ReferenceGlobalID")["ReferenceValue"]
    log = pd.read_csv(root / "unitlog.csv.gz", usecols=["id", "CaseID", "Offset", "HospitalUnit"])
    log = log.assign(unit=log["HospitalUnit"].map(names)).merge(cases[["CaseID", "ICUOffset"]], on="CaseID")
    log = log[log["unit"].isin(ICU_WARDS) & (log["Offset"] == log["ICUOffset"])].sort_values(["CaseID", "id"])
    first_unit = log.drop_duplicates("CaseID", keep="last").set_index("CaseID")["unit"]
    counts = {"cases": int(cases["CaseID"].nunique()),
              "cases_without_admission_unit": int((~cases["CaseID"].isin(first_unit.index)).sum()),
              "cases_with_several_admission_units": int((log.groupby("CaseID")["unit"].nunique() > 1).sum())}
    if cases["OffsetAfterFirstAdmission"].isna().any():
        raise ValueError("OffsetAfterFirstAdmission missing: the patient timeline is undefined")
    cases = cases.dropna(subset=["ICUOffset", "TimeOfStay"])
    cases = cases[cases["CaseID"].isin(first_unit.index) & (cases["TimeOfStay"] > cases["ICUOffset"] + horizon_hours * 3600)]
    cases = cases.sort_values("CaseID").reset_index(drop=True)
    base = cases["OffsetAfterFirstAdmission"].astype(float)
    stays = pd.DataFrame({
        "subject_id": cases["PatientID"].astype(str),
        "stay_id": cases["CaseID"].astype(int),
        "first_careunit": cases["CaseID"].map(first_unit).astype(str),
        "last_careunit": cases["HospitalUnit"].map(names).fillna("unknown").astype(str),
        "base_s": base,
        "start_s": cases["ICUOffset"].astype(float),
        "stop_s": cases["TimeOfStay"].astype(float),
        "intime": BASE_TIME + pd.to_timedelta(base + cases["ICUOffset"].astype(float), unit="s"),
        "outtime": BASE_TIME + pd.to_timedelta(base + cases["TimeOfStay"].astype(float), unit="s"),
    })
    return stays, counts


def _in_stay(c: duckdb.DuckDBPyConnection, stays: pd.DataFrame, query: str, duration_s: float = 0.0) -> pd.DataFrame:
    """Rows of ``query`` (CaseID, item, offset_s, value) whose whole window [offset_s - duration_s, offset_s] lies in the stay."""

    c.register("stays", stays[["subject_id", "stay_id", "base_s", "start_s", "stop_s"]])
    rows = c.sql(f"""
        SELECT s.subject_id, s.stay_id, q.item, s.base_s + q.offset_s AS clock_s, q.value
        FROM ({query}) q JOIN stays s ON q.CaseID = s.stay_id
        WHERE q.offset_s - {float(duration_s)} >= s.start_s AND q.offset_s <= s.stop_s AND q.value IS NOT NULL
    """).df()
    return rows.assign(charttime=BASE_TIME + pd.to_timedelta(rows["clock_s"], unit="s"))


def as_measurements(rows: pd.DataFrame, var: pd.Series, valuenum: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"subject_id": rows["subject_id"], "stay_id": rows["stay_id"], "charttime": rows["charttime"],
                         "var": var, "valuenum": valuenum})


def collapse(measurements: pd.DataFrame) -> pd.DataFrame:
    """One value per stay, variable and time: simultaneous values from several items are averaged (order-invariant)."""

    return (measurements.groupby(["subject_id", "stay_id", "var", "charttime"], as_index=False, sort=True)["valuenum"].mean()
            [["subject_id", "stay_id", "charttime", "var", "valuenum"]])


def load_labs(c: duckdb.DuckDBPyConnection, root: Path, stays: pd.DataFrame) -> pd.DataFrame:
    items = ",".join(str(i) for i in LAB_ITEMS)
    rows = _in_stay(c, stays, f"""
        SELECT CaseID, LaboratoryID AS item, CAST("Offset" AS DOUBLE) AS offset_s, CAST(LaboratoryValue AS DOUBLE) AS value
        FROM read_csv('{root / "laboratory.csv.gz"}', header = true, auto_detect = true)
        WHERE LaboratoryID IN ({items})""")
    return as_measurements(rows, rows["item"].map({k: v[0] for k, v in LAB_ITEMS.items()}),
                           rows["value"] * rows["item"].map({k: v[1] for k, v in LAB_ITEMS.items()}))


def load_signals(c: duckdb.DuckDBPyConnection, root: Path, stays: pd.DataFrame) -> pd.DataFrame:
    """Hourly aggregates whose whole hour lies in the stay, timed at the end of their hour."""

    items = ",".join(str(i) for i in SIGNAL_ITEMS)
    rows = _in_stay(c, stays, f"""
        SELECT CaseID, DataID AS item, CAST("Offset" AS DOUBLE) + {HOURLY_AGGREGATE_S} AS offset_s, CAST(Val AS DOUBLE) AS value
        FROM read_csv('{root / "data_float_h.csv.gz"}', header = true, auto_detect = true,
                      columns = {{'id': 'BIGINT', 'CaseID': 'BIGINT', 'DataID': 'BIGINT', 'Offset': 'BIGINT', 'Val': 'DOUBLE',
                                  'cnt': 'BIGINT', 'rawdata': 'VARCHAR'}})
        WHERE DataID IN ({items})""", duration_s=HOURLY_AGGREGATE_S)
    return as_measurements(rows, rows["item"].map(SIGNAL_ITEMS), rows["value"])


def add_calculated_bicarbonate(measurements: pd.DataFrame) -> pd.DataFrame:
    """HCO3 from same-time (validated) pH and PaCO2 where no validated bicarbonate exists at that time."""

    key = ["subject_id", "stay_id", "charttime"]
    wide = (measurements[measurements["var"].isin(["ph", "paco2", "bicarbonate"])]
            .pivot_table(index=key, columns="var", values="valuenum", aggfunc="mean")
            .reindex(columns=["ph", "paco2", "bicarbonate"]))
    derive = wide["bicarbonate"].isna() & wide["ph"].notna() & wide["paco2"].notna()
    rows = wide[derive].reset_index()
    added = rows[key].assign(var="bicarbonate", valuenum=calculated_bicarbonate(rows["ph"], rows["paco2"]))
    return pd.concat([measurements, contract.clean_measurements(added)], ignore_index=True)


def measurements_for(c: duckdb.DuckDBPyConnection, root: Path, stays: pd.DataFrame) -> pd.DataFrame:
    """Validate measured inputs, collapse simultaneous values, then derive (validated) bicarbonate."""

    measured = collapse(contract.clean_measurements(pd.concat([load_labs(c, root, stays), load_signals(c, root, stays)],
                                                              ignore_index=True)))
    return collapse(add_calculated_bicarbonate(measured)).sort_values(["stay_id", "charttime"]).reset_index(drop=True)


def build(root: Path, horizon_hours: float, anchor_step_hours: float) -> tuple[pd.DataFrame, dict]:
    contract.DELTA_H = float(horizon_hours)
    contract.ANCHOR_STEP_H = float(anchor_step_hours)
    contract.TARGET_TOL_H = float(min(2.0, horizon_hours / 2.0))
    suffix = f"_tp{int(horizon_hours)}" if float(horizon_hours).is_integer() else f"_tp{horizon_hours:g}"
    stays, case_counts = load_stays(root, horizon_hours)
    measurements = measurements_for(con(), root, stays)
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
    frame["hospitalid"] = "SICdb"
    report = {
        "artifact": "SICdb 1.0.8 ICU observation transitions (MIMIC-IV v5 contract; lockbox under protocol v10)",
        "delta_h": float(horizon_hours),
        "anchor_step_h": float(anchor_step_hours),
        "lookback_h": contract.LOOKBACK_H,
        "target_tolerance_h": contract.TARGET_TOL_H,
        "episode_max_h": contract.EPISODE_MAX_H,
        "case_counts": case_counts,
        "selected_stays": int(len(stays)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "rows": int(len(frame)),
        "careunits": sorted(frame["first_careunit"].unique().tolist()),
        "careunit_definition": "ward at ICU admission, fixed for the stay (last of the four wards logged at exactly ICUOffset); "
                               "last_careunit = cases.HospitalUnit, retrospective",
        "measurement_rows_by_var": {str(k): int(v) for k, v in measurements["var"].value_counts().items()},
        "pair_counts": contract.pair_counts(frame, suffix),
        "map_semantics": "validated measured arterial or non-invasive mean only",
        "signal_timing": "hourly aggregates timed at the end of their hour; only hours wholly inside the stay",
        "simultaneous_values": "averaged per stay, variable and time after validation",
        "glucose_poc_available": False,
        "fio2_peep_mapped": False,
        "treatment_context_available": False,
        "patient_ids_included_in_report": False,
        "causal_claim_allowed": False,
    }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory with the SICdb csv.gz files")
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
