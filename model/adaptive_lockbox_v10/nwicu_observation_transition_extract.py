"""Build NWICU v0.1.0 observation transitions in the MIMIC-IV contract.

NWICU (Northwestern ICU) follows the MIMIC-IV table layout under ``nw_hosp`` and
``nw_icu`` but uses its own item ids.  Anchors, as-of state, future labels and
plausibility cleaning reuse ``mimiciv_observation_transition_extract`` so both
databases go through the same code path.

Known semantic differences from MIMIC-IV, declared in the external protocol:

- MAP: NWICU has no measured mean-pressure item.  MAP is derived as
  ``(SBP + 2 * DBP) / 3`` from same-time arterial-line or non-invasive pairs.
- Urine output and administered treatments (inputevents) are not available, so
  urine targets and treatment-context features are absent.

Factual observation data only; no causal or treatment-effect claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mimiciv_observation_transition_extract as contract
from osler_jepa.derived_measurements import mean_arterial_pressure


NWICU_LAB_ITEMIDS: dict[str, tuple[int, ...]] = {
    "glucose": (100001,),
    "glucose_bg": (100045,),
    "creatinine": (100002,),
    "calcium": (100003,),
    "bun": (100004,),
    "hematocrit": (100006,),
    "hemoglobin": (100007,),
    "magnesium": (100009,),
    "sodium": (100010, 100050),
    "potassium": (100011, 100047),
    "chloride": (100012,),
    "bicarbonate": (100013,),
    "platelets": (100014,),
    "phosphate": (100015,),
    "wbc": (100016,),
    "bilirubin": (100020,),
    "albumin": (100021,),
    "total_protein": (100027,),
    "inr": (100030,),
    "lactate": (100031,),
    "ast": (100032,),
    "alkaline_phos": (100033,),
    "anion_gap": (100040, 100041),
    "alt": (100042,),
    "ionized_calcium": (100044,),
    "ptt": (100046,),
    "fibrinogen": (100048,),
    "bilirubin_direct": (100049,),
    "ferritin": (100052,),
    "crp": (100053,),
    "ldh": (100055,),
    "triglycerides": (100056,),
    "troponin_t": (100057,),
    "troponin_i": (100059,),
    "uric_acid": (100069,),
    "bnp": (100071,),
    "cpk": (100085,),
    "serum_osmolality": (100089,),
    "lipase": (100091,),
    "tsh": (100092,),
    "ck_mb": (100099,),
    "free_t4": (100103,),
    "cortisol": (100128,),
    "pth": (100131,),
    "amylase": (100137,),
    "tpo_antibody": (100218,),
    "total_t4": (100276,),
    "ph": (100339,),
}
NWICU_VITAL_ITEMIDS: dict[str, tuple[int, ...]] = {
    "heart_rate": (320045,),
    "respiratory_rate": (320210,),
    "o2sat": (320277,),
    "temperature_f": (323761,),
}
BLOOD_PRESSURE_PAIRS = {
    "arterial": (320050, 320051),
    "noninvasive": (320179, 320180),
}


def load_stays(nwicu_dir: Path, horizon_hours: float) -> pd.DataFrame:
    stays = pd.read_csv(
        nwicu_dir / "nw_icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    stays = stays[stays["outtime"] > stays["intime"] + pd.Timedelta(hours=horizon_hours)]
    return stays.sort_values("stay_id").reset_index(drop=True)


def load_labs(nwicu_dir: Path, stays: pd.DataFrame, chunksize: int = 5_000_000) -> pd.DataFrame:
    item_to_var = {itemid: var for var, ids in NWICU_LAB_ITEMIDS.items() for itemid in ids}
    windows = stays[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]
    pieces = []
    for chunk in pd.read_csv(
        nwicu_dir / "nw_hosp" / "labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        dtype={"hadm_id": "float64", "itemid": "int64"},
        na_values=["NULL", ""],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["itemid"].isin(item_to_var)]
        chunk = chunk.assign(valuenum=pd.to_numeric(chunk["valuenum"], errors="coerce"))
        chunk = chunk[chunk["valuenum"].notna() & chunk["hadm_id"].notna()]
        if chunk.empty:
            continue
        chunk = chunk.assign(
            hadm_id=chunk["hadm_id"].astype("int64"),
            charttime=pd.to_datetime(chunk["charttime"]),
            var=chunk["itemid"].map(item_to_var),
        ).merge(windows, on="hadm_id", how="inner")
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["outtime"])]
        pieces.append(chunk[["subject_id", "stay_id", "charttime", "var", "valuenum"]])
    return pd.concat(pieces, ignore_index=True)


def derive_map(chart: pd.DataFrame) -> pd.DataFrame:
    """MAP = (SBP + 2 * DBP) / 3 from same-stay, same-time systolic/diastolic pairs."""

    pieces = []
    for systolic, diastolic in BLOOD_PRESSURE_PAIRS.values():
        sbp = chart.loc[chart["itemid"] == systolic, ["subject_id", "stay_id", "charttime", "valuenum"]]
        dbp = chart.loc[chart["itemid"] == diastolic, ["stay_id", "charttime", "valuenum"]]
        sbp = sbp.groupby(["subject_id", "stay_id", "charttime"], as_index=False)["valuenum"].mean()
        dbp = dbp.groupby(["stay_id", "charttime"], as_index=False)["valuenum"].mean()
        paired = sbp.merge(dbp, on=["stay_id", "charttime"], suffixes=("_sbp", "_dbp"))
        pieces.append(pd.DataFrame({
            "subject_id": paired["subject_id"],
            "stay_id": paired["stay_id"],
            "charttime": paired["charttime"],
            "var": "map",
            "valuenum": mean_arterial_pressure(paired["valuenum_sbp"], paired["valuenum_dbp"]),
        }).dropna(subset=["valuenum"]))
    return pd.concat(pieces, ignore_index=True)


def load_vitals(nwicu_dir: Path, stays: pd.DataFrame) -> pd.DataFrame:
    pair_items = {itemid for pair in BLOOD_PRESSURE_PAIRS.values() for itemid in pair}
    item_to_var = {itemid: var for var, ids in NWICU_VITAL_ITEMIDS.items() for itemid in ids}
    chart = pd.read_csv(
        nwicu_dir / "nw_icu" / "chartevents.csv.gz",
        usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
        dtype={"itemid": "int64"},
        na_values=["NULL", ""],
    )
    chart = chart[chart["itemid"].isin(set(item_to_var) | pair_items) & chart["stay_id"].isin(set(stays["stay_id"]))]
    chart = chart.assign(
        charttime=pd.to_datetime(chart["charttime"]),
        valuenum=pd.to_numeric(chart["valuenum"], errors="coerce"),
    ).dropna(subset=["valuenum"])
    direct = chart[chart["itemid"].isin(item_to_var)].assign(var=lambda frame: frame["itemid"].map(item_to_var))
    return pd.concat(
        [direct[["subject_id", "stay_id", "charttime", "var", "valuenum"]], derive_map(chart)],
        ignore_index=True,
    )


def build(nwicu_dir: Path, horizon_hours: float, anchor_step_hours: float) -> tuple[pd.DataFrame, dict]:
    contract.DELTA_H = float(horizon_hours)
    contract.ANCHOR_STEP_H = float(anchor_step_hours)
    contract.TARGET_TOL_H = float(min(2.0, horizon_hours / 2.0))
    suffix = f"_tp{int(horizon_hours)}" if float(horizon_hours).is_integer() else f"_tp{horizon_hours:g}"
    stays = load_stays(nwicu_dir, horizon_hours)
    measurements = contract.clean_measurements(
        pd.concat([load_labs(nwicu_dir, stays), load_vitals(nwicu_dir, stays)], ignore_index=True)
    )
    anchors = contract.build_anchors(stays)
    state_t = contract.assemble_state(anchors, measurements, "_t", "t", "backward", contract.LOOKBACK_H)
    state_f = contract.assemble_state(
        anchors, measurements, suffix, "t_plus", "nearest", contract.TARGET_TOL_H
    )
    frame = pd.concat(
        [
            anchors[["subject_id", "stay_id", "first_careunit", "last_careunit", "t", "t_plus"]],
            state_t.drop(columns=["stay_id", "t"]),
            state_f.drop(columns=["stay_id", "t_plus"]),
        ],
        axis=1,
    )
    # Every anchor is kept, labelled or not (see mimiciv_observation_transition_extract).
    frame = frame.reset_index(drop=True)
    frame["hospitalid"] = "NWICU"
    counts = contract.pair_counts(frame, suffix)
    report = {
        "artifact": "NWICU v0.1.0 whole-ICU observation transition cohort (MIMIC-IV contract)",
        "delta_h": float(horizon_hours),
        "anchor_step_h": float(anchor_step_hours),
        "lookback_h": contract.LOOKBACK_H,
        "target_tolerance_h": contract.TARGET_TOL_H,
        "episode_max_h": contract.EPISODE_MAX_H,
        "selected_stays": int(len(stays)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "rows": int(len(frame)),
        "careunits": int(frame["first_careunit"].nunique()),
        "measurement_rows_by_var": {str(k): int(v) for k, v in measurements["var"].value_counts().items()},
        "pair_counts": counts,
        "map_semantics": "derived (SBP + 2*DBP)/3 from same-time arterial or non-invasive pairs",
        "urine_output_available": False,
        "treatment_context_available": False,
        "patient_ids_included_in_report": False,
        "causal_claim_allowed": False,
    }
    return frame, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwicu-dir", type=Path, required=True)
    parser.add_argument("--horizon-hours", type=float, required=True)
    parser.add_argument("--anchor-step-hours", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, report = build(args.nwicu_dir, args.horizon_hours, args.anchor_step_hours)
    frame.to_parquet(args.output, index=False)
    report["output"] = str(args.output.resolve())
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "subjects", "stays")}, indent=2))


if __name__ == "__main__":
    main()
