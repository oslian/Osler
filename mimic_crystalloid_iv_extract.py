"""MIMIC-IV target-trial data for balanced crystalloids versus 0.9% saline (calendar-time preference IV).

Target trial (specified before any outcome was extracted), mirroring the eICU
design (``eicu_crystalloid_iv_extract.py``) on an independent database:

- eligibility: adults, first ICU stay of the admission, >= 1,000 mL of typed
  isotonic crystalloid (NaCl 0.9% or balanced) in the first 24 h after ICU
  admission, a chemistry chloride from 12 h before to 1 h after ICU admission,
  and >= 20 eligible stays in the stay's estimated calendar year;
- strategies: predominantly balanced (balanced share of typed volume >= 0.5)
  versus predominantly saline over the first 24 h; time zero: ICU admission;
- instrument: the balanced share among the other eligible stays of the same
  estimated calendar year (practice drift; SMART was published in 2018);
- outcomes (``--part outcomes``, only after the protocol freeze): chemistry
  chloride and bicarbonate change from baseline to the value nearest 24 h
  (18-30 h), platelet change as a negative control.

Fluids (inputevents, main order component only, rewritten rows dropped):
saline = NaCl 0.9% (225158); balanced = LR (225828), Ringers Lactate (220955),
Ringers (220953), Ringers Acetate (220956), Normosol (221208).  Dextrose-
containing, 0.45% and hypertonic solutions are not isotonic crystalloids here.

MIMIC-IV dates are shifted per patient; the real calendar year of a stay is
estimated as the start of ``anchor_year_group`` + 1 + (ICU admission year -
``anchor_year``), accurate to about +-1 year.  Factual data; research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MIMIC = Path("/Users/chunyouchang/Downloads/mimic-iv-3.1")
SALINE_ITEMS = {225158}
BALANCED_ITEMS = {225828, 220955, 220953, 220956, 221208}
WINDOW_H = 24
MIN_VOLUME_ML = 1000.0
MIN_CLUSTER_STAYS = 20
BASELINE_WINDOW_H = (-12, 1)
OUTCOME_WINDOW_H = (18, 30)
BASELINE_LABS = {50902: "chloride", 50983: "sodium", 50882: "bicarbonate", 50912: "creatinine", 51006: "bun",
                 50931: "glucose", 50971: "potassium", 51265: "platelets", 51222: "hemoglobin"}
OUTCOME_LABS = {50902: "chloride", 50882: "bicarbonate", 51265: "platelets"}


def stays() -> pd.DataFrame:
    icu = pd.read_csv(MIMIC / "icu" / "icustays.csv.gz", parse_dates=["intime"])
    icu = icu.sort_values("intime").drop_duplicates("hadm_id")              # first ICU stay of the admission
    patients = pd.read_csv(MIMIC / "hosp" / "patients.csv.gz")
    admissions = pd.read_csv(MIMIC / "hosp" / "admissions.csv.gz", usecols=["hadm_id", "admission_type", "admission_location"])
    frame = icu.merge(patients, on="subject_id").merge(admissions, on="hadm_id", how="left")
    years = frame["intime"].dt.year - frame["anchor_year"]
    frame["age"] = frame["anchor_age"] + years
    frame["estimated_year"] = frame["anchor_year_group"].str[:4].astype(int) + 1 + years
    return frame[frame["age"] >= 18].reset_index(drop=True)


def fluid_volumes(frame: pd.DataFrame) -> pd.DataFrame:
    intime = frame.set_index("stay_id")["intime"]
    pieces = []
    for chunk in pd.read_csv(MIMIC / "icu" / "inputevents.csv.gz", chunksize=2_000_000, parse_dates=["starttime"],
                             usecols=["stay_id", "starttime", "itemid", "amount", "amountuom", "ordercomponenttypedescription",
                                      "statusdescription", "patientweight"]):
        chunk = chunk[chunk["itemid"].isin(SALINE_ITEMS | BALANCED_ITEMS) & chunk["stay_id"].isin(intime.index)
                      & (chunk["ordercomponenttypedescription"] == "Main order parameter")
                      & (chunk["statusdescription"] != "Rewritten") & (chunk["amountuom"].str.lower() == "ml")]
        hours = (chunk["starttime"] - intime.reindex(chunk["stay_id"]).to_numpy()).dt.total_seconds() / 3600
        chunk = chunk[(hours >= 0) & (hours < WINDOW_H)]
        pieces.append(pd.DataFrame({"stay_id": chunk["stay_id"], "amount": chunk["amount"],
                                    "kind": np.where(chunk["itemid"].isin(BALANCED_ITEMS), "balanced", "saline"),
                                    "weight": chunk["patientweight"]}))
    events = pd.concat(pieces)
    volumes = events.pivot_table(index="stay_id", columns="kind", values="amount", aggfunc="sum", fill_value=0.0)
    volumes = volumes.reindex(columns=["balanced", "saline"], fill_value=0.0)
    volumes["admissionweight"] = events.groupby("stay_id")["weight"].first()
    return volumes


def nearest_labs(frame: pd.DataFrame, items: dict[int, str], window_h: tuple[int, int], anchor_h: float) -> pd.DataFrame:
    """Value nearest ``anchor_h`` hours after ICU admission, matched by subject and time (emergency labs lack hadm_id)."""

    stays = frame[["subject_id", "stay_id", "intime"]]
    pieces = []
    for chunk in pd.read_csv(MIMIC / "hosp" / "labevents.csv.gz", chunksize=5_000_000, parse_dates=["charttime"],
                             usecols=["subject_id", "itemid", "charttime", "valuenum"]):
        chunk = chunk[chunk["itemid"].isin(items) & chunk["subject_id"].isin(stays["subject_id"])].dropna(subset=["valuenum"])
        joined = chunk.merge(stays, on="subject_id")
        joined["hours"] = (joined["charttime"] - joined["intime"]).dt.total_seconds() / 3600
        pieces.append(joined[(joined["hours"] >= window_h[0]) & (joined["hours"] <= window_h[1])])
    labs = pd.concat(pieces)
    labs["distance"] = (labs["hours"] - anchor_h).abs()
    labs = labs.sort_values("distance").drop_duplicates(["stay_id", "itemid"])
    return labs.pivot(index="stay_id", columns="itemid", values="valuenum").rename(columns=items)


def design() -> tuple[pd.DataFrame, dict]:
    cohort = stays()
    volumes = fluid_volumes(cohort)
    typed = volumes["balanced"] + volumes["saline"]
    frame = cohort.merge(volumes[typed >= MIN_VOLUME_ML].reset_index(), on="stay_id")
    baseline = nearest_labs(frame, BASELINE_LABS, BASELINE_WINDOW_H, 0.0).add_prefix("baseline_")
    frame = frame.merge(baseline.reset_index(), on="stay_id", how="left")
    frame = frame[frame["baseline_chloride"].notna()].copy()
    frame["balanced_share"] = frame["balanced"] / (frame["balanced"] + frame["saline"])
    frame["treated_balanced"] = (frame["balanced_share"] >= 0.5).astype(int)
    frame["typed_volume_ml"] = frame["balanced"] + frame["saline"]
    frame["cluster_id"] = frame["estimated_year"].astype(str)
    sizes = frame.groupby("cluster_id")["stay_id"].transform("size")
    frame = frame[sizes >= MIN_CLUSTER_STAYS].copy()
    total = frame.groupby("cluster_id")["treated_balanced"].transform("sum")
    count = frame.groupby("cluster_id")["treated_balanced"].transform("size")
    frame["instrument"] = (total - frame["treated_balanced"]) / (count - 1)
    frame = frame.rename(columns={"stay_id": "stay_key", "first_careunit": "careunit"})
    report = {"stays": int(len(frame)), "clusters": int(frame["cluster_id"].nunique()),
              "treated_balanced_share": round(float(frame["treated_balanced"].mean()), 4), "outcomes_included": False}
    keep = ["stay_key", "subject_id", "cluster_id", "instrument", "treated_balanced", "balanced_share", "typed_volume_ml",
            "estimated_year", "age", "gender", "careunit", "admission_type", "admission_location", "admissionweight",
            *[c for c in frame.columns if c.startswith("baseline_")], "intime", "hadm_id"]
    return frame[keep].reset_index(drop=True), report


def outcomes(design_frame: pd.DataFrame) -> pd.DataFrame:
    frame = design_frame.rename(columns={"stay_key": "stay_id"})
    later = nearest_labs(frame, OUTCOME_LABS, OUTCOME_WINDOW_H, 24.0).add_prefix("day1_")
    return later.reset_index().rename(columns={"stay_id": "stay_key"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("design", "outcomes"), required=True)
    parser.add_argument("--design", type=Path, default=Path("mimic_crystalloid_iv_design.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path("mimic_crystalloid_iv_outcomes.parquet"))
    args = parser.parse_args()
    if args.part == "design":
        frame, report = design()
        frame.to_parquet(args.design, index=False)
        args.design.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    else:
        outcomes(pd.read_parquet(args.design)).to_parquet(args.outcomes, index=False)
        print("outcomes written")


if __name__ == "__main__":
    main()
