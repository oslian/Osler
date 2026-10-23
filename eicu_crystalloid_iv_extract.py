"""eICU target-trial data for balanced crystalloids versus 0.9% saline (hospital-preference IV).

Target trial (specified before any outcome was extracted):

- eligibility: adults (>= 18 y), first ICU stay of the hospital stay, at least
  1,000 mL of typed isotonic crystalloid (0.9% saline or balanced) in the first
  24 h after ICU admission, untyped crystalloid ("crystalloids", "IVF") below
  half of all crystalloid volume, a chloride measured from 12 h before to 1 h
  after ICU admission, and a hospital with >= 20 eligible stays;
- strategies: predominantly balanced (balanced share of typed volume >= 0.5)
  versus predominantly saline, over the first 24 h;
- time zero: ICU admission;
- instrument: the hospital's balanced share among its other eligible stays
  (leave-one-out preference);
- outcomes (``--part outcomes``, extracted only after the protocol freeze):
  chloride and bicarbonate change from baseline to the measurement nearest
  24 h (18-30 h), and platelet change as a negative control.

Fluid classification is by intake label: balanced = lactated Ringer's,
Plasma-Lyte, Normosol, Isolyte (not dextrose-containing); saline = 0.9% sodium
chloride / normal saline infusions and boluses (with or without added KCl);
excluded = drug carriers (``in sodium chloride``, ``ivpb``, ``pca``, ``-in``),
flushes, KVO, dextrose-containing, 0.45% and 3% saline; untyped = "crystalloids",
"ivf" without a type.  Factual observation data; research-only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


EICU = Path("/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0")
WINDOW_MIN = 24 * 60
MIN_VOLUME_ML = 1000.0
MAX_ENTRY_ML = 5000.0
MIN_HOSPITAL_STAYS = 20
BASELINE_WINDOW = (-12 * 60, 60)
OUTCOME_WINDOW = (18 * 60, 30 * 60)
BASELINE_LABS = {"chloride": "chloride", "bicarbonate": "bicarbonate", "sodium": "sodium", "creatinine": "creatinine",
                 "BUN": "bun", "glucose": "glucose", "potassium": "potassium", "platelets x 1000": "platelets", "Hgb": "hemoglobin"}
OUTCOME_LABS = {"chloride": "chloride", "bicarbonate": "bicarbonate", "platelets x 1000": "platelets"}

EXCLUDE = re.compile(r"\bin sodium chloride|\bin nacl|ivpb|\bpca\b|\)-in\b|-in$|flush|kvo|dextrose|\bd5\b|0\.45|1/2|half|3 ?%|hypertonic|heparin|\+ add", re.I)
BALANCED = re.compile(r"lactated ringer|\blr\b|plasma-?lyte|normosol|isolyte", re.I)
SALINE = re.compile(r"0\.9\s*%?\s*(sodium chloride|nacl|ns)|sodium chloride\s*0\.9|normal saline|\bns\b", re.I)
UNTYPED = re.compile(r"^crystalloids?\b|^ivf\b|^iv fluids?\b|^maintenance ivf", re.I)


def classify(label: str) -> str | None:
    text = str(label).strip().lower()
    if EXCLUDE.search(text):
        return None
    if BALANCED.search(text):
        return "balanced"
    if SALINE.search(text):
        return "saline"
    if UNTYPED.search(text):
        return "untyped"
    return None


def stays() -> pd.DataFrame:
    patient = pd.read_csv(EICU / "patient.csv.gz", usecols=[
        "patientunitstayid", "uniquepid", "gender", "age", "hospitalid", "apacheadmissiondx", "unittype",
        "unitadmitsource", "unitvisitnumber", "admissionweight"])
    patient["age"] = pd.to_numeric(patient["age"].replace("> 89", "90"), errors="coerce")
    patient = patient[(patient["age"] >= 18) & (patient["unitvisitnumber"] == 1)]
    hospital = pd.read_csv(EICU / "hospital.csv.gz")
    return patient.merge(hospital, on="hospitalid", how="left")


def fluid_volumes(stay_ids: set) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(EICU / "intakeOutput.csv.gz", chunksize=2_000_000,
                             usecols=["patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel", "cellvaluenumeric"]):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)
                      & chunk["cellpath"].str.contains("Intake", case=False, na=False)
                      & chunk["intakeoutputoffset"].between(0, WINDOW_MIN - 1)]
        kind = chunk["celllabel"].map(classify)
        volume = pd.to_numeric(chunk["cellvaluenumeric"], errors="coerce")
        keep = kind.notna() & volume.between(0, MAX_ENTRY_ML)
        pieces.append(pd.DataFrame({"patientunitstayid": chunk.loc[keep, "patientunitstayid"],
                                    "kind": kind[keep], "volume": volume[keep]}))
    volumes = pd.concat(pieces).pivot_table(index="patientunitstayid", columns="kind", values="volume", aggfunc="sum", fill_value=0.0)
    return volumes.reindex(columns=["balanced", "saline", "untyped"], fill_value=0.0)


def nearest_labs(stay_ids: set, names: dict[str, str], window: tuple[int, int], anchor: int) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(EICU / "lab.csv.gz", chunksize=4_000_000,
                             usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"]):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids) & chunk["labname"].isin(names)
                      & chunk["labresultoffset"].between(*window)]
        pieces.append(chunk.dropna(subset=["labresult"]))
    labs = pd.concat(pieces)
    labs["distance"] = (labs["labresultoffset"] - anchor).abs()
    labs = labs.sort_values("distance").drop_duplicates(["patientunitstayid", "labname"])
    wide = labs.pivot(index="patientunitstayid", columns="labname", values="labresult")
    return wide.rename(columns=names)


def design() -> tuple[pd.DataFrame, dict]:
    cohort = stays()
    volumes = fluid_volumes(set(cohort["patientunitstayid"]))
    typed = volumes["balanced"] + volumes["saline"]
    eligible = volumes[(typed >= MIN_VOLUME_ML) & (volumes["untyped"] < 0.5 * (typed + volumes["untyped"]))]
    frame = cohort.merge(eligible.reset_index(), on="patientunitstayid")
    baseline = nearest_labs(set(frame["patientunitstayid"]), BASELINE_LABS, BASELINE_WINDOW, 0).add_prefix("baseline_")
    frame = frame.merge(baseline.reset_index(), on="patientunitstayid", how="left")
    frame = frame[frame["baseline_chloride"].notna()]
    frame["balanced_share"] = frame["balanced"] / (frame["balanced"] + frame["saline"])
    frame["treated_balanced"] = (frame["balanced_share"] >= 0.5).astype(int)
    frame["typed_volume_ml"] = frame["balanced"] + frame["saline"]
    sizes = frame.groupby("hospitalid")["patientunitstayid"].transform("size")
    frame = frame[sizes >= MIN_HOSPITAL_STAYS].copy()
    total = frame.groupby("hospitalid")["treated_balanced"].transform("sum")
    count = frame.groupby("hospitalid")["treated_balanced"].transform("size")
    frame["instrument_hospital_preference"] = (total - frame["treated_balanced"]) / (count - 1)
    report = {
        "stays": int(len(frame)),
        "hospitals": int(frame["hospitalid"].nunique()),
        "treated_balanced_share": round(float(frame["treated_balanced"].mean()), 4),
        "outcomes_included": False,
    }
    return frame.reset_index(drop=True), report


def outcomes(design_frame: pd.DataFrame) -> pd.DataFrame:
    later = nearest_labs(set(design_frame["patientunitstayid"]), OUTCOME_LABS, OUTCOME_WINDOW, 24 * 60).add_prefix("day1_")
    return later.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("design", "outcomes"), required=True)
    parser.add_argument("--design", type=Path, default=Path("eicu_crystalloid_iv_design.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path("eicu_crystalloid_iv_outcomes.parquet"))
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
