"""Stay-level eICU disease-module membership rebuilt for MIMIC-IV-schema databases.

The eICU transition cohorts are disease modules (AKI, respiratory, ...).  This
module applies the same definitions to MIMIC-IV and NWICU ICU stays so that
intervals can be calibrated by module membership:

- diagnosis: any eICU module pattern in the admission's ICD titles or codes
  (letter-prefixed code patterns match ICD-10 codes, numeric ones ICD-9, both by
  prefix after removing dots);
- AKI also: KDIGO-like creatinine rise within the ICU stay (>= 0.3 mg/dL over the
  48h minimum, or >= 1.5x the 7-day minimum), as in the eICU AKI extractor;
- respiratory also: SpO2 <= 60%, as in the eICU respiratory extractor.

Declared deviations from eICU: membership is stay-level (MIMIC ICD codes carry no
timestamp, so there is no onset time), and short alphabetic patterns (<= 4
letters, e.g. "dic", "arf") match whole words only to avoid hits such as
"medical" or "dwarfism".
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_aki_transition_extract import AKI_PATTERNS
from eicu_body_system_configs import BODY_SYSTEM_CONFIGS
from eicu_respiratory_transition_extract import RESPIRATORY_PATTERNS
from mimiciv_observation_transition_extract import LAB_ITEMIDS


MODULE_PATTERNS: dict[str, tuple[str, ...]] = {
    "aki": AKI_PATTERNS,
    "respiratory": RESPIRATORY_PATTERNS,
    **{
        name: BODY_SYSTEM_CONFIGS[name].diagnosis_patterns
        for name in ("endocrine_stress", "cardiovascular_instability", "coagulopathy_heme", "electrolyte_acid_base")
    },
}
CODE_PATTERN = re.compile(r"[a-z]?\d{2,3}(\.\d{1,2})?")


@dataclass(frozen=True)
class DatabaseLayout:
    name: str
    hosp: str
    icu: str
    creatinine_itemids: tuple[int, ...]
    spo2_itemids: tuple[int, ...]


LAYOUTS = {
    "MIMIC-IV": DatabaseLayout("MIMIC-IV", "hosp", "icu", tuple(LAB_ITEMIDS["creatinine"]), (220277,)),
    "NWICU": DatabaseLayout("NWICU", "nw_hosp", "nw_icu", (100002,), (320277,)),
}


def split_patterns(patterns: tuple[str, ...]) -> tuple[list[str], list[re.Pattern], list[str]]:
    """Return (ICD code prefixes, whole-word regexes, substring phrases)."""

    codes, words, phrases = [], [], []
    for pattern in patterns:
        lowered = pattern.lower()
        if CODE_PATTERN.fullmatch(lowered):
            codes.append(lowered.replace(".", ""))
        elif lowered.isalpha() and len(lowered) <= 4:
            words.append(re.compile(rf"\b{re.escape(lowered)}\b"))
        else:
            phrases.append(lowered)
    return codes, words, phrases


def diagnosis_matches(titles: pd.Series, codes: pd.Series, versions: pd.Series, patterns: tuple[str, ...]) -> np.ndarray:
    code_prefixes, words, phrases = split_patterns(patterns)
    title = titles.fillna("").str.lower()
    code = codes.fillna("").str.lower().str.replace(".", "", regex=False)
    version = pd.to_numeric(versions, errors="coerce").to_numpy()
    hit = np.zeros(len(titles), dtype=bool)
    for phrase in phrases:
        hit |= title.str.contains(phrase, regex=False).to_numpy()
    for word in words:
        hit |= title.str.contains(word).to_numpy()
    for prefix in code_prefixes:
        wanted = 10 if prefix[0].isalpha() else 9
        hit |= (code.str.startswith(prefix).to_numpy() & (version == wanted))
    return hit


def kdigo_stays(creatinine: pd.DataFrame) -> set[int]:
    hits = set()
    for stay_id, group in creatinine.sort_values(["stay_id", "charttime"]).groupby("stay_id", sort=False):
        hours = group["charttime"].to_numpy().astype("datetime64[ns]").astype(np.int64) / 3.6e12
        values = group["valuenum"].to_numpy(dtype=np.float64)
        for index in range(1, len(values)):
            prior48 = values[:index][hours[:index] >= hours[index] - 48.0]
            prior168 = values[:index][hours[:index] >= hours[index] - 168.0]
            absolute = len(prior48) and values[index] - prior48.min() >= 0.3
            ratio = len(prior168) and prior168.min() > 0 and values[index] / prior168.min() >= 1.5
            if absolute or ratio:
                hits.add(int(stay_id))
                break
    return hits


def build_membership(root: Path, layout: DatabaseLayout) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect()
    stays = pd.read_csv(root / layout.icu / "icustays.csv.gz", usecols=["hadm_id", "stay_id", "intime", "outtime"])
    diagnoses = pd.read_csv(root / layout.hosp / "diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dictionary_path = root / layout.hosp / "d_icd_diagnoses.csv.gz"
    if dictionary_path.exists():
        dictionary = pd.read_csv(dictionary_path, dtype={"icd_code": str})
        diagnoses = diagnoses.merge(dictionary, on=["icd_code", "icd_version"], how="left")
    else:
        diagnoses["long_title"] = ""
    diagnoses = diagnoses.merge(stays[["hadm_id", "stay_id"]], on="hadm_id", how="inner")
    rows = []
    for module, patterns in MODULE_PATTERNS.items():
        hit = diagnosis_matches(diagnoses["long_title"], diagnoses["icd_code"], diagnoses["icd_version"], patterns)
        for stay_id in diagnoses.loc[hit, "stay_id"].unique():
            rows.append({"stay_id": int(stay_id), "module": module, "criteria": "diagnosis"})

    con.register("stays", stays)
    creatinine = con.execute(f"""
        SELECT s.stay_id,
               TRY_CAST(l.charttime AS TIMESTAMP) AS charttime,
               TRY_CAST(l.valuenum AS DOUBLE) AS valuenum
        FROM read_csv_auto('{root / layout.hosp / "labevents.csv.gz"}', all_varchar=true) l
        JOIN stays s ON TRY_CAST(l.hadm_id AS BIGINT) = s.hadm_id
        WHERE TRY_CAST(l.itemid AS BIGINT) IN ({",".join(map(str, layout.creatinine_itemids))})
          AND TRY_CAST(l.valuenum AS DOUBLE) BETWEEN 0.05 AND 30
          AND TRY_CAST(l.charttime AS TIMESTAMP) BETWEEN TRY_CAST(s.intime AS TIMESTAMP) AND TRY_CAST(s.outtime AS TIMESTAMP)
    """).df()
    for stay_id in kdigo_stays(creatinine):
        rows.append({"stay_id": stay_id, "module": "aki", "criteria": "kdigo_like_creatinine"})

    hypoxemic = con.execute(f"""
        SELECT DISTINCT TRY_CAST(stay_id AS BIGINT) AS stay_id
        FROM read_csv_auto('{root / layout.icu / "chartevents.csv.gz"}', all_varchar=true)
        WHERE TRY_CAST(itemid AS BIGINT) IN ({",".join(map(str, layout.spo2_itemids))})
          AND TRY_CAST(valuenum AS DOUBLE) > 40 AND TRY_CAST(valuenum AS DOUBLE) <= 60
    """).df()
    for stay_id in hypoxemic["stay_id"].dropna().astype(int):
        rows.append({"stay_id": int(stay_id), "module": "respiratory", "criteria": "spo2_le_60"})
    membership = pd.DataFrame(rows)
    return membership.groupby(["stay_id", "module"], as_index=False)["criteria"].agg(lambda item: "+".join(sorted(set(item))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    membership = build_membership(args.root, LAYOUTS[args.database])
    membership.to_parquet(args.output, index=False)
    stays = pd.read_csv(args.root / LAYOUTS[args.database].icu / "icustays.csv.gz", usecols=["stay_id"])
    summary = {
        module: {
            "stays": int((membership["module"] == module).sum()),
            "share_of_icu_stays": float((membership["module"] == module).sum() / len(stays)),
        }
        for module in MODULE_PATTERNS
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
