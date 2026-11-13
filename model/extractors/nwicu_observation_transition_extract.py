"""Extract measurement-pure Northwestern ICU observations.

NWICU uses a MIMIC-like table layout but a distinct item dictionary.  This
adapter therefore maps only labels verified in NWICU's own dictionaries.  It
does not reuse MIMIC item IDs and does not infer absent urine or treatment
channels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from extractors.eicu_sepsis_transition_extract import PLAUSIBLE
from extractors.mimiciv_observation_transition_extract import EXTRA_PLAUSIBLE


NWICU_LAB_ITEMIDS: dict[str, tuple[int, ...]] = {
    "glucose": (100001, 100045),
    "creatinine": (100002,),
    "calcium": (100003,),
    "bun": (100004,),
    "hematocrit": (100006,),
    "hemoglobin": (100007,),
    "magnesium": (100009,),
    "sodium": (100010, 100050),
    "potassium": (100011, 100047),
    "chloride": (100012,),
    # NWICU labels this serum chemistry measurement "Total CO2".
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
    "troponin_i": (100059,),
    "uric_acid": (100069,),
    "cpk": (100085,),
    "serum_osmolality": (100089,),
    "lipase": (100091,),
    "tsh": (100092,),
    "ck_mb": (100099,),
    "free_t4": (100103,),
    "cortisol": (100128,),
    "amylase": (100137,),
    "crp_hs": (100251,),
    "ph": (100339,),
}

NWICU_VITAL_ITEMIDS: dict[str, tuple[int, ...]] = {
    "heart_rate": (320045,),
    "respiratory_rate": (320210,),
    "o2sat": (320277,),
    "temperature_f": (323761,),
    "body_weight_ounces": (326531,),
}

ARTERIAL_SYSTOLIC = 320050
ARTERIAL_DIASTOLIC = 320051
NIBP_SYSTOLIC = 320179
NIBP_DIASTOLIC = 320180


def con():
    import duckdb

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    return connection


def csv_expr(path: Path) -> str:
    return f"read_csv_auto('{path}', all_varchar=false)"


def hosp_path(root: Path, name: str) -> Path:
    return Path(root) / "data" / "nw_hosp" / f"{name}.csv.gz"


def icu_path(root: Path, name: str) -> Path:
    return Path(root) / "data" / "nw_icu" / f"{name}.csv.gz"


def sql_case(mapping: dict[str, tuple[int, ...]], alias: str) -> str:
    clauses = []
    for variable, itemids in mapping.items():
        values = ",".join(str(value) for value in itemids)
        clauses.append(f"WHEN {alias}.itemid IN ({values}) THEN '{variable}'")
    return " ".join(clauses)


def pull_stay_meta(connection, root: Path, max_stays: int | None) -> pd.DataFrame:
    limit = "" if max_stays is None else f"LIMIT {int(max_stays)}"
    frame = connection.execute(
        f"""
        SELECT subject_id, hadm_id, stay_id, first_careunit, last_careunit,
               intime, outtime
        FROM {csv_expr(icu_path(root, 'icustays'))}
        WHERE outtime > intime + INTERVAL '6 hours'
        ORDER BY stay_id
        {limit}
        """
    ).df()
    for column in ("intime", "outtime"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def register_stays(connection, meta: pd.DataFrame) -> None:
    connection.register("_nwicu_stays", meta[["stay_id"]])
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE target_stays AS "
        "SELECT stay_id FROM _nwicu_stays"
    )


def _derive_map(raw_pressure: pd.DataFrame) -> pd.DataFrame:
    """Derive MAP from a same-time SBP/DBP pair, preferring arterial BP."""

    if raw_pressure.empty:
        return pd.DataFrame(
            columns=("subject_id", "stay_id", "charttime", "var", "valuenum", "quality")
        )
    frame = raw_pressure.copy()
    frame["valuenum"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    frame = frame.dropna(subset=["valuenum"])
    rows = []
    pair_contract = (
        (ARTERIAL_SYSTOLIC, ARTERIAL_DIASTOLIC, "arterial_bp"),
        (NIBP_SYSTOLIC, NIBP_DIASTOLIC, "noninvasive_bp"),
    )
    keys = ["subject_id", "stay_id", "charttime"]
    used = set()
    for systolic_id, diastolic_id, source in pair_contract:
        systolic = (
            frame.loc[frame["itemid"].eq(systolic_id)]
            .groupby(keys, as_index=False)["valuenum"]
            .median()
            .rename(columns={"valuenum": "systolic"})
        )
        diastolic = (
            frame.loc[frame["itemid"].eq(diastolic_id)]
            .groupby(keys, as_index=False)["valuenum"]
            .median()
            .rename(columns={"valuenum": "diastolic"})
        )
        paired = systolic.merge(diastolic, on=keys, how="inner")
        if paired.empty:
            continue
        paired["_key"] = list(map(tuple, paired[keys].to_numpy()))
        paired = paired.loc[~paired["_key"].isin(used)].copy()
        used.update(paired["_key"])
        paired["var"] = "map"
        paired["valuenum"] = (paired["systolic"] + 2.0 * paired["diastolic"]) / 3.0
        paired["quality"] = "derived"
        paired["map_source"] = source
        rows.append(paired[keys + ["var", "valuenum", "quality", "map_source"]])
    if not rows:
        return pd.DataFrame(
            columns=("subject_id", "stay_id", "charttime", "var", "valuenum", "quality")
        )
    return pd.concat(rows, ignore_index=True, sort=False)


def clean_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cleaned = frame.copy()
    cleaned["charttime"] = pd.to_datetime(cleaned["charttime"], errors="coerce")
    cleaned["valuenum"] = pd.to_numeric(cleaned["valuenum"], errors="coerce")
    cleaned["quality"] = cleaned.get("quality", "observed")

    temperature = cleaned["var"].eq("temperature_f")
    cleaned.loc[temperature, "valuenum"] = (
        cleaned.loc[temperature, "valuenum"] - 32.0
    ) * (5.0 / 9.0)
    cleaned.loc[temperature, "var"] = "temperature"

    weight = cleaned["var"].eq("body_weight_ounces")
    cleaned.loc[weight, "valuenum"] = cleaned.loc[weight, "valuenum"] * 0.028349523125
    cleaned.loc[weight, "var"] = "body_weight"

    unit = cleaned.get("valueuom", pd.Series("", index=cleaned.index)).fillna("").str.lower()
    crp = cleaned["var"].isin(("crp", "crp_hs")) & unit.eq("mg/l")
    cleaned.loc[crp, "valuenum"] = cleaned.loc[crp, "valuenum"] / 10.0

    for variable, (lower, upper) in {**PLAUSIBLE, **EXTRA_PLAUSIBLE}.items():
        selected = cleaned["var"].eq(variable)
        cleaned.loc[selected, "valuenum"] = cleaned.loc[selected, "valuenum"].where(
            cleaned.loc[selected, "valuenum"].between(lower, upper)
        )
    cleaned = cleaned.dropna(subset=["charttime", "valuenum"])
    return cleaned.sort_values(["stay_id", "charttime", "var"]).reset_index(drop=True)


def pull_labs(connection, root: Path) -> pd.DataFrame:
    itemids = sorted({value for values in NWICU_LAB_ITEMIDS.values() for value in values})
    cases = sql_case(NWICU_LAB_ITEMIDS, "le")
    frame = connection.execute(
        f"""
        SELECT ie.subject_id, ie.stay_id, le.charttime,
               CASE {cases} END AS var, le.valuenum, le.valueuom
        FROM {csv_expr(hosp_path(root, 'labevents'))} le
        JOIN {csv_expr(icu_path(root, 'icustays'))} ie
          ON le.hadm_id = ie.hadm_id
         AND le.charttime BETWEEN ie.intime AND ie.outtime
        JOIN target_stays ts ON ie.stay_id = ts.stay_id
        WHERE le.itemid IN ({','.join(str(value) for value in itemids)})
          AND le.valuenum IS NOT NULL
        """
    ).df()
    return clean_measurements(frame)


def pull_vitals(connection, root: Path) -> pd.DataFrame:
    vital_ids = sorted({value for values in NWICU_VITAL_ITEMIDS.values() for value in values})
    pressure_ids = (
        ARTERIAL_SYSTOLIC,
        ARTERIAL_DIASTOLIC,
        NIBP_SYSTOLIC,
        NIBP_DIASTOLIC,
    )
    cases = sql_case(NWICU_VITAL_ITEMIDS, "ce")
    frame = connection.execute(
        f"""
        SELECT ce.subject_id, ce.stay_id, ce.charttime,
               CASE {cases} END AS var, ce.valuenum, ce.valueuom
        FROM {csv_expr(icu_path(root, 'chartevents'))} ce
        JOIN target_stays ts ON ce.stay_id = ts.stay_id
        WHERE ce.itemid IN ({','.join(str(value) for value in vital_ids)})
          AND ce.valuenum IS NOT NULL
        """
    ).df()
    pressure = connection.execute(
        f"""
        SELECT ce.subject_id, ce.stay_id, ce.charttime, ce.itemid, ce.valuenum
        FROM {csv_expr(icu_path(root, 'chartevents'))} ce
        JOIN target_stays ts ON ce.stay_id = ts.stay_id
        WHERE ce.itemid IN ({','.join(str(value) for value in pressure_ids)})
          AND ce.valuenum IS NOT NULL
        """
    ).df()
    derived_map = _derive_map(pressure)
    return clean_measurements(
        pd.concat([frame, derived_map], ignore_index=True, sort=False)
    )


__all__ = [
    "NWICU_LAB_ITEMIDS",
    "NWICU_VITAL_ITEMIDS",
    "clean_measurements",
    "con",
    "pull_labs",
    "pull_stay_meta",
    "pull_vitals",
    "register_stays",
]
