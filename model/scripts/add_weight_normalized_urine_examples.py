"""Add causal body-weight context and mL/kg/h urine to existing examples.

This bounded utility avoids rebuilding unrelated event examples while keeping
the same causal contract: only weight measured at or before the anchor is used.
The future urine target is divided by the anchor-known weight, never by a
future weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MIMIC_WEIGHT_ITEMIDS = (224639, 226512)


def _read_horizon(path: Path, horizon: float) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        filters=[("_source_horizon", "=", float(horizon))],
    )


def _eicu_anchor_weights(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    meta = pd.read_csv(
        data_root / "patient.csv.gz",
        usecols=["patientunitstayid", "admissionweight"],
        low_memory=False,
    ).rename(columns={"patientunitstayid": "stay_id"})
    meta["stay_id"] = meta["stay_id"].astype(str)
    meta["body_weight_t"] = pd.to_numeric(
        meta["admissionweight"], errors="coerce"
    ).where(lambda value: value.between(25.0, 300.0))
    output = frame.merge(
        meta[["stay_id", "body_weight_t"]].drop_duplicates("stay_id"),
        on="stay_id",
        how="left",
        validate="many_to_one",
    )
    output["body_weight_age_hr"] = (
        pd.to_datetime(output["anchor_time"], errors="coerce")
        - pd.Timestamp("2100-01-01")
    ).dt.total_seconds() / 3600.0
    return output


def _mimic_anchor_weights(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    import duckdb

    stays = frame[["stay_id"]].drop_duplicates().copy()
    stays["stay_id"] = pd.to_numeric(stays["stay_id"], errors="raise").astype("int64")
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA disable_progress_bar")
    connection.register("target_weight_stays", stays)
    source = data_root / "icu" / "chartevents.csv.gz"
    weights = connection.execute(
        f"""
        SELECT cast(ce.stay_id AS VARCHAR) AS stay_id,
               ce.charttime AS weight_time,
               ce.valuenum AS body_weight_t
        FROM read_csv_auto('{source}') ce
        JOIN target_weight_stays ts ON ce.stay_id = ts.stay_id
        WHERE ce.itemid IN ({','.join(str(value) for value in MIMIC_WEIGHT_ITEMIDS)})
          AND ce.valuenum BETWEEN 25.0 AND 300.0
        """
    ).df()
    connection.close()
    weights["weight_time"] = pd.to_datetime(weights["weight_time"], errors="coerce")
    weights = weights.dropna(subset=["weight_time", "body_weight_t"])

    groups = []
    for stay_id, anchor_rows in frame.groupby("stay_id", sort=False):
        stay_weights = weights.loc[weights["stay_id"].eq(str(stay_id))]
        left = anchor_rows.copy()
        left["_original_index"] = left.index
        left["anchor_time"] = pd.to_datetime(left["anchor_time"], errors="coerce")
        left = left.sort_values("anchor_time")
        if stay_weights.empty:
            left["body_weight_t"] = np.nan
            left["weight_time"] = pd.NaT
        else:
            left = pd.merge_asof(
                left,
                stay_weights.sort_values("weight_time"),
                left_on="anchor_time",
                right_on="weight_time",
                by="stay_id",
                direction="backward",
                tolerance=pd.Timedelta(hours=168),
            )
        groups.append(left)
    output = pd.concat(groups, ignore_index=True, sort=False)
    output["body_weight_age_hr"] = (
        output["anchor_time"] - output["weight_time"]
    ).dt.total_seconds() / 3600.0
    return output.sort_values("_original_index").drop(columns=["_original_index", "weight_time"])


def _add_contract(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    valid_weight = pd.to_numeric(frame["body_weight_t"], errors="coerce").between(25.0, 300.0)
    frame["_observed_body_weight"] = valid_weight.astype(np.float32)
    frame["future_body_weight"] = np.nan
    frame["future_event_time_body_weight"] = pd.NaT
    frame["future_delta_t_hr_body_weight"] = np.nan
    frame["future_label_quality_body_weight"] = "missing"
    frame["future_label_weight_body_weight"] = 0.0
    frame["next_measurement_time_hr_body_weight"] = np.nan

    current_urine = pd.to_numeric(frame["urine_output_t"], errors="coerce")
    future_urine = pd.to_numeric(frame["future_urine_output"], errors="coerce")
    weight = pd.to_numeric(frame["body_weight_t"], errors="coerce")
    frame["urine_output_per_kg_hour_t"] = current_urine / weight
    frame["urine_output_per_kg_hour_age_hr"] = frame["urine_output_age_hr"]
    frame["_observed_urine_output_per_kg_hour"] = (
        frame["_observed_urine_output"].fillna(0.0).astype(float) * valid_weight.astype(float)
    )
    frame["future_urine_output_per_kg_hour"] = future_urine / weight
    for prefix in (
        "future_event_time_",
        "future_delta_t_hr_",
        "future_label_quality_",
        "future_label_weight_",
        "next_measurement_time_hr_",
    ):
        frame[f"{prefix}urine_output_per_kg_hour"] = frame[f"{prefix}urine_output"]
    invalid = ~valid_weight
    frame.loc[invalid, "future_urine_output_per_kg_hour"] = np.nan
    frame.loc[invalid, "future_label_weight_urine_output_per_kg_hour"] = 0.0
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=("eicu", "mimiciv"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=3.0)
    args = parser.parse_args()

    frame = _read_horizon(args.input, args.horizon)
    frame["stay_id"] = frame["stay_id"].astype(str)
    if args.database == "eicu":
        frame = _eicu_anchor_weights(frame, args.data_root)
    else:
        frame = _mimic_anchor_weights(frame, args.data_root)
    frame = _add_contract(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    valid = frame["_observed_body_weight"].eq(1.0)
    report = {
        "schema": "weight_normalized_urine_examples.v1",
        "database": args.database,
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "weight_observed_rows": int(valid.sum()),
        "weight_observed_fraction": float(valid.mean()),
        "normalized_urine_strict_rows": int(
            (
                pd.to_numeric(
                    frame["future_label_weight_urine_output_per_kg_hour"], errors="coerce"
                ).fillna(0.0)
                > 0.0
            ).sum()
        ),
        "future_weight_used": False,
        "causal_claim_allowed": False,
        "output": str(args.output.resolve()),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
