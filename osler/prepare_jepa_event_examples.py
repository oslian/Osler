"""Combine canonical event examples from several databases into one harmonized JEPA cohort.

The August eICU canonical examples predate the October measurement
harmonization.  Training the whole-body JEPA on MIMIC-IV and testing on eICU
exposed two problems (``JEPA_EXPERT_ANCHORED_DEPLOYMENT_FORM_FINDINGS``):

- unit mismatch: eICU ionized calcium is mg/dL (median 3.8) while MIMIC-IV is
  mmol/L (median 1.12) -> eICU values / 4.008 (as in the Zigong extractor);
  eICU CRP is mg/dL (median 8.2) while MIMIC-IV is mg/L (median 84) -> x 10;
- variables unsuited to a shared ICU physiology latent: tumour markers (AFP,
  CA-125, CA 19-9; extreme tails, e.g. robust z^2 ~7e8 for AFP, no ICU
  dynamics), lipase (assays differ: medians 151 vs 27 U/L, not convertible),
  BNP (MIMIC-IV's item is NT-proBNP, median 3,665 vs eICU BNP 237: different
  analytes) and PTH (NWICU median 212 vs MIMIC-IV 58 pg/mL: assays differ).

Excluded variables lose their ``future_*`` label columns, so the JEPA neither
predicts nor encodes them (its variable set is read from ``future_*``); their
``*_t`` values stay available to the frozen expert recipes, which use them as
deployed.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


IONIZED_CALCIUM_MGDL_PER_MMOL = 4.008
CRP_MGL_PER_MGDL = 10.0
UNIT_CONVERSIONS = {"eicu": {"ionized_calcium": 1.0 / IONIZED_CALCIUM_MGDL_PER_MMOL, "crp": CRP_MGL_PER_MGDL}}
EXCLUDED_FROM_JEPA = ("afp", "ca_125", "ca_19_9", "lipase", "bnp", "pth")
LABEL_PREFIXES = ("future_", "future_delta_t_hr_", "future_event_time_", "future_label_weight_", "future_label_quality_")


def prepare(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True, sort=False)
    database = frame["database"].astype(str).str.lower()
    conversions = {}
    for db, factors in UNIT_CONVERSIONS.items():
        rows = database == db
        for variable, factor in factors.items():
            for column in (f"{variable}_t", f"future_{variable}"):
                if column in frame:
                    frame.loc[rows, column] = pd.to_numeric(frame.loc[rows, column], errors="coerce") * factor
            conversions[f"{db}:{variable}"] = factor
    dropped = [c for c in frame.columns for v in EXCLUDED_FROM_JEPA for p in LABEL_PREFIXES if c == f"{p}{v}"]
    frame = frame.drop(columns=dropped)
    report = {"rows": int(len(frame)), "by_database": database.value_counts().to_dict(), "unit_conversions": conversions,
              "excluded_from_jepa": list(EXCLUDED_FROM_JEPA), "dropped_label_columns": len(dropped)}
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, report = prepare(args.inputs)
    frame.to_parquet(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
