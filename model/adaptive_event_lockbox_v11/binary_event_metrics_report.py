"""Brier metrics behind a multi-sample binary event audit.

``multi_sample_external_audit.py`` (frozen by protocol v10) keeps only pass
flags for binary event cells.  This report reruns the same frozen evaluator
(``evaluate_event_cell``) on the same MIMIC samples and external cohort and
keeps the Brier scores, prevalence and paired comparisons.  Pass flags must
equal the audit's; any mismatch raises.  Development databases only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from external_value_conditional_interval_audit import evaluability, evaluate_event_cell, external_cohort
from mimic_to_eicu_49_cell_external_audit import normalize_cohort
from multi_sample_external_audit import mimic_sample

KEEP = ("rows", "subjects", "hospitals", "event_prevalence", "candidate_brier", "persistence_brier",
        "mimic_prevalence_brier", "comparisons", "mimic_calibrated_shrinkage", "point_pass")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="multi_sample_external_audit output to explain")
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_harmonized_20261022"))
    parser.add_argument("--external-template", default=None)
    parser.add_argument("--max-rows-per-database", type=int, default=15000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    args.database = audit["external_database"]
    cells = []
    for entry in audit["cells"]:
        if not entry["evaluable"]:
            cells.append(entry)
            continue
        target, horizon = entry["cell"].rsplit("@", 1)
        horizon = int(horizon.rstrip("h"))
        external, _ = external_cohort(args, target, horizon)
        if evaluability(args.database, external):
            raise RuntimeError(f"{entry['cell']}: external cohort differs from the audit's")
        full, _ = normalize_cohort(pd.read_parquet(audit["mimic_template"].format(horizon=horizon)),
                                   database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
        samples = []
        for flags in entry["per_sample"]:
            mimic = mimic_sample(full, target=target, horizon=horizon, sample=flags["sample"], rows=audit["rows_per_sample"])
            metrics = evaluate_event_cell(mimic, external, cell=entry["cell"], database=args.database)
            if metrics["point_pass"] != flags["point"]:
                raise RuntimeError(f"{entry['cell']} sample {flags['sample']}: pass flag differs from the audit")
            samples.append({"sample": flags["sample"], **{key: metrics[key] for key in KEEP}})
        cells.append({"cell": entry["cell"], "evaluable": True, "robust": entry["robust"], "per_sample": samples})
        print(json.dumps({"cell": entry["cell"], "briers": [round(s["candidate_brier"], 5) for s in samples]}), flush=True)
    args.output.write_text(json.dumps({
        "schema": "binary_event_metrics_report.v1",
        "audit": str(args.audit),
        "external_database": args.database,
        "independent_validation": False,
        "cells": cells,
        "causal_claim_allowed": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
