#!/usr/bin/env python3
"""Rebuild anchor-safe bedside and administered-treatment context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.build_waveform_event_cohort import attach_clinical_context


def run(args: argparse.Namespace) -> dict[str, object]:
    cohort = pd.read_parquet(args.cohort)
    enriched = attach_clinical_context(cohort, args.mimic_root)
    if len(enriched) != len(cohort):
        raise RuntimeError("context enrichment changed row count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    groups = {
        "bedside": [column for column in enriched if column.startswith("ctx_")],
        "treatment_intent": [column for column in enriched if column.startswith("intent_")],
        "observed_intervention": [column for column in enriched if column.startswith("observed_")],
        "administered_treatment": [column for column in enriched if column.startswith("tx_")],
    }
    support = {
        column: int(pd.to_numeric(enriched[column], errors="coerce").notna().sum())
        for columns in groups.values()
        for column in columns
    }
    report = {
        "schema": "whole_body_jepa.waveform_context_enrichment.v1",
        "rows": int(len(enriched)),
        "patients": int(enriched["subject_id"].nunique()),
        "feature_counts": {name: len(columns) for name, columns in groups.items()},
        "nonmissing_support": support,
        "anchor_safe": True,
        "causal_claim_allowed": False,
        "row_level_output_committable": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
