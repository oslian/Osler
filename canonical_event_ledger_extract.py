"""CLI for building a canonical MIMIC-IV/eICU factual event ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osler_jepa.event_adapters import (
    extract_eicu_events,
    extract_mimiciv_events,
    extract_nwicu_events,
    extract_zigong_events,
    write_event_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=("mimiciv", "eicu", "nwicu", "zigong"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--exclude-treatments", action="store_true")
    args = parser.parse_args()

    extractor = {
        "mimiciv": extract_mimiciv_events,
        "eicu": extract_eicu_events,
        "nwicu": extract_nwicu_events,
        "zigong": extract_zigong_events,
    }[args.database]
    events, encounters = extractor(
        args.data_root,
        max_stays=args.max_stays,
        include_treatments=not args.exclude_treatments,
    )
    paths = write_event_dataset(events, args.output_dir)
    report = {
        "schema": "canonical_event_ledger_extract.v1",
        "database": args.database,
        "encounters": int(encounters["stay_id"].nunique()),
        "patients": int(encounters["subject_id"].nunique()),
        "events": int(len(events)),
        "event_kinds": {
            str(key): int(value)
            for key, value in events["kind"].value_counts().sort_index().items()
        },
        "variables": int(events["variable"].nunique()),
        "source_tables": sorted(events["source_table"].dropna().astype(str).unique()),
        "quality": {
            str(key): int(value)
            for key, value in events["quality"].value_counts().sort_index().items()
        },
        "output_files": [str(path.resolve()) for path in paths],
        "row_level_data_in_report": False,
        "provenance_is_audit_only": True,
        "future_treatment_is_excluded": True,
        "causal_claim_allowed": False,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
