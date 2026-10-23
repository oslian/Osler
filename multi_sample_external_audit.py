"""Pass frequency of external gates over several MIMIC training samples.

Every earlier audit fit each recipe on one 15,000-row MIMIC sample; near the
gates that choice alone moves coverage by about 0.01-0.03.  This script refits
each catalog cell on ``--samples`` different patient samples of the same size
(sample 0 reproduces the original sample) and scores the same external cohort
with the frozen recipe, point gate and both intervals of
``external_value_conditional_interval_audit``.

Predeclared robustness rule (written before any multi-sample result was seen):
a cell is *robust* when it passes the full gate (point gate and the
value-conditional v2 interval) in at least 80% of samples.  Promoted cells that
are not robust go to demotion review; this script changes no registry.

External labels are re-read for scoring only (both databases were already
used: non-independent).  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osler_jepa.recipe_availability import deployable_recipe

from external_value_conditional_interval_audit import (
    verify_protocol,
    GATE_AXES,
    evaluability,
    evaluate_continuous_cell,
    evaluate_event_cell,
    external_cohort,
)
from mimic_to_eicu_49_cell_external_audit import cap_by_subject, normalize_cohort, stable_seed


ROBUST_SHARE = 0.8


def mimic_sample(full: pd.DataFrame, *, target: str, horizon: int, sample: int, rows: int) -> pd.DataFrame:
    """Sample 0 is the original audit sample; others use independent seeds."""

    seed = stable_seed("MIMIC-IV", target, horizon) if sample == 0 else stable_seed("multi_sample", target, horizon, sample)
    return cap_by_subject(full, rows, seed).reset_index(drop=True)


def summarize(results: list[dict]) -> dict:
    full = [result["full"] for result in results]
    coverage = [result["coverage"] for result in results if result["coverage"] is not None]
    share = float(np.mean(full))
    return {
        "samples": len(results),
        "full_pass_share": share,
        "point_pass_share": float(np.mean([result["point"] for result in results])),
        "interval_pass_share": (
            float(np.mean([result["interval"] for result in results if result["interval"] is not None]))
            if any(result["interval"] is not None for result in results) else None
        ),
        "coverage_mean": float(np.mean(coverage)) if coverage else None,
        "coverage_range": [float(min(coverage)), float(max(coverage))] if coverage else None,
        "robust": bool(share >= ROBUST_SHARE),
        "per_sample": results,
    }


LOCKBOX_DATABASES = frozenset({"SICdb", "HiRID", "Emory"})  # Zigong lockbox was scored on 2026-10-18; now development data


def run(args: argparse.Namespace) -> dict:
    if args.database in LOCKBOX_DATABASES:
        if args.protocol is None:
            raise ValueError(f"{args.database} is an untouched lockbox: --protocol is required")
        verify_protocol(args.protocol)
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = sorted(development["cells"], key=lambda row: (int(row["horizon_hours"]), str(row["target"])))
    if args.cells:
        wanted = set(args.cells.split(","))
        contracts = [row for row in contracts if row["cell"] in wanted]
    cells = []
    current_horizon, raw = None, None
    for contract in contracts:
        cell, target = contract["cell"], contract["target"]
        horizon, source = int(contract["horizon_hours"]), contract["source_model"]
        if horizon != current_horizon:
            raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
            current_horizon = horizon
        external, _ = external_cohort(args, target, horizon)
        source, recipe_reason = deployable_recipe(source, external, target)
        reason = evaluability(args.database, external)
        if reason:
            cells.append({"cell": cell, "evaluable": False, "reason": reason})
            continue
        full, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
        results = []
        for sample in range(args.samples):
            mimic = mimic_sample(full, target=target, horizon=horizon, sample=sample, rows=args.max_rows_per_database)
            if contract["task_type"] == "binary_event_probability":
                metrics = evaluate_event_cell(mimic, external, cell=cell, database=args.database)
                results.append({"sample": sample, "point": metrics["point_pass"], "interval": None,
                                "full": metrics["value_conditional_full_pass"], "coverage": None})
            else:
                metrics = evaluate_continuous_cell(
                    mimic, external, cell=cell, target=target, source=source, database=args.database
                )
                results.append({
                    "sample": sample,
                    "point": metrics["point_pass"],
                    "interval": metrics["value_conditional"]["pass"],
                    "full": metrics["value_conditional_full_pass"],
                    "coverage": metrics["value_conditional"]["patient_equal_coverage"],
                    "candidate_mae": metrics["candidate_mae"],
                    "persistence_mae": metrics["persistence_mae"],
                })
        summary = summarize(results)
        cells.append({"cell": cell, "evaluable": True, "source_model": source, **summary})
        print(json.dumps({"cell": cell, "full_pass_share": summary["full_pass_share"],
                          "coverage_range": summary["coverage_range"]}), flush=True)
    output = {
        "schema": "multi_sample_external_audit.v1",
        "external_database": args.database,
        "mimic_template": args.mimic_template,
        "samples": args.samples,
        "rows_per_sample": args.max_rows_per_database,
        "robust_rule": f"full pass in >= {ROBUST_SHARE:.0%} of samples",
        "independent_validation": False,
        "cells": cells,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=sorted(GATE_AXES), required=True)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--mimic-template", required=True)
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_future_label_repaired"))
    parser.add_argument("--external-template", default=None, help="parquet path template with {horizon} for single-site databases")
    parser.add_argument("--max-rows-per-database", type=int, default=15000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cells", default="")
    parser.add_argument("--protocol", type=Path, default=None, help="frozen lockbox protocol; required for lockbox databases (SICdb, HiRID, Emory)")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
