"""Aggregate a fixed-protocol source-cohort repair; never select a new model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def failure_breakdown(task):
    if task.get("status") != "evaluated":
        return {"status": "insufficient_support"}
    controls = dict(task["ensemble"]["comparisons"])
    controls["trained_placebo"] = task["vs_trained_placebo"]
    result = {}
    for name, comparison in controls.items():
        units = comparison["careunits"]
        result[name] = {
            "pooled": comparison["all" if name == "trained_placebo" else "paired"],
            "patient_splits_passed": sum(bool(s["pass"]) for s in comparison["patient_splits"]),
            "care_units_passed": sum(bool(u["pass"]) for u in units.values()),
            "care_units_supported": len(units),
            "failed_care_units": {
                name: value for name, value in units.items() if not value["pass"]},
            "significantly_worse_care_units": [
                name for name, value in units.items()
                if value.get("available") and value["ci95"][0] > 0],
        }
    return result


def compare_reports(before, after):
    for key in ("release_binding", "seed_list", "source_split_seed"):
        if before[key] != after[key]:
            raise ValueError(f"unmatched comparison: {key}")
    for key in ("max_patients", "threads", "mimic_current", "nwicu_current"):
        if before["settings"][key] != after["settings"][key]:
            raise ValueError(f"unmatched evaluation setting: {key}")
    for key in ("mimic_current", "nwicu_current"):
        path = after["settings"][key]
        if before["input_hashes"][path] != after["input_hashes"][path]:
            raise ValueError("external evaluation data changed")
    if not before["complete"] or not after["complete"]:
        raise ValueError("cannot compare incomplete runs")
    result = {"schema": "whole_body_jepa.masked_source_repair_comparison.v1",
              "runtime_modified": False, "promotion_allowed": False,
              "causal_claim_allowed": False, "untouched_lockbox_claim": False,
              "source_cohort": after["source_cohort"], "databases": {}}
    for database in ("mimic", "nwicu"):
        tasks = {}
        for target, candidate in after["evaluation"][database].items():
            original = before["evaluation"][database][target]
            if candidate.get("status") != "evaluated":
                tasks[target] = {"status": "insufficient_support"}
                continue
            old, new = original["ensemble"]["mae"], candidate["ensemble"]["mae"]
            tasks[target] = {
                "status": "evaluated", "previous_ensemble_mae": old,
                "balanced_ensemble_mae": new, "delta_mae": new - old,
                "relative_mae_reduction": (old - new) / old,
                "previous_diagnostic_pass": original["diagnostic_pass"],
                "diagnostic_pass": candidate["diagnostic_pass"],
                "failures": failure_breakdown(candidate),
            }
        evaluated = [v for v in tasks.values() if v["status"] == "evaluated"]
        result["databases"][database] = {
            "evaluated": len(evaluated),
            "point_estimate_improved": sum(v["delta_mae"] < 0 for v in evaluated),
            "diagnostic_pass": [t for t, v in tasks.items() if v.get("diagnostic_pass")],
            "tasks": tasks,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_reports(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({d: {k: v for k, v in r.items() if k != "tasks"}
                      for d, r in result["databases"].items()}, indent=2))


if __name__ == "__main__":
    main()
