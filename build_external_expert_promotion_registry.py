"""Promotion registry for MIMIC expert cells validated on the NWICU lockbox.

Eligibility: full pass in the NWICU one-shot lockbox
(NWICU_EXTERNAL_VALIDATION_PROTOCOL_20261014.md).  Tiers follow the
multi-sample robustness rule approved on 2026-10-16
(MULTI_SAMPLE_ROBUSTNESS_FINDINGS_20261016.md): a cell is robust on a database
when it passes the full gate in at least 80% of five MIMIC training samples.

- A: robust on NWICU and eICU -> point forecast and v2 interval, two databases;
- B: robust on NWICU -> point forecast and v2 interval, general ICU scope;
- C: point gate passes in >= 80% of NWICU samples -> point forecast only;
- otherwise not promoted.

Each entry pins the MIMIC training data hashes used by the lockbox, so a recipe
refit on any other data is not covered.  Research runtime only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROBUST_SHARE = 0.8
TIERS = {
    "A": {
        "point_status": "externally_replicated",
        "interval_status": "calibrated",
        "interval_method": "normalized_bin_scale.v2",
        "scope": "robust on NWICU (general ICU) and eICU (disease-module cohorts)",
    },
    "B": {
        "point_status": "externally_replicated",
        "interval_status": "calibrated_general_icu_only",
        "interval_method": "normalized_bin_scale.v2",
        "scope": "robust on the general ICU population; interval not robust in disease-enriched cohorts (eICU)",
    },
    "C": {
        "point_status": "externally_replicated",
        "interval_status": "uncalibrated",
        "interval_method": None,
        "scope": "point forecast only; lower and upper are null",
    },
}


def cells_by_name(path: Path) -> dict[str, dict]:
    return {cell["cell"]: cell for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]}


def tier(lockbox: dict, nwicu: dict, eicu: dict) -> str | None:
    if not (lockbox["evaluable"] and lockbox["metrics"]["value_conditional_full_pass"]):
        return None
    if not nwicu["evaluable"]:
        return None
    nwicu_robust = nwicu["full_pass_share"] >= ROBUST_SHARE
    eicu_robust = eicu["evaluable"] and eicu["full_pass_share"] >= ROBUST_SHARE
    if nwicu_robust and eicu_robust:
        return "A"
    if nwicu_robust:
        return "B"
    if nwicu["point_pass_share"] >= ROBUST_SHARE:
        return "C"
    return None


def build(args: argparse.Namespace) -> dict:
    lockbox_report = json.loads(args.lockbox.read_text(encoding="utf-8"))
    lockbox, nwicu, eicu = cells_by_name(args.lockbox), cells_by_name(args.nwicu_samples), cells_by_name(args.eicu_samples)
    entries = []
    for name in sorted(lockbox, key=lambda key: (int(lockbox[key]["horizon_hours"]), key)):
        assigned = tier(lockbox[name], nwicu[name], eicu[name])
        if assigned is None:
            continue
        cell = lockbox[name]
        event = "fixed" not in cell["metrics"]
        entries.append({
            "cell": name,
            "target": cell["target"],
            "horizon_hours": int(cell["horizon_hours"]),
            "task_type": "binary_event_probability" if event else "continuous_value",
            "source_model": cell["source_model"],
            "tier": assigned,
            **TIERS[assigned],
            **({"interval_status": "not_applicable", "interval_method": None} if event else {}),
            "nwicu_full_pass_share": nwicu[name]["full_pass_share"],
            "nwicu_point_pass_share": nwicu[name]["point_pass_share"],
            "eicu_full_pass_share": eicu[name]["full_pass_share"] if eicu[name]["evaluable"] else None,
        })
    registry = {
        "schema": "external_expert_promotion_registry.v2",
        "approved_on": "2026-10-16",
        "decision_records": ["PROMOTION_REVIEW_NWICU_20261015.md", "MULTI_SAMPLE_ROBUSTNESS_FINDINGS_20261016.md"],
        "eligibility": "full pass in the NWICU one-shot lockbox",
        "robust_rule": f"full pass in >= {ROBUST_SHARE:.0%} of {nwicu[next(iter(nwicu))].get('samples', 5)} MIMIC training samples",
        "lockbox_protocol": lockbox_report["protocol"],
        "lockbox_report": str(args.lockbox),
        "multi_sample_reports": {"NWICU": str(args.nwicu_samples), "eICU": str(args.eicu_samples)},
        "training_database": "MIMIC-IV v3.1",
        "pinned_mimic_training_data_sha256": lockbox_report["mimic_data_sha256"],
        "recipes_refit_on_other_data_are_not_promoted": True,
        "tier_counts": {name: sum(entry["tier"] == name for entry in entries) for name in TIERS},
        "cells": entries,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockbox", type=Path, default=Path("external_NWICU_value_conditional_interval_49cell_20261014.json"))
    parser.add_argument("--nwicu-samples", type=Path, default=Path("multi_sample_NWICU_promotion_version_20261016.json"))
    parser.add_argument("--eicu-samples", type=Path, default=Path("multi_sample_eICU_promotion_version_20261016.json"))
    parser.add_argument("--output", type=Path, default=Path("external_expert_promotion_registry_20261016.json"))
    print(json.dumps(build(parser.parse_args())["tier_counts"]))


if __name__ == "__main__":
    main()
