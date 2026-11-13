"""Write the frozen lockbox protocol v11: the binary event cells with online Platt scaling.

Hashes every project module the event scorer and the addendum tool load (as
``site_adaptive_event_audit.loaded_project_files`` sees them in this process),
the event configuration, the cell catalog and the previous protocol (v10), plus
the MIMIC training files of the event horizons.  The output is written once.
Run from the project folder, only after the user approves the freeze.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import lockbox_addendum  # noqa: F401  (loaded so that its modules are hashed)
from osler_jepa.frozen_protocol import sha256, write_once
from site_adaptive_event_audit import FORM_ARGUMENTS, PROJECT, TASK_TYPE, event_contracts, loaded_project_files

PREVIOUS = "adaptive_model_lockbox_protocol_v10_20261027.json"
DEFAULTS = {"mimic_template": "mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet",
            "development_report": "mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json",
            "event_config": "site_adaptive_event_amendment_20261028.json",
            "samples": 5, "max_rows_per_database": 60000}


def build() -> dict:
    if Path.cwd().resolve() != PROJECT:
        raise RuntimeError(f"run from {PROJECT}")
    contracts = event_contracts(Path(DEFAULTS["development_report"]))
    horizons = sorted({int(row["horizon_hours"]) for row in contracts})
    code = sorted(loaded_project_files() | {DEFAULTS["event_config"], DEFAULTS["development_report"], PREVIOUS})
    return {
        "schema": "adaptive_event_lockbox_protocol.v11",
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": "Do the two hypoxemia event cells, deployed as the frozen static MIMIC model with online Platt scaling, "
                    "pass the event gate on SICdb and HiRID in a retrospective observation-time replay?",
        "databases": ["SICdb", "HiRID"],
        "previous_protocol": PREVIOUS,
        "prior_project_use": {
            "SICdb": "scored once under v10 for continuous cells (44/47 robust, including o2sat@12h); current SpO2 and its "
                     "history were model features there; the 1h/3h future SpO2 columns were never used as labels or viewed",
            "HiRID": "as SICdb (36/38 robust under v10); third-party copy not verifiable against official checksums",
            "claim": "a newly frozen binary-task evaluation on previously used external sites, not untouched databases",
        },
        "scored_task_types": [TASK_TYPE],
        "scored_cells": sorted(row["cell"] for row in contracts),
        "deployment_form": "frozen static hypoxemia model (_fit_hypoxemia, 5 MIMIC samples) + online Platt scaling "
                           "(Gupta & Ramdas 2023; gamma 0.1, rho 100, radius 100, start (1, 0)); each hospital its own "
                           "deployment on the shared deployment clock; warm-up stays not scored",
        "deployment_form_arguments": dict(DEFAULTS),
        "gate": {
            "point": "Brier beats persistence (current SpO2 <= 92) and the OPS-recalibrated MIMIC-prevalence comparator: "
                     "patient-level paired bootstrap upper bound < 0",
            "calibration": "pooled point estimates CITL in [-0.2, 0.2] and slope in [0.8, 1.25] (research tolerances, "
                           "not statistical equivalence); not identifiable or any invalid prediction = fail",
            "robust": "full pass in >= 4 of 5 MIMIC samples",
            "diagnostics_not_gated": "patient-cluster bootstrap intervals; calibration by hospital, care unit and stream phase",
        },
        "primary_endpoint": "robust cells per database",
        "decision_rule": "a cell robust on both SICdb and HiRID is an independently replicated event cell within the claim "
                         "scope; robust on one = candidate.  Any registry change requires the user's approval.",
        "claim_scope": "retrospective observation-time replay (final values, label availability = anchor + h + |offset|; "
                       "SICdb SpO2 is an hourly aggregate; HiRID patient = admission); pooled site-level calibration",
        "post_hoc_decisions": [
            "calibration tolerances fixed after the static calibration check on eICU, NWICU and Emory",
            "online Platt scaling adopted by amendment after the MIMIC-only rule selected static (4 vs 4 streams)",
            "data-free Codex pre-freeze review: all 7 findings fixed (CODEX_REVIEW_EVENT_PREFREEZE_20261029.md)",
        ],
        "development_reference_not_a_prediction": {
            "online_platt_robust": {"eICU": 2, "NWICU": 2, "Emory": 2}, "static_robust": {"eICU": 0, "NWICU": 0, "Emory": 0},
            "results": [f"site_adaptive_event_{db}_20261029.json" for db in ("eICU", "NWICU", "Emory")],
            "diagnostics": "eICU first-third slope 0.75-0.86; about a third of estimable eICU hospitals outside the "
                           "tolerances; single-site databases stable from the first third; care-unit CITL up to "
                           "+/-0.66 (NWICU)",
        },
        "preconditions_before_any_label_is_read": [
            "lockbox_addendum.py --protocol <this file> --database <db> --files <extractor, its test, mapping file, the 1h "
            "and 3h extraction parquets and reports>: the parquets must be byte-identical to the previous protocol's addendum",
        ],
        "evaluation": {"command": "python site_adaptive_event_audit.py --database <SICdb|HiRID> --external-template "
                                  "<db>_observation_transitions_{horizon}h_v1.parquet --protocol <this file> --output "
                                  "site_adaptive_event_lockbox_<db>_v11.json (run from the project folder; every other "
                                  "option at its default)"},
        "one_shot_rule": "each database is scored once (ledger step binary_<db>); the claim cannot be repeated, so a crash "
                         "after the claim ends that database's evaluation unless the user decides otherwise, and is reported",
        "refreeze_rule": "if any hashed file changes before scoring, a new protocol version must be frozen before any label is read",
        "frozen_code_sha256": {name: sha256(Path(name)) for name in code},
        "frozen_training_data_sha256": {DEFAULTS["mimic_template"].format(horizon=h): sha256(Path(DEFAULTS["mimic_template"].format(horizon=h)))
                                        for h in horizons},
        "independent_validation": False,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = build()
    assert set(FORM_ARGUMENTS) == set(protocol["deployment_form_arguments"])
    write_once(args.output).write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "code_files": len(protocol["frozen_code_sha256"])}))


if __name__ == "__main__":
    main()
