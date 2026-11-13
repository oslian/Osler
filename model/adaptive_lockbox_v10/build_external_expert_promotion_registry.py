"""Replication registry for the frozen self-adaptive deployment form (protocol v10).

Decision rule of ``adaptive_model_lockbox_protocol_v10_20261027.json``: a
continuous cell robust under ``point_and_dtaci_adaptive`` on both one-shot
lockboxes (SICdb and HiRID) is an independently replicated adaptive cell;
robust on one = candidate; otherwise not promoted.  Robust = full pass in at
least 80% of five MIMIC training samples (computed by the scorer).

Each lockbox result is accepted only if its SHA-256 equals the output recorded
in the ledger receipt ``continuous_<db>`` of exactly this protocol.  The entry
is the frozen form (MIMIC-only training, the cell's recipe, online point
correction and DtACI intervals); a recipe refit on other data or another
online configuration is not covered.  Research runtime only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osler_jepa.frozen_protocol import sha256, step_record


METHOD = "point_and_dtaci_adaptive"
DATABASES = ("SICdb", "HiRID")


def outcome(cell: dict | None) -> str:
    if cell is None or not cell.get("evaluable", True):
        return "not_evaluable"
    return "robust" if cell["methods"][METHOD]["robust"] else "failed"


def status(outcomes: list[str]) -> str:
    robust = outcomes.count("robust")
    if robust == len(DATABASES):
        return "independently_replicated"
    return "candidate" if robust else "not_promoted"


def lockbox_cells(protocol: Path, database: str, result: Path) -> dict[str, dict]:
    recorded = step_record(protocol, f"continuous_{database}")["outputs_sha256"]
    if recorded.get(str(result)) != sha256(result):
        raise RuntimeError(f"{result} is not the scored output recorded in continuous_{database}")
    report = json.loads(result.read_text(encoding="utf-8"))
    if report["external_database"] != database:
        raise RuntimeError(f"{result} scores {report['external_database']}, not {database}")
    return {cell["cell"]: cell for cell in report["cells"]}


def build(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    results = dict(zip(DATABASES, (args.sicdb, args.hirid)))
    cells = {database: lockbox_cells(args.protocol, database, path) for database, path in results.items()}
    names = sorted(set().union(*cells.values()), key=lambda name: (int(name.rsplit("@", 1)[1].rstrip("h")), name))
    entries = []
    for name in names:
        per_db = {database: cells[database].get(name) for database in DATABASES}
        outcomes = [outcome(per_db[database]) for database in DATABASES]
        recipes = {cell["recipe"] for cell in per_db.values() if cell is not None and cell.get("evaluable", True)}
        target, horizon = name.rsplit("@", 1)
        entries.append({
            "cell": name,
            "target": target,
            "horizon_hours": int(horizon.rstrip("h")),
            "task_type": "continuous_value",
            "status": status(outcomes),
            "recipe": recipes.pop() if len(recipes) == 1 else None,
            **{f"{database.lower()}_outcome": result for database, result in zip(DATABASES, outcomes)},
            **{f"{database.lower()}_full_pass_share": (per_db[database]["methods"][METHOD]["full_pass_share"]
                                                     if outcome(per_db[database]) != "not_evaluable" else None)
               for database in DATABASES},
        })
    registry = {
        "schema": "adaptive_replication_registry.v1",
        "approved_on": args.approved_on,
        "decision_records": ["SICDB_LOCKBOX_RESULT_V10_20261027.md", "HIRID_LOCKBOX_RESULT_V10_20261028.md"],
        "decision_rule": protocol["decision_rule"],
        "method": METHOD,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "deployment_form_arguments": protocol["deployment_form_arguments"],
        "lockbox_results_sha256": {str(path): sha256(path) for path in results.values()},
        "training_database": "MIMIC-IV v3.1",
        "data_provenance": "neither lockbox copy is verified against PhysioNet checksums: the HiRID copy is a "
                           "third-party resale (HIRID_LOCKBOX_PROVENANCE_20261028.md) and the SICdb archive "
                           "appears repackaged, source unconfirmed",
        "scope_limits": ["HiRID is one department: care-unit robustness is supported by SICdb only",
                         "no glucose cell is replicated (HiRID glucose not evaluable)"],
        "status_counts": {name: sum(entry["status"] == name for entry in entries)
                          for name in ("independently_replicated", "candidate", "not_promoted")},
        "cells": entries,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("adaptive_model_lockbox_protocol_v10_20261027.json"))
    parser.add_argument("--sicdb", type=Path, default=Path("site_adaptive_lockbox_SICdb_v10.json"))
    parser.add_argument("--hirid", type=Path, default=Path("site_adaptive_lockbox_HiRID_v10.json"))
    parser.add_argument("--approved-on", default="2026-10-28")
    parser.add_argument("--output", type=Path, default=Path("adaptive_replication_registry_20261028.json"))
    print(json.dumps(build(parser.parse_args())["status_counts"]))


if __name__ == "__main__":
    main()
