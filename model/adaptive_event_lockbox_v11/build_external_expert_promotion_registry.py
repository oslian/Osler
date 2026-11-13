"""Replication registry of the frozen self-adaptive deployment forms (protocols v10 and v11).

Each source is one frozen lockbox protocol with its own deployment method and
task type: v10 scores the continuous cells under ``point_and_dtaci_adaptive``,
v11 the binary event cells under ``online_platt`` (static MIMIC model + online
Platt scaling).  Decision rule of both: a cell robust on both one-shot
lockboxes (SICdb and HiRID) is independently replicated; robust on one =
candidate; otherwise not promoted.  Robust = full pass in at least 80% of five
MIMIC training samples (computed by the scorer).

Each lockbox result is accepted only if its SHA-256 equals the output recorded
in the ledger receipt ``<step>_<db>`` of exactly that protocol.  An entry covers
only its source's frozen form; a model refit on other data or another online
configuration is not covered.  Research runtime only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from osler_jepa.frozen_protocol import sha256, step_record

DATABASES = ("SICdb", "HiRID")
STATUSES = ("independently_replicated", "candidate", "not_promoted")


@dataclass(frozen=True)
class Source:
    protocol: str
    step: str
    method: str
    task_type: str
    results: tuple[str, str]                 # SICdb, HiRID
    decision_records: tuple[str, ...]
    scope_limits: tuple[str, ...]


SOURCES = (
    Source(protocol="adaptive_model_lockbox_protocol_v10_20261027.json", step="continuous",
           method="point_and_dtaci_adaptive", task_type="continuous_value",
           results=("site_adaptive_lockbox_SICdb_v10.json", "site_adaptive_lockbox_HiRID_v10.json"),
           decision_records=("SICDB_LOCKBOX_RESULT_V10_20261027.md", "HIRID_LOCKBOX_RESULT_V10_20261028.md"),
           scope_limits=("HiRID is one department: care-unit robustness is supported by SICdb only",
                         "no glucose cell is replicated (HiRID glucose not evaluable)")),
    Source(protocol="adaptive_event_lockbox_protocol_v11_20261029.json", step="binary",
           method="online_platt", task_type="binary_event_probability",
           results=("site_adaptive_event_lockbox_SICdb_v11.json", "site_adaptive_event_lockbox_HiRID_v11.json"),
           decision_records=("ADAPTIVE_EVENT_LOCKBOX_RESULTS_V11_20261029.md",),
           scope_limits=("retrospective observation-time replay; pooled site-level calibration only",
                         "SICdb 3h care units INBD and INIC outside the calibration tolerances in 6 of 20 unit-samples; "
                         "HiRID has one care unit, so unit-level calibration is untested there",
                         "SICdb and HiRID were already scored under v10 (continuous cells); OPS and the calibration "
                         "tolerances were adopted post hoc on the development databases")),
)


def outcome(cell: dict | None, method: str) -> str:
    if cell is None or not cell.get("evaluable", True):
        return "not_evaluable"
    return "robust" if cell["methods"][method]["robust"] else "failed"


def status(outcomes: list[str]) -> str:
    robust = outcomes.count("robust")
    if robust == len(DATABASES):
        return "independently_replicated"
    return "candidate" if robust else "not_promoted"


def lockbox_cells(protocol: Path, step: str, database: str, result: Path) -> dict[str, dict]:
    recorded = step_record(protocol, f"{step}_{database}")["outputs_sha256"]
    if recorded.get(str(result)) != sha256(result):
        raise RuntimeError(f"{result} is not the scored output recorded in {step}_{database}")
    report = json.loads(result.read_text(encoding="utf-8"))
    if report["external_database"] != database:
        raise RuntimeError(f"{result} scores {report['external_database']}, not {database}")
    return {cell["cell"]: cell for cell in report["cells"]}


def source_entries(source: Source) -> tuple[dict, list[dict]]:
    protocol_path = Path(source.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    results = dict(zip(DATABASES, map(Path, source.results)))
    cells = {database: lockbox_cells(protocol_path, source.step, database, path) for database, path in results.items()}
    names = sorted(set().union(*cells.values()), key=lambda name: (int(name.rsplit("@", 1)[1].rstrip("h")), name))
    entries = []
    for name in names:
        per_db = {database: cells[database].get(name) for database in DATABASES}
        outcomes = [outcome(per_db[database], source.method) for database in DATABASES]
        recipes = {cell.get("recipe") for cell in per_db.values() if cell is not None and cell.get("evaluable", True)}
        target, horizon = name.rsplit("@", 1)
        entries.append({
            "cell": name,
            "target": target,
            "horizon_hours": int(horizon.rstrip("h")),
            "task_type": source.task_type,
            "method": source.method,
            "protocol": source.protocol,
            "status": status(outcomes),
            "recipe": recipes.pop() if len(recipes) == 1 else None,
            **{f"{database.lower()}_outcome": result for database, result in zip(DATABASES, outcomes)},
            **{f"{database.lower()}_full_pass_share": (per_db[database]["methods"][source.method]["full_pass_share"]
                                                     if outcome(per_db[database], source.method) != "not_evaluable" else None)
               for database in DATABASES},
        })
    record = {
        "protocol": source.protocol,
        "protocol_sha256": sha256(protocol_path),
        "method": source.method,
        "task_type": source.task_type,
        "decision_rule": protocol["decision_rule"],
        "deployment_form_arguments": protocol["deployment_form_arguments"],
        "lockbox_results_sha256": {str(path): sha256(path) for path in results.values()},
        "decision_records": list(source.decision_records),
        "scope_limits": list(source.scope_limits),
    }
    return record, entries


def build(args: argparse.Namespace) -> dict:
    sources, entries = [], []
    for source in SOURCES:
        record, cells = source_entries(source)
        sources.append(record)
        entries.extend(cells)
    registry = {
        "schema": "adaptive_replication_registry.v2",
        "approved_on": args.approved_on,
        "sources": sources,
        "training_database": "MIMIC-IV v3.1",
        "data_provenance": "neither lockbox copy is verified against PhysioNet checksums: the HiRID copy is a "
                           "third-party resale (HIRID_LOCKBOX_PROVENANCE_20261028.md) and the SICdb archive "
                           "appears repackaged, source unconfirmed",
        "status_counts": {name: sum(entry["status"] == name for entry in entries) for name in STATUSES},
        "cells": entries,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-on", default="2026-10-29")
    parser.add_argument("--output", type=Path, default=Path("adaptive_replication_registry_20261029.json"))
    print(json.dumps(build(parser.parse_args())["status_counts"]))


if __name__ == "__main__":
    main()
