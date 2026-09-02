"""Generate the versioned WholeBody release contract fixture from the real release.

Runs the pinned release through the same worker the runtime uses, on the
synthetic patient's full delivered timeline, and records: the pinned commit,
manifest checksums, registry hash, the 41-cell catalog, the 53 variables and
units, and one real input/output sample for contract tests. Weights are never
copied; only metadata and one JSON output are committed.

    python -m oslian_demo.monitoring.build_release_contract_fixture --release-dir <pinned checkout> \
        --commit 5a49b3969d69145efa8ef66d0c6e80d24daef5d2 --write
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from .fhir import now_iso
from .ingest import Ingestor
from .mapping import CANONICAL_UNITS
from .repository import Repository
from .source import MockFhirSource
from .timeline import build_model_input, delivered_timeline
from .wholebody import CONTRACT_FIXTURE, ReleaseWorker, RuntimeConfig

FORMAL_GATE = {
    "rule": ("local AGENTS.md: formal whole-body candidates must be "
             "osler_jepa.adjacent_hierarchy.AdjacentThreeLevelResidualAdapter with canonical organ mapping "
             "(canonical_target_organ_membership) and system->organ->body staged training"),
    "release_satisfies_rule": False,
    "evidence": ("release 2026-08-22 ships a frozen JointWholeBodyJEPAWorldModel with 11 functional modules and "
                 "11 shared target heads; neither the release nor this repository contains "
                 "osler_jepa.adjacent_hierarchy or osler_jepa.organ_schema"),
    "consequence": ("runtime attachment requires explicit approval of the release as a bounded research-demo "
                    "runtime (OSLIAN_WHOLEBODY_RUNTIME=release); default mode keeps it detached"),
}


def _synthetic_timeline(source: MockFhirSource) -> tuple[list[dict], str]:
    repo = Repository(":memory:")
    ingestor = Ingestor(repo, source.source_system)
    ingestor.ingest_bundle(source.initial_bundle(), received_at=now_iso(), event_id="fixture")
    session = {"session_id": "fixture", "patient_id": source.patient_id, "source_system": source.source_system,
               "fixture_version": source.fixture_version, "status": "idle", "speed": 1.0, "epoch": 1, "cursor": 0,
               "sim_elapsed_seconds": 0.0, "last_event_sequence": 0, "created_at": now_iso(), "updated_at": now_iso()}
    repo.create_session(session)
    for entry in source.initial_bundle()["entry"]:
        resource = entry["resource"]
        if resource["resourceType"] == "Observation":
            raw = repo.raw_by_reference(source.source_system, "Observation", resource["id"], resource["meta"]["versionId"])
            repo.record_delivery("fixture", 1, int(raw["raw_id"]), 0, "fixture", now_iso())
    for event in source.events():
        result = ingestor.ingest_resource(event.resource, received_at=now_iso(), event_id=f"fixture:{event.source_sequence}")
        if result.accepted and result.raw_id is not None:
            repo.record_delivery("fixture", 1, result.raw_id, event.source_sequence, "fixture", now_iso())
    timeline = delivered_timeline(repo, "fixture", 1)
    onset = repo.encounter_for_patient(source.patient_id)["period_start"]
    return timeline, onset


def build(release_dir: Path, commit: str, python: str, pythonpath: str | None) -> dict:
    config = RuntimeConfig(mode="release", release_dir=release_dir, python=python, pythonpath=pythonpath)
    worker = ReleaseWorker(config, commit)
    startup = worker.ensure_started()
    if not worker.alive():
        raise SystemExit(f"release validation failed: {worker.error}\n{json.dumps(startup, indent=1)}")
    registry = json.loads((release_dir / "models" / "runtime_registry.json").read_text(encoding="utf-8"))
    contract = json.loads((release_dir / "models" / "FEATURE_CONTRACT.json").read_text(encoding="utf-8"))
    source = MockFhirSource()
    timeline, onset = _synthetic_timeline(source)
    cutoff = max(row["effective_time"] for row in timeline)
    model_input = build_model_input(timeline, cutoff, onset)
    reply = worker.predict(model_input.csv_text)
    worker.stop()
    result = reply["result"]
    units_by_cell = {cell["cell"]: cell["unit"] for cell in result["cells"]}
    catalog_cells = [
        {"cell": c["cell"], "target": c["target"], "horizon_hours": int(c["horizon_hours"]),
         "unit": c.get("target_unit") or units_by_cell.get(c["cell"]), "source_model": c["source_model"],
         "can_move": bool(c["can_move"]), "fallback_reason": c.get("fallback_reason"),
         "source_report": c.get("source_report")}
        for c in registry["cells"]
    ]
    variables = sorted(CANONICAL_UNITS)
    return {
        "schema": "oslian.wholebody-release-contract.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "generated_from": f"WholeBody-JEPA release checkout at {commit} via oslian_demo/monitoring/worker.py",
        "model_repository": "https://github.com/Chun0613-code/WholeBody-JEPA.git",
        "model_commit": commit,
        "release_date": startup["release_date"],
        "manifest": startup["manifest"],
        "registry_sha256": startup["registry_sha256"],
        "scikit_learn_pin": startup.get("scikit_learn"),
        "worker_checks": startup["checks"],
        "formal_architecture_gate": FORMAL_GATE,
        "variables": variables,
        "variable_units": {v: CANONICAL_UNITS[v] for v in variables},
        "modules": ["cardiovascular_perfusion", "respiratory", "renal_urinary", "acid_base_electrolyte",
                    "endocrine_metabolic", "hepatic_synthetic", "gastrointestinal_pancreatic",
                    "hematologic_coagulation", "immune_inflammatory", "cardiac_injury", "musculoskeletal_injury"],
        "forward_fill_max_age_hours": float(contract["asof_contract"]["release_max_age_hours"]),
        "catalog": {
            "schema": registry["schema"],
            "catalog_cells": int(registry["catalog_cells"]),
            "externally_validated_point_cells": int(registry["externally_validated_point_cells"]),
            "fallback_cells": int(registry["fallback_cells"]),
            "causal_claim_allowed": bool(registry["causal_claim_allowed"]),
            "clinical_promotion_allowed": bool(registry["clinical_promotion_allowed"]),
            "cells": catalog_cells,
        },
        "sample": {
            "fixture_version": source.fixture_version,
            "input_cutoff": cutoff,
            "input_hash": model_input.input_hash,
            "input_rows": model_input.rows,
            "input_csv": model_input.csv_text,
            "release_output": result,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--python", default=os.environ.get("OSLIAN_WHOLEBODY_PYTHON") or sys.executable)
    parser.add_argument("--pythonpath", default=os.environ.get("OSLIAN_WHOLEBODY_PYTHONPATH"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    document = build(Path(args.release_dir).resolve(), args.commit, args.python, args.pythonpath)
    if args.write:
        CONTRACT_FIXTURE.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {CONTRACT_FIXTURE}")
    print(json.dumps({"catalog_cells": len(document["catalog"]["cells"]),
                      "sample_summary": document["sample"]["release_output"]["summary"]}))


if __name__ == "__main__":
    main()
