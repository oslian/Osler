"""Compile only point-qualified, interval-rescued cells into the unified selector."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audit_unified_sources import findings, summarize
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.source_selection import load_selector


def build_selector(parent: dict, report: dict) -> dict:
    if report["release_binding"] != parent["release_binding"]:
        raise ValueError("rescue and selector model bindings disagree")
    output = copy.deepcopy(parent)
    for row in output["cells"].values():
        row.setdefault("interval_method", "constant")
    promoted = []
    for cell in report["passed_cells"]:
        row, result = output["cells"][cell], report["cells"][cell]
        source = result["selected_source"]
        if row["can_move"] or source != row["selected_source"]:
            raise ValueError("rescue cannot change the development-selected source")
        if not row["selection"]["scores"][source]["eligible_challenger"]:
            raise ValueError("source did not pass development selection")
        if not result["pass"] or result["failure_reasons"]:
            raise ValueError("rescue did not pass")
        for db in ("mimic_test", "nwicu"):
            if not row["validation"][db]["point"]["pass"]:
                raise ValueError("parent point gate did not pass")
            if not all(result[db][key]["pass"] for key in ("point", "coverage", "careunit_coverage", "late_coverage")):
                raise ValueError("incomplete rescued validation")
        row.update({"source_model": source, "can_move": True, "source_changed": True,
                    "use_selector_interval": True, "interval_method": result["method"],
                    "radius": None, "normalized_radius": result["normalized_radius"], "scale_model": result["scale_model"],
                    "validation": {"promotion_allowed": True, "failure_reasons": [],
                                   "method": result["method"], "mimic_test": result["mimic_test"], "nwicu": result["nwicu"]}})
        promoted.append(cell)
    output["interval_rescue"] = {"parent_selector_sha256": report["parent_selector_sha256"],
                                 "promoted_cells": promoted, "point_weights_changed": False,
                                 "new_untouched_lockbox_claim": False}
    output["summary"] = summarize(output["cells"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/persistence_interval_rescue_20260902.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/unified_source_selector_pre_interval_rescue_20260902.json"))
    parser.add_argument("--findings", type=Path, default=Path("UNIFIED_SOURCE_SELECTION_FINDINGS_20260902.md"))
    args = parser.parse_args()
    path = args.model_dir / "unified_source_selector.json"
    parent = json.loads(path.read_text())
    report = json.loads(args.report.read_text())
    if sha256(path) != report["parent_selector_sha256"]:
        raise ValueError("rescue report belongs to a different parent selector")
    output = build_selector(parent, report)
    output["interval_rescue"]["report_sha256"] = sha256(args.report)
    serialized = json.dumps(output, indent=2, allow_nan=False)
    with TemporaryDirectory() as temporary:
        probe = Path(temporary) / "selector.json"
        probe.write_text(serialized)
        load_selector(probe, args.model_dir)
    if args.snapshot.exists() and sha256(args.snapshot) != sha256(path):
        raise ValueError("refusing to overwrite a different evidence snapshot")
    args.snapshot.write_bytes(path.read_bytes())
    path.write_text(serialized)
    manifest_path = args.model_dir / "MODEL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest["runtime_move_cells"] = output["summary"]["runtime_move_cells"]
    manifest["fallback_cells"] = sum(not r["can_move"] for r in output["cells"].values())
    manifest["multisource_shared_head_cells"] = output["summary"]["source_counts"].get("multisource_shared_jepa_target_head", 0)
    manifest["new_source_audit_pass_cells"] = output["summary"]["new_audit_pass_cells"]
    original_intervals = json.loads((args.model_dir / "interval_calibrator.json").read_text())
    manifest["runtime_interval_cells"] = sum(
        r["can_move"] and r["target"] != "hypoxemia_event" and (
            r["use_selector_interval"] or (not r["source_changed"] and original_intervals["cells"].get(c, {}).get("runtime_eligible", False))
        ) for c, r in output["cells"].items()
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    findings(output, args.findings)
    print(json.dumps({"promoted_cells": output["interval_rescue"]["promoted_cells"], **output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
