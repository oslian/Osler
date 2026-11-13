#!/usr/bin/env python3
"""Verify checksums, feature width, model load, and one end-to-end example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from whole_body_jepa import load_release, predict_patient


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path("models")
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise ValueError(f"checksum mismatch: {name}")
    assets = load_release(root)
    if len(assets.registry["cells"]) != 41:
        raise ValueError("runtime registry must contain 41 cells")
    replication = json.loads(
        (root / "cross_database_replication_registry.json").read_text()
    )
    if replication["strict_replicated_cells"] != 27:
        raise ValueError("cross-database registry must contain 27 strict cells")
    runtime_cells = {
        cell["cell"] for cell in assets.registry["cells"] if cell["can_move"]
    }
    if runtime_cells != set(replication["strict_cell_names"]):
        raise ValueError("runtime and cross-database strict cells disagree")
    result = predict_patient(
        Path("examples/patient_1.csv").read_text(),
        filename="patient_1.csv",
        model_dir=root,
    )
    if result["summary"]["catalog_cells"] != 41:
        raise ValueError("end-to-end output must contain 41 cells")
    decisions = assets.source_selector["cells"]
    maximum_move_cells = assets.source_selector["summary"]["runtime_move_cells"]
    if result["summary"]["runtime_move_cells"] > maximum_move_cells:
        raise ValueError("patient runtime exceeds the externally validated move set")
    for cell in result["cells"]:
        selected = decisions[cell["cell"]]
        if selected["can_move"] and not cell["can_move"]:
            allowed = {
                "continuous_head_abstained",
                "no_observed_target_value",
                "stale_target_anchor",
                "missing_or_stale_o2sat_history",
            }
            if cell["fallback_reason"] not in allowed:
                raise ValueError("validated cell stopped without an explicit patient-level reason")
        if cell["can_move"] and selected["source_changed"]:
            if cell["source_model"] != selected["source_model"]:
                raise ValueError("runtime ignored the selected source")
            if cell["interval_status"] != "unified_source_disjoint_calibration_90":
                raise ValueError("new source used a different calibrator")
    print(json.dumps({"status": "ok", **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
