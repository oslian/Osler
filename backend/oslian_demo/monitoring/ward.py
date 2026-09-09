"""The ward view: loads the ward fixture and serves one attention order.

Kept apart from `api.py` on purpose. The stream blueprint owns a single
patient's timeline, its ledger and the runtime adapter; this owns an ordering
over many patients and touches none of that. The two share `ranking.py` and
nothing else, and this route never calls the WholeBody runtime -- the order is
computed from observations alone, which is why it needs no model contract and
carries no forecast.

    GET /api/ward?cap=15   ->  oslian.ward-attention-order.v2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from ..config import FIXTURE_DIR
from .ranking import (BAND_DISPLAY_CAP, CUTPOINTS, PatientSnapshot, board_to_dict,
                      rank_ward)

# Creatinine has its own field on the snapshot because it is the score; the rest
# of the banded analytes travel together.
BANDED_LABS = tuple(v for v in CUTPOINTS if v != "creatinine")

WARD_FIXTURE = FIXTURE_DIR / "ward_eicu_demo_v1.json"
WARD_SCHEMA = "oslian.ward-fixture.v1"
MAX_DISPLAY_CAP = 200


def load_ward(path: Path = WARD_FIXTURE) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != WARD_SCHEMA:
        raise ValueError(f"unsupported ward fixture schema: {document.get('schema_version')}")
    if not document.get("retrospective") or not document.get("not_live"):
        raise ValueError("the ward fixture must declare itself retrospective and not live")
    return document


def to_snapshots(document: dict[str, Any]) -> list[PatientSnapshot]:
    snapshots = []
    for patient in document["ward"]["patients"]:
        snapshots.append(PatientSnapshot(
            patient_id=patient["id"],
            display_label=patient["display_name"],
            now_hours=float(patient["now_hours"]),
            creatinine=tuple((float(h), float(v)) for h, v in patient.get("creatinine", ())),
            urine_output=tuple((float(h), float(v)) for h, v in patient.get("urine_output", ())),
            treatments=patient.get("treatments", {}),
            # A fixture built before the weight column existed simply has none,
            # and every such stay gets no rate per kg rather than a guessed one.
            weight_kg=(None if patient.get("weight_kg") is None
                       else float(patient["weight_kg"])),
            labs={v: tuple((float(h), float(x)) for h, x in patient.get(v, ()))
                  for v in BANDED_LABS if patient.get(v)},
        ))
    return snapshots


def build_payload(display_cap: int = BAND_DISPLAY_CAP, path: Path = WARD_FIXTURE) -> dict[str, Any]:
    document = load_ward(path)
    payload = board_to_dict(rank_ward(to_snapshots(document), display_cap=display_cap))
    # Care unit is a property of the ward document, not of the ordering rule, so
    # it is joined on here rather than pushed into the generic snapshot.
    detail = {p["id"]: p for p in document["ward"]["patients"]}
    groups = [b["rows"] for b in payload["bands"]] + [payload["unrankable"]]
    for group in groups:
        for row in group:
            patient = detail.get(row["patientId"], {})
            row["careUnit"] = patient.get("care_unit")
            row["nowHours"] = patient.get("now_hours")
            # The observed series the rank was read off, so the detail pane can
            # chart exactly what the order saw and nothing the order did not.
            row["series"] = {v: patient.get(v, [])
                             for v in ("creatinine", "urine_output") + BANDED_LABS}
            row["treatments"] = {k: bool(v) for k, v in (patient.get("treatments") or {}).items()}
    # The provenance travels with the board, so a reader never has to go looking
    # for what these rows are before deciding how much to trust the order.
    payload["source"] = {
        "label": document["ward"]["label"],
        "retrospective": document["retrospective"],
        "notLive": document["not_live"],
        "note": document["note"],
        "metadata": document["source_metadata"],
        "variableUnits": document["ward"]["variable_units"],
    }
    payload["safetyBoundary"] = document["safety_boundary"]
    return payload


def create_blueprint(path: Path = WARD_FIXTURE) -> Blueprint:
    bp = Blueprint("ward", __name__)

    @bp.get("/api/ward")
    def ward():
        raw = request.args.get("cap", str(BAND_DISPLAY_CAP))
        try:
            cap = int(raw)
        except ValueError:
            return jsonify({"error": "cap must be an integer"}), 400
        if cap < 1 or cap > MAX_DISPLAY_CAP:
            return jsonify({"error": f"cap must be between 1 and {MAX_DISPLAY_CAP}"}), 400
        try:
            return jsonify(build_payload(cap, path))
        except (OSError, ValueError) as exc:
            # Fail closed: no ward document, no board. Never an empty list that
            # reads as "nobody needs looking at".
            return jsonify({"error": f"ward fixture unavailable: {exc}"}), 503

    return bp
