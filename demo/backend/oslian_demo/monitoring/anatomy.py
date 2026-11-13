"""Organ <-> variable <-> forecast-cell mapping for the anatomy view.

The frontend never guesses by name. It receives this document, which joins the
approved organ map with the pinned 41-cell catalog. Organs are anatomical only
(the local AGENTS.md canonical organ list); the checkpoint's functional modules
are reported separately and are never drawn as organs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import ORGAN_MAP_PATH
from .mapping import CANONICAL_UNITS

CANONICAL_ORGANS = (
    "heart", "blood_vessels", "lungs", "kidneys", "liver", "pancreas", "stomach", "intestines", "thyroid",
    "adrenal_glands", "bone_marrow", "spleen", "brain", "skeletal_muscle",
)
FUNCTIONAL_SYSTEM_WORDS = ("perfusion", "acid_base", "coagulation", "inflammatory", "metabolic", "synthetic")


def load_organ_map(path: Path = ORGAN_MAP_PATH) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "oslian.organ-variable-map.v1":
        raise ValueError("unsupported organ map schema")
    seen = set()
    for organ in document["organs"]:
        organ_id = organ["id"]
        if organ_id not in CANONICAL_ORGANS:
            raise ValueError(f"{organ_id} is not a canonical anatomical organ")
        if any(word in organ_id for word in FUNCTIONAL_SYSTEM_WORDS):
            raise ValueError(f"{organ_id} names a functional system, not an organ")
        if organ_id in seen:
            raise ValueError(f"duplicate organ {organ_id}")
        seen.add(organ_id)
        for variable in organ["variables"]:
            if variable not in CANONICAL_UNITS:
                raise ValueError(f"{organ_id} references unknown variable {variable}")
    for variable in document["systemic"]["variables"]:
        if variable not in CANONICAL_UNITS:
            raise ValueError(f"systemic group references unknown variable {variable}")
    return document


def anatomy_document(catalog_cells: list[dict[str, Any]], path: Path = ORGAN_MAP_PATH) -> dict[str, Any]:
    document = load_organ_map(path)
    cells_by_variable: dict[str, list[dict[str, Any]]] = {}
    for cell in catalog_cells:
        variable = "o2sat" if cell["target"] == "hypoxemia_event" else cell["target"]
        cells_by_variable.setdefault(variable, []).append({
            "cell": cell["cell"], "target": cell["target"], "horizon_hours": cell["horizon_hours"],
            "can_move": cell["can_move"], "source_model": cell["source_model"],
        })
    organs = []
    for organ in document["organs"]:
        variables = list(organ["variables"])
        cells = [c for v in variables for c in cells_by_variable.get(v, [])]
        organs.append({
            "id": organ["id"], "label": organ["label"], "variables": variables,
            "cells": cells, "cell_count": len(cells),
        })
    attributed = {v for organ in document["organs"] for v in organ["variables"]}
    systemic = list(document["systemic"]["variables"])
    unattributed = sorted(v for v in CANONICAL_UNITS if v not in attributed and v not in systemic)
    return {
        "schema": document["schema"],
        "basis": document["basis"],
        "organs": organs,
        "systemic": {"label": document["systemic"]["label"], "variables": systemic,
                     "cells": [c for v in systemic for c in cells_by_variable.get(v, [])]},
        "unattributed_variables": unattributed,
        "variable_units": dict(CANONICAL_UNITS),
    }
