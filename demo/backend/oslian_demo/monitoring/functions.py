"""Clinical-function navigation: the 11 checkpoint modules as explicit, versioned groups.

Loaded from ``functional_group_map_v1.json`` and validated against the mapping
registry (every variable must exist) and the pinned contract (every module id
and its variable membership must match the checkpoint). Organ workspaces use
the organ's primary function to split its variables into primary / related.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import FUNCTION_MAP_PATH
from .anatomy import CANONICAL_ORGANS, load_organ_map
from .mapping import CANONICAL_UNITS



def load_function_map(path: Path = FUNCTION_MAP_PATH, modules: list[str] | None = None) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "oslian.functional-group-map.v1":
        raise ValueError("unsupported functional group map schema")
    seen: set[str] = set()
    covered: set[str] = set()
    for group in document["groups"]:
        gid = group["id"]
        if gid in seen:
            raise ValueError(f"duplicate functional group {gid}")
        seen.add(gid)
        module_vars = set(group["module_variables"])
        if not set(group["primary"]) <= module_vars or not set(group["secondary"]) <= module_vars:
            raise ValueError(f"{gid}: primary/secondary must be module variables")
        if set(group["primary"]) & set(group["secondary"]):
            raise ValueError(f"{gid}: a variable cannot be both primary and secondary")
        if set(group["primary"]) | set(group["secondary"]) != module_vars:
            raise ValueError(f"{gid}: every module variable must be primary or secondary")
        if set(group["related"]) & module_vars:
            raise ValueError(f"{gid}: related variables must come from other modules")
        for variable in module_vars | set(group["related"]):
            if variable not in CANONICAL_UNITS:
                raise ValueError(f"{gid}: unknown variable {variable}")
        for organ in group["organs"]:
            if organ not in CANONICAL_ORGANS:
                raise ValueError(f"{gid}: {organ} is not an anatomical organ")
        covered |= module_vars
    if modules is not None and seen != set(modules):
        raise ValueError(f"functional groups {sorted(seen)} do not match checkpoint modules {sorted(modules)}")
    if covered != set(CANONICAL_UNITS):
        raise ValueError("functional groups must partition the 53 checkpoint variables")
    for organ, gid in document["organ_primary_function"].items():
        if organ not in CANONICAL_ORGANS:
            raise ValueError(f"organ_primary_function: {organ} is not an anatomical organ")
        if gid is not None and gid not in seen:
            raise ValueError(f"organ_primary_function: unknown group {gid}")
    return document


def functions_document(catalog_cells: list[dict[str, Any]], modules: list[str] | None = None) -> dict[str, Any]:
    document = load_function_map(modules=modules)
    organ_map = load_organ_map()
    organ_labels = {o["id"]: o["label"] for o in organ_map["organs"]}
    cells_by_variable: dict[str, list[dict[str, Any]]] = {}
    for cell in catalog_cells:
        variable = "o2sat" if cell["target"] == "hypoxemia_event" else cell["target"]
        cells_by_variable.setdefault(variable, []).append({
            "cell": cell["cell"], "target": cell["target"], "horizon_hours": cell["horizon_hours"],
            "can_move": cell["can_move"], "source_model": cell["source_model"],
        })
    groups = []
    for group in document["groups"]:
        own = list(group["primary"]) + list(group["secondary"])
        groups.append({
            "id": group["id"], "label": group["label"], "module": group["id"],
            "primary": list(group["primary"]), "secondary": list(group["secondary"]),
            "related": list(group["related"]),
            "organs": [{"id": o, "label": organ_labels.get(o, o)} for o in group["organs"]],
            "cells": [c for v in own for c in cells_by_variable.get(v, [])],
            "related_cells": [c for v in group["related"] for c in cells_by_variable.get(v, [])],
        })
    return {
        "schema": document["schema"],
        "basis": document["basis"],
        "groups": groups,
        "organ_primary_function": document["organ_primary_function"],
        "variable_units": dict(CANONICAL_UNITS),
    }
