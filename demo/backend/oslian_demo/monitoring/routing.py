"""Variable -> organ / functional-group reverse index.

The two approved maps (``organ_target_map.json`` and ``functional_group_map_v1.json``)
are authored forward: an organ or a module lists the variables attributed to it.
Every consumer that starts from a *variable* -- the stream event payload, the
event console, the motion renderer -- needs the reverse direction, and none of
them may guess it by name. This module inverts the two maps once and hands out
the resulting index.

Both maps are validated by their own loaders first, so an id that appears here is
already known to be a canonical anatomical organ or a checkpoint module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .anatomy import load_organ_map
from .functions import load_function_map


class VariableRouting:
    """Immutable variable -> (organ ids, functional group ids) index."""

    def __init__(self, organs: dict[str, tuple[str, ...]], functions: dict[str, tuple[str, ...]],
                 systemic: tuple[str, ...]) -> None:
        self._organs = organs
        self._functions = functions
        self._systemic = systemic

    def organs_for(self, variable: str) -> tuple[str, ...]:
        """Anatomical organs this variable is attributed to (empty for systemic-only)."""
        return self._organs.get(variable, ())

    def functions_for(self, variable: str) -> tuple[str, ...]:
        """Checkpoint functional modules this variable belongs to."""
        return self._functions.get(variable, ())

    def is_systemic(self, variable: str) -> bool:
        return variable in self._systemic

    def route(self, variable: str) -> dict[str, Any]:
        return {
            "variable": variable,
            "organ_ids": list(self.organs_for(variable)),
            "functional_group_ids": list(self.functions_for(variable)),
            "systemic": self.is_systemic(variable),
        }


@lru_cache(maxsize=1)
def variable_routing() -> VariableRouting:
    organ_map = load_organ_map()
    function_map = load_function_map()
    organs: dict[str, list[str]] = {}
    for organ in organ_map["organs"]:
        for variable in organ["variables"]:
            organs.setdefault(variable, []).append(organ["id"])
    functions: dict[str, list[str]] = {}
    for group in function_map["groups"]:
        for variable in group["module_variables"]:
            functions.setdefault(variable, []).append(group["id"])
    return VariableRouting(
        {v: tuple(ids) for v, ids in organs.items()},
        {v: tuple(ids) for v, ids in functions.items()},
        tuple(organ_map["systemic"]["variables"]),
    )
