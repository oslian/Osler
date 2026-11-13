"""The physiological state vocabulary the Medication Safety engine canonicalises against.

Why this file exists
--------------------
The engine used to import ``osler_jepa.ontology`` from the research repository.
That made a plain Flask request depend on a research checkout, which is exactly
the coupling this demo has to be free of. The vocabulary itself is small and
stable, so it is owned here instead of being reached for across repositories.

What is in it
-------------
Only the states the demo's own knowledge actually names. The engine feeds it
every ``variable`` in ``medication/data/drugs_pkpd.json`` and every treatment
target in ``medication/case_targets.py``; seventeen of those resolve to a
canonical state and the rest are demo-local vocabulary that keeps its own slug.
``tests/test_shared_ontology.py`` pins both halves of that split, so a knowledge
file cannot quietly grow a name this table does not know about.

This is not a trimmed copy of the research ontology's behaviour: there is no
flag derivation, no schema export and no model-facing mapping here, because the
demo calls exactly one method — ``canonicalize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StateSpec:
    canonical: str
    organ: str
    unit: str
    aliases: tuple[str, ...] = ()


class StateOntology:
    """Alias-to-canonical lookup, insensitive to case, spaces and hyphens."""

    def __init__(self, version: str, specs: Iterable[StateSpec]):
        self.version = version
        self.specs = {spec.canonical: spec for spec in specs}
        self._aliases: dict[str, str] = {}
        for spec in self.specs.values():
            for name in (spec.canonical, *spec.aliases):
                alias = self._normalize(name)
                existing = self._aliases.get(alias)
                if existing is not None and existing != spec.canonical:
                    raise ValueError(
                        f"Ontology alias {name!r} maps to both {existing} and {spec.canonical}")
                self._aliases[alias] = spec.canonical

    @staticmethod
    def _normalize(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    def canonicalize(self, name: str) -> str | None:
        return self._aliases.get(self._normalize(name))


OSLIAN_STATE_ONTOLOGY = StateOntology(
    version="demo-1.0.0",
    specs=(
        StateSpec("metabolic.glucose", "pancreas", "mg/dL", aliases=(
            "G", "glucose", "blood_glucose", "blood_glucose_elevation")),
        StateSpec("metabolic.ketogenesis", "liver", "normalized", aliases=("ketogenesis",)),
        StateSpec("cardiovascular.map", "vessel", "mmHg", aliases=(
            "MAP", "map", "systemic_BP", "systemic_bp")),
        StateSpec("cardiovascular.perfusion", "vessel", "normalized", aliases=("perfusion",)),
        StateSpec("cardiovascular.heart_rate", "heart", "bpm", aliases=("heart_rate", "HR")),
        StateSpec("cardiovascular.preload", "heart", "normalized", aliases=("preload",)),
        StateSpec("cardiovascular.afterload", "heart", "normalized", aliases=("afterload",)),
        StateSpec("respiratory.hypoxia", "lung", "normalized", aliases=("hypoxia",)),
        StateSpec("respiratory.airflow_resistance", "lung", "normalized", aliases=(
            "airflow_resistance",)),
        StateSpec("respiratory.bronchospasm", "lung", "normalized", aliases=("bronchospasm",)),
        StateSpec("fluid.hypovolemia", "vessel", "normalized", aliases=("hypovolemia",)),
        StateSpec("fluid.overload", "vessel", "normalized", aliases=("fluid_overload",)),
        StateSpec("hemostasis.platelet_aggregation", "blood", "normalized", aliases=(
            "platelet_aggregation",)),
        StateSpec("hemostasis.clot_propagation_risk", "blood", "normalized", aliases=(
            "clot_propagation_risk",)),
        StateSpec("hemostasis.coagulation_activity", "blood", "normalized", aliases=(
            "coagulation_cascade",)),
        StateSpec("infection.load", "immune", "normalized", aliases=("infection_load",)),
        StateSpec("infection.pathogen_burden", "immune", "normalized", aliases=(
            "bacterial_pathogen",)),
    ),
)
