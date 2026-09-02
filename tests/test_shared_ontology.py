"""The demo owns its state vocabulary: it must cover the knowledge files exactly.

The Medication Safety engine used to canonicalise against the research
repository's ontology. It now canonicalises against
``oslian_demo.shared.ontology``, and these tests are what make that safe: they
read the demo's own knowledge files and assert that every variable in them lands
where it did before — resolved to its canonical state, or deliberately left as
demo-local vocabulary. A knowledge file that grows a name this table does not
know about fails here rather than silently changing a ranking.
"""

import json
import unittest

from oslian_demo.config import MEDICATION_DATA_DIR
from oslian_demo.medication import case_targets
from oslian_demo.medication.engine import reasoning_engine
from oslian_demo.shared.ontology import OSLIAN_STATE_ONTOLOGY, StateOntology, StateSpec

# Every variable the engine canonicalises that resolves to a canonical state.
# Frozen deliberately: this is the behaviour the research ontology produced.
CANONICAL_VOCABULARY = {
    "afterload": "cardiovascular.afterload",
    "airflow_resistance": "respiratory.airflow_resistance",
    "bacterial_pathogen": "infection.pathogen_burden",
    "blood_glucose_elevation": "metabolic.glucose",
    "bronchospasm": "respiratory.bronchospasm",
    "clot_propagation_risk": "hemostasis.clot_propagation_risk",
    "coagulation_cascade": "hemostasis.coagulation_activity",
    "fluid_overload": "fluid.overload",
    "heart_rate": "cardiovascular.heart_rate",
    "hypovolemia": "fluid.hypovolemia",
    "hypoxia": "respiratory.hypoxia",
    "infection_load": "infection.load",
    "ketogenesis": "metabolic.ketogenesis",
    "perfusion": "cardiovascular.perfusion",
    "platelet_aggregation": "hemostasis.platelet_aggregation",
    "preload": "cardiovascular.preload",
    "systemic_BP": "cardiovascular.map",
}

# Demo-local vocabulary: named by the knowledge files, deliberately outside the
# canonical state table, and carried through as its own slug.
DEMO_LOCAL_VOCABULARY = {
    "anginal_pain", "angiotensin_II", "cranial_vasodilation", "gas_exchange", "gastric_acid",
    "histamine_release", "inflammation", "ischemic_event_progression_risk", "mucosal_irritation",
    "myocardial_oxygen_demand", "neurogenic_inflammation", "pain_perception", "prostaglandin",
    "pruritus", "pulmonary_edema", "sympathetic_drive", "vasoconstriction",
}


def _knowledge_variables() -> set[str]:
    """Every variable name the engine will ever be handed by the demo's own data."""
    names = set()
    drugs = json.loads((MEDICATION_DATA_DIR / "drugs_pkpd.json").read_text(encoding="utf-8"))["drugs"]
    for drug in drugs.values():
        for effect in drug.get("state_effects", []):
            names.add(effect["variable"])
    for indication in case_targets.INDICATIONS:
        targets, _, _ = case_targets.targets_for(indication)
        for target in targets:
            names.add(target["variable"])
    return names


class SharedOntologyTests(unittest.TestCase):
    def test_the_engine_canonicalises_against_this_repository_only(self):
        source = (reasoning_engine.__file__)
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("from ...shared.ontology import OSLIAN_STATE_ONTOLOGY", text)
        self.assertNotIn("osler_jepa", text)

    def test_the_knowledge_files_name_exactly_the_vocabulary_we_carry(self):
        found = _knowledge_variables()
        expected = set(CANONICAL_VOCABULARY) | DEMO_LOCAL_VOCABULARY
        self.assertEqual(found, expected,
                         "a knowledge file changed its variable names; update the ontology, not this test")

    def test_every_canonical_name_resolves_the_way_it_always_did(self):
        for name, canonical in CANONICAL_VOCABULARY.items():
            self.assertEqual(OSLIAN_STATE_ONTOLOGY.canonicalize(name), canonical, name)

    def test_demo_local_names_stay_demo_local_and_keep_their_slug(self):
        for name in DEMO_LOCAL_VOCABULARY:
            self.assertIsNone(OSLIAN_STATE_ONTOLOGY.canonicalize(name), name)
            self.assertEqual(reasoning_engine._canonical_variable(name),
                             name.strip().lower().replace(" ", "_").replace("-", "_"))

    def test_lookup_ignores_case_spaces_and_hyphens(self):
        for spelling in ("Systemic BP", "systemic-bp", "  SYSTEMIC_BP  "):
            self.assertEqual(OSLIAN_STATE_ONTOLOGY.canonicalize(spelling), "cardiovascular.map", spelling)

    def test_no_canonical_state_is_carried_that_the_demo_never_names(self):
        used = set(CANONICAL_VOCABULARY.values())
        self.assertEqual(set(OSLIAN_STATE_ONTOLOGY.specs), used,
                         "the table carries a state the demo never asks for; keep it minimal")

    def test_an_ambiguous_alias_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            StateOntology("bad", (StateSpec("a.one", "organ", "unit", aliases=("shared",)),
                                  StateSpec("a.two", "organ", "unit", aliases=("shared",))))


if __name__ == "__main__":
    unittest.main()
