import json
import unittest
from pathlib import Path

from osler_jepa.event_adapters import CANONICAL_UNITS


class RuntimeArtifactIntegrityTests(unittest.TestCase):
    def test_validated_registries_have_complete_canonical_contracts(self):
        root = Path(__file__).resolve().parent
        paths = set(root.glob("*validated_registry*.json"))
        paths.update(root.glob("*hurdle_validated_registry*.json"))
        checked = 0
        for path in sorted(paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "whole_body_promotion_registry.v3":
                continue
            cells = payload.get("validated_target_horizon_cells", [])
            if not cells:
                continue
            checked += 1
            self.assertTrue(payload.get("input_contract_id"), path.name)
            units = payload.get("variable_units", {})
            for cell in cells:
                target = cell.split("@", 1)[0]
                self.assertIn(target, units, f"{path.name}:{cell}")
                if target in CANONICAL_UNITS:
                    self.assertEqual(
                        units[target],
                        CANONICAL_UNITS[target],
                        f"{path.name}:{cell}",
                    )
        self.assertGreater(checked, 0)

    def test_current_catalog_has_executable_cells_only(self):
        root = Path(__file__).resolve().parent
        payload = json.loads(
            (root / "whole_body_runtime_candidate_catalog_v12.json").read_text(
                encoding="utf-8"
            )
        )
        for candidate in payload["candidates"]:
            cells = candidate.get("validated_cells", [])
            if not cells:
                continue
            registry = candidate.get("target_horizon_registry", {})
            self.assertEqual(set(cells), set(registry), candidate["candidate_id"])
            self.assertTrue(candidate.get("input_contract_id"))
            for cell in cells:
                target = cell.split("@", 1)[0]
                declared_unit = candidate["variable_units"].get(target)
                self.assertTrue(
                    declared_unit,
                    f"{candidate['candidate_id']}:{cell}",
                )
                if target in CANONICAL_UNITS:
                    self.assertEqual(
                        declared_unit,
                        CANONICAL_UNITS[target],
                        f"{candidate['candidate_id']}:{cell}",
                    )


if __name__ == "__main__":
    unittest.main()
