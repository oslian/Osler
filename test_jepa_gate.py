import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jepa_gate import choose_model, gate_seed, required_ablations
from site_adaptive_external_audit import frozen_dtaci_config, frozen_network_config, frozen_point_config


def export(seed, noise):
    """Synthetic export: ``noise`` maps each system to its error scale."""

    rng = np.random.default_rng(seed)
    out = []
    for split, database, patients in (("calibration", "mimiciv", 400), ("test", "nwicu", 160)):
        for patient in range(patients):
            level = rng.normal(4.0, 0.5)
            for k in range(5):
                current = level + rng.normal(0, 0.2)
                future = current + rng.normal(0, 0.3)
                row = {"subject_id": f"{database}:{patient}", "stay_id": f"{database}:{patient}", "hospitalid": "h0",
                       "careunit": "MICU", "database": database, "horizon_hours": 6.0, "hours_since_onset": 6.0 * k,
                       "label_available_hr": 6.5, "target": "potassium", "cell": "potassium@6h", "current": current,
                       "future": future, "split": split}
                row.update({system: future + rng.normal(0, scale) for system, scale in noise.items()})
                out.append(row)
    return pd.DataFrame(out)


SCALES = {"anchor": 0.3, "full": 0.1, "direct_organ": 0.2, "body_mediated": 0.2, "organ_route": 0.25, "system_only": 0.28}


class JepaGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, point = frozen_point_config(Path("site_adaptive_point_dev_20261016.json"))
        _, dtaci = frozen_dtaci_config(Path("site_adaptive_dtaci_dev_20261017.json"), Path("site_adaptive_dtaci_prior_dev_20261019.json"))
        cls.configs = (point, dtaci, frozen_network_config(Path("site_adaptive_network_dev_20261020.json"), dtaci))

    def test_full_model_is_chosen_only_when_both_components_earn_their_place(self):
        rows = export(0, SCALES)
        self.assertEqual(choose_model(rows[rows["split"] == "calibration"].reset_index(drop=True), "c")["chosen"], "full")
        no_body_gain = export(1, {**SCALES, "direct_organ": 0.1})          # body feedback adds nothing
        chosen = choose_model(no_body_gain[no_body_gain["split"] == "calibration"].reset_index(drop=True), "c")["chosen"]
        self.assertEqual(chosen, "direct_organ")

    def test_system_only_never_takes_over_and_required_ablations_include_no_route(self):
        flat = export(2, {system: 0.2 for system in SCALES})
        self.assertIsNone(choose_model(flat[flat["split"] == "calibration"].reset_index(drop=True), "c")["chosen"])
        self.assertEqual(required_ablations("full"), ("body_mediated", "direct_organ", "system_only"))

    def test_seed_gate_scores_the_test_database_in_deployment_form(self):
        path = Path(__file__).with_name("_jepa_gate_test.parquet")
        try:
            export(3, SCALES).to_parquet(path)
            result = gate_seed(path, True, self.configs)["NWICU:potassium@6h"]
            self.assertEqual(result["chosen"], "full")
            self.assertTrue(result["beats"]["anchor"]["pass"])
            self.assertLess(result["scored_rows"], 160 * 5)                 # warm-up stays are not scored
            self.assertFalse(gate_seed(path, False, self.configs)["NWICU:potassium@6h"]["pass"])   # SVD failure fails the seed
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
