import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jepa_deployment_form_dev import deploy
from site_adaptive_external_audit import frozen_dtaci_config, frozen_network_config, frozen_point_config


def rows(stays, hospitals, seed):
    rng = np.random.default_rng(seed)
    out = []
    for stay in range(stays):
        level = rng.normal(4.0, 0.5)
        for k in range(6):
            current = level + rng.normal(0, 0.2)
            out.append({"subject_id": f"p{stay}", "stay_id": f"s{stay}", "hospitalid": f"h{stay % hospitals}", "careunit": "MICU",
                        "hours_since_onset": 6.0 * k, "label_available_hr": 6.5, "current": current,
                        "future": current + rng.normal(0, 0.3), "anchor": current + 0.1, "jepa": current + 0.05})
    return pd.DataFrame(out)


class JepaDeploymentFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.point = frozen_point_config(Path("site_adaptive_point_dev_20261016.json"))
        _, cls.dtaci = frozen_dtaci_config(Path("site_adaptive_dtaci_dev_20261017.json"), Path("site_adaptive_dtaci_prior_dev_20261019.json"))
        cls.network = frozen_network_config(Path("site_adaptive_network_dev_20261020.json"), cls.dtaci)
        cls.calibration = rows(80, 1, 0)

    def run_form(self, test):
        return deploy(self.calibration, test, "jepa", "potassium@6h", self.point, self.dtaci, self.network)

    def test_unavailable_outcome_changes_nothing(self):
        for hospitals in (1, 3):
            test = rows(120, hospitals, 1)
            base = self.run_form(test)
            late = test.copy()
            late.loc[5, "label_available_hr"] = 1e9          # never available during the replay
            late_base = self.run_form(late)
            late.loc[5, "future"] += 50.0
            changed = self.run_form(late)
            np.testing.assert_allclose(late_base[0], changed[0])
            np.testing.assert_allclose(late_base[1], changed[1])
            self.assertFalse(np.allclose(base[0], changed[0]))   # an available outcome is used

    def test_warmup_stays_are_not_scored_and_missing_availability_is_refused(self):
        test = rows(120, 1, 2)
        scored = self.run_form(test)[2]
        warm_stays, scored_stays = set(test.loc[~scored, "stay_id"]), set(test.loc[scored, "stay_id"])
        self.assertEqual(len(warm_stays), 30)
        self.assertFalse(warm_stays & scored_stays)
        test.loc[0, "label_available_hr"] = np.nan
        with self.assertRaises(ValueError):
            self.run_form(test)


if __name__ == "__main__":
    unittest.main()
