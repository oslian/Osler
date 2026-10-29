import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import adaptive_event_weights as weights
from mimic_to_eicu_49_cell_external_audit import normalize_cohort
from site_adaptive_event_dev import static_model


def transitions(n_stays, seed, prefix=""):
    rng = np.random.default_rng(seed)
    rows = []
    for stay in range(n_stays):
        level = rng.normal(95.0, 2.5)
        for k in range(10):
            value = level + rng.normal(0, 1.5)
            rows.append({"subject_id": f"{prefix}{stay}", "stay_id": f"{prefix}{stay}", "first_careunit": "MICU",
                         "t": pd.Timestamp("2100-01-01") + pd.Timedelta(hours=1 + k),
                         "o2sat_t": value, "o2sat_age_hr": 0.5, "heart_rate_t": 85 + rng.normal(0, 10), "heart_rate_age_hr": 0.5,
                         "o2sat_tp1": 0.7 * value + 28.5 + rng.normal(0, 1.5) if rng.random() < 0.9 else np.nan,
                         "o2sat_tp1_offset_hr": rng.uniform(-0.25, 0.25)})
    return pd.DataFrame(rows)


def cohort(raw, database):
    frame, _ = normalize_cohort(raw, database=database, target="hypoxemia_event", horizon=1, max_rows=0)
    return frame


class AdaptiveEventWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = cohort(transitions(400, 0), "MIMIC-IV").sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        cls.train = np.arange(len(cls.frame))

    def test_bundle_reproduces_the_frozen_static_model(self):
        frozen = static_model(self.frame, self.train, self.train, "hypoxemia_event@1h")["probability"]
        bundle = weights.fit_static(self.frame, self.train, seed=weights.stable_seed("hypoxemia_event@1h", "mimic_to_eicu"))
        with tempfile.TemporaryDirectory() as folder:
            joblib.dump(bundle, Path(folder) / "b.joblib")
            loaded = joblib.load(Path(folder) / "b.joblib")
        self.assertGreater(len(frozen), 3000)
        reproduced = weights.predict(loaded, self.frame)[frozen.index.to_numpy()]
        self.assertTrue(weights.reproduces(reproduced, frozen.to_numpy()))
        self.assertTrue(0.0 <= loaded["shrinkage"] <= 1.0)

    def test_site_rows_do_not_change_the_model(self):
        """The scorer fits on MIMIC rows concatenated with the site; the bundle fitted on MIMIC alone predicts the same."""
        site = cohort(transitions(150, 1, prefix="S"), "eICU").drop(columns=["heart_rate_t"], errors="ignore")
        joined = pd.concat([self.frame, site], ignore_index=True, sort=False)
        joined = joined.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        subject = joined["subject_id"].astype(str)
        scored = static_model(joined, np.flatnonzero(subject.str.startswith("MIMIC:")),
                              np.flatnonzero(subject.str.startswith("EICU:")), "hypoxemia_event@1h")["probability"]
        self.assertGreater(len(scored), 1000)
        bundle = weights.fit_static(self.frame, self.train, seed=weights.stable_seed("hypoxemia_event@1h", "mimic_to_eicu"))
        self.assertTrue(weights.reproduces(weights.predict(bundle, joined)[scored.index.to_numpy()], scored.to_numpy()))


if __name__ == "__main__":
    unittest.main()
