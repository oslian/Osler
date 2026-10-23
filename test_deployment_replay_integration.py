import unittest

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import fit_continuous_recipe, normalize_cohort
from osler_jepa.site_adaptive_point import PointCorrectionConfig
from site_adaptive_external_audit import DTACI_METHOD, FROZEN_CONFIG, simulate_sites
from osler_jepa.site_adaptive_conformal import DtaciConfig


def transitions(n_stays, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for stay in range(n_stays):
        level = rng.normal(4.0 + shift, 0.4)
        for k in range(8):
            value = level + rng.normal(0, 0.2)
            rows.append({"subject_id": stay, "stay_id": stay, "first_careunit": ["MICU", "SICU"][stay % 2],
                         "t": pd.Timestamp("2100-01-01") + pd.Timedelta(hours=6 + 6 * k),
                         "potassium_t": value, "potassium_age_hr": 1.0, "sodium_t": 140 + rng.normal(0, 3), "sodium_age_hr": 2.0,
                         "potassium_tp6": (value + rng.normal(0, 0.25)) if rng.random() < 0.8 else np.nan,
                         "potassium_tp6_offset_hr": rng.uniform(-2, 2)})
    return pd.DataFrame(rows)


class DeploymentReplayIntegrationTests(unittest.TestCase):
    def test_replay_runs_scores_only_non_warmup_and_respects_availability(self):
        mimic, _ = normalize_cohort(transitions(300, 0), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        site, _ = normalize_cohort(transitions(120, 1, shift=0.3), database="NWICU", target="potassium", horizon=6, max_rows=0)
        frame = pd.concat([mimic, site], ignore_index=True, sort=False).sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        subject = frame["subject_id"].astype(str)
        fitted = fit_continuous_recipe(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                       np.flatnonzero(subject.str.startswith("NWICU:")), target="potassium",
                                       source="measurement_pure_ridge", seed=0)
        rows = fitted["test_rows"]
        out = simulate_sites(frame, rows, fitted, "potassium", 6, "potassium@6h", 0, PointCorrectionConfig(eta_offset=0.005),
                             DtaciConfig(meta_rate=20.0, initial_log_multiplier=0.15))
        values = out[DTACI_METHOD]
        self.assertEqual(int((~values["scored"]).sum() > 0), 1)
        self.assertTrue(np.all(np.isfinite(values["width"])))
        self.assertTrue(np.all(np.isfinite(values["prediction"])))
        warm_stays = set(frame.iloc[rows[~values["scored"]]]["stay_id"])
        scored_stays = set(frame.iloc[rows[values["scored"]]]["stay_id"])
        self.assertFalse(warm_stays & scored_stays)
        self.assertLessEqual(len(warm_stays), 30)                     # min(200, 25% of 120)

    def test_network_replay_runs_for_multi_hospital_sites(self):
        mimic, _ = normalize_cohort(transitions(300, 4), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        raw = transitions(150, 5, shift=0.3)
        raw["hospitalid"] = raw["stay_id"] % 3
        site, _ = normalize_cohort(raw, database="eICU", target="potassium", horizon=6, max_rows=0)
        frame = pd.concat([mimic, site], ignore_index=True, sort=False).sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        subject = frame["subject_id"].astype(str)
        fitted = fit_continuous_recipe(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                       np.flatnonzero(~subject.str.startswith("MIMIC:")), target="potassium",
                                       source="measurement_pure_ridge", seed=0)
        dtaci = DtaciConfig(meta_rate=20.0, initial_log_multiplier=0.15)
        values = simulate_sites(frame, fitted["test_rows"], fitted, "potassium", 6, "potassium@6h", 0,
                                PointCorrectionConfig(eta_offset=0.005), dtaci, labels=(DTACI_METHOD,),
                                network_config=dtaci)[DTACI_METHOD]
        self.assertGreater(frame.iloc[fitted["test_rows"]]["hospitalid"].nunique(), 1)
        self.assertTrue(np.all(np.isfinite(values["width"])))
        self.assertTrue(values["scored"].any() and (~values["scored"]).any())

    def test_mimic_selection_streams_run_on_the_clock(self):
        from site_adaptive_dtaci_dev import run_variant
        from site_adaptive_point_dev import scenario_streams

        raw = transitions(400, 6)
        raw["first_careunit"] = np.where(raw["stay_id"] % 4 == 0, "Coronary Care Unit (CCU)", "MICU")
        full, _ = normalize_cohort(raw, database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        full = full.assign(first_careunit=full["careunit"])
        subjects = full["subject_id"].unique()
        years = pd.Series(np.where(np.arange(len(subjects)) % 2 == 0, "2008 - 2010", "2017 - 2019"), index=subjects)
        streams = dict(scenario_streams(full, years, cell="potassium@6h", target="potassium", horizon=6,
                                        source="measurement_pure_ridge"))
        self.assertEqual(set(streams), {"S1_time", "S2_cardiac_units"})
        for stream in streams.values():
            self.assertTrue(stream["scored"].any() and (~stream["scored"]).any())
            for variant in (None, 60):
                result = run_variant(stream, variant)
                self.assertTrue(all(np.isfinite(r["patient_equal_coverage"]) for r in result.values()))
        from site_adaptive_interval_dev import run_cell
        intervals = run_cell(full, years, cell="potassium@6h", target="potassium", horizon=6, source="measurement_pure_ridge")
        self.assertTrue(all(0 < r["scored_rows"] < r["stream_rows"] for r in intervals.values()))

    def test_missing_label_offsets_are_refused(self):
        old = transitions(60, 2).drop(columns=["potassium_tp6_offset_hr"])
        mimic, _ = normalize_cohort(transitions(200, 3), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        site, _ = normalize_cohort(old, database="NWICU", target="potassium", horizon=6, max_rows=0)
        frame = pd.concat([mimic, site], ignore_index=True, sort=False).sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        subject = frame["subject_id"].astype(str)
        fitted = fit_continuous_recipe(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                       np.flatnonzero(subject.str.startswith("NWICU:")), target="potassium",
                                       source="measurement_pure_ridge", seed=0)
        with self.assertRaises(ValueError):
            simulate_sites(frame, fitted["test_rows"], fitted, "potassium", 6, "c", 0, PointCorrectionConfig(),
                           DtaciConfig(meta_rate=20.0), labels=(DTACI_METHOD,))


if __name__ == "__main__":
    unittest.main()
