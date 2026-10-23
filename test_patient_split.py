import unittest

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import fit_continuous_recipe, normalize_cohort
from osler_jepa.patient_split import selection_patients
from target_specific_domain_conformal_audit import _split_fit_calibration


def transitions(n_stays, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for stay in range(n_stays):
        level = rng.normal(4.0, 0.5)
        for k in range(8):
            value = level + rng.normal(0, 0.3)
            jump = rng.normal(0, 0.8) if rng.random() < 0.3 else 0.0
            rows.append({"subject_id": stay, "stay_id": stay, "first_careunit": ["MICU", "SICU"][stay % 2],
                         "t": pd.Timestamp("2100-01-01") + pd.Timedelta(hours=6 + 6 * k), "potassium_t": value,
                         "potassium_age_hr": 1.0, "sodium_t": 140 + rng.normal(0, 3), "sodium_age_hr": 2.0,
                         "potassium_tp6": value + jump + rng.normal(0, 0.1), "potassium_tp6_offset_hr": 0.5})
    return pd.DataFrame(rows)


class PatientSplitTests(unittest.TestCase):
    def test_split_is_by_patient_fixed_and_balanced(self):
        subjects = np.repeat([f"MIMIC:{i}" for i in range(400)], 3)
        mask = selection_patients(subjects)
        for subject in np.unique(subjects):
            self.assertEqual(len(set(mask[subjects == subject])), 1)       # never splits a patient
        np.testing.assert_array_equal(mask, selection_patients(subjects[::-1])[::-1])  # order-free
        self.assertAlmostEqual(mask.mean(), 0.5, delta=0.08)

    def test_conformal_patients_do_not_influence_the_hurdle_choice(self):
        mimic, _ = normalize_cohort(transitions(500, 0), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        site, _ = normalize_cohort(transitions(80, 1), database="NWICU", target="potassium", horizon=6, max_rows=0)
        frame = pd.concat([mimic, site], ignore_index=True, sort=False).sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        subject = frame["subject_id"].astype(str)
        train, test = np.flatnonzero(subject.str.startswith("MIMIC:")), np.flatnonzero(subject.str.startswith("NWICU:"))
        fit = lambda f: fit_continuous_recipe(f, train, test, target="potassium", source="target_specific_change_hurdle", seed=3)
        base = fit(frame)
        _, held_out = _split_fit_calibration(frame, train, 3)
        conformal = held_out[~selection_patients(frame.iloc[held_out]["subject_id"].to_numpy())]
        perturbed = frame.copy()
        perturbed.loc[conformal, "future_potassium"] += np.random.default_rng(9).normal(0, 3, len(conformal))
        moved = fit(perturbed)
        self.assertEqual(base["mimic_selected_policy"], moved["mimic_selected_policy"])
        np.testing.assert_allclose(base["test_prediction"], moved["test_prediction"])


if __name__ == "__main__":
    unittest.main()
