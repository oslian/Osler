import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from canonical_event_examples import build_partitioned_examples
from osler_jepa.event_adapters import (
    _assert_eicu_event_time_alignment,
    _measurement_events,
    _treatment_events,
    canonicalize_event_frame,
    eicu_relative_timestamp,
)
from whole_body_joint_jepa import load_joint_event_examples


class EventAdapterTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame(
            {
                "stay_id": [10],
                "subject_id": ["patient-a"],
                "hospitalid": ["hospital-a"],
                "careunit": ["MICU"],
            }
        )

    def test_measurement_adapter_preserves_audit_provenance(self):
        measurements = pd.DataFrame(
            {
                "stay_id": [10],
                "charttime": [pd.Timestamp("2025-01-01 03:00")],
                "var": ["potassium"],
                "valuenum": [4.2],
            }
        )
        result = _measurement_events(
            measurements,
            self.meta,
            database="mimiciv",
            source_table="labevents",
            time_column="charttime",
            time_basis="absolute_charttime",
        )
        row = result.iloc[0]
        self.assertEqual(row["patient_id"], "patient-a")
        self.assertEqual(row["unit"], "mEq/L")
        self.assertEqual(row["hospital_id"], "hospital-a")
        self.assertEqual(row["care_unit"], "MICU")
        self.assertEqual(row["source_table"], "labevents")

    def test_treatment_adapter_keeps_only_observed_timing_and_dose(self):
        evidence = pd.DataFrame(
            {
                "stay_id": [10],
                "starttime": [pd.Timestamp("2025-01-01 01:00")],
                "endtime": [pd.Timestamp("2025-01-01 02:00")],
                "action": ["kcl"],
                "source": ["mimic_inputevents"],
                "dose_observed": [True],
                "amount_like": [20.0],
                "rate_like": [10.0],
                "original_label": ["Potassium chloride"],
            }
        )
        result = _treatment_events(evidence, self.meta, database="mimiciv")
        row = result.iloc[0]
        self.assertEqual(row["kind"], "infusion")
        self.assertAlmostEqual(float(row["dose"]), 20.0)
        self.assertAlmostEqual(float(row["rate"]), 10.0)
        self.assertEqual(row["quality"], "observed")

    def test_eicu_measurement_and_treatment_offsets_share_one_epoch(self):
        measurement_time = eicu_relative_timestamp(pd.Series([60])).iloc[0]
        self.assertEqual(measurement_time, pd.Timestamp("2100-01-01 01:00"))
        measurement = pd.DataFrame({"event_time": [measurement_time]})
        treatment = pd.DataFrame(
            {"event_time": [pd.Timestamp("2100-01-01 00:30")]}
        )
        _assert_eicu_event_time_alignment([measurement, treatment])

    def test_eicu_time_alignment_rejects_century_mismatch(self):
        measurement = pd.DataFrame(
            {"event_time": [pd.Timestamp("1970-01-01 01:00")]}
        )
        treatment = pd.DataFrame(
            {"event_time": [pd.Timestamp("2100-01-01 00:30")]}
        )
        with self.assertRaisesRegex(ValueError, "inconsistent relative-time"):
            _assert_eicu_event_time_alignment([measurement, treatment])

    def test_invalid_identity_is_removed(self):
        result = canonicalize_event_frame(
            pd.DataFrame(
                {
                    "patient_id": ["a", "a"],
                    "encounter_id": ["x", "x"],
                    "event_time": [pd.NaT, pd.Timestamp("2025-01-01")],
                    "variable": ["x", "x"],
                    "kind": ["measurement", "measurement"],
                    "value": [1.0, np.nan],
                }
            )
        )
        self.assertEqual(len(result), 1)

    def test_joint_loader_accepts_canonical_event_examples(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "examples.parquet"
            frame = pd.DataFrame(
                {
                    "subject_id": ["p1", "p1"],
                    "stay_id": ["s1", "s1"],
                    "t": pd.to_datetime(
                        ["2025-01-01 00:00", "2025-01-01 01:00"]
                    ),
                    "_source_horizon": [6.0, 6.0],
                    "database": ["mimiciv", "mimiciv"],
                    "hospitalid": ["h1", "h1"],
                    "careunit": ["MICU", "MICU"],
                    "glucose_t": [220.0, 200.0],
                    "creatinine_t": [1.4, 1.5],
                    "glucose_age_hr": [0.0, 0.0],
                    "creatinine_age_hr": [0.0, 0.0],
                    "_observed_glucose": [1.0, 1.0],
                    "_observed_creatinine": [1.0, 1.0],
                    "future_glucose": [180.0, 170.0],
                    "future_creatinine": [1.5, 1.6],
                    "future_label_weight_glucose": [1.0, 1.0],
                    "future_label_weight_creatinine": [1.0, 1.0],
                    "future_delta_t_hr_glucose": [6.0, 6.0],
                    "future_delta_t_hr_creatinine": [6.0, 6.0],
                }
            )
            frame.to_parquet(path, index=False)
            loaded, variables, modules = load_joint_event_examples(
                path,
                ("sepsis", "aki"),
                horizons=(6,),
                max_stays=None,
                min_modules=1,
            )
            self.assertEqual(set(variables), {"glucose", "creatinine"})
            self.assertTrue(set(modules).issubset({"sepsis", "aki"}))
            self.assertEqual(loaded.attrs["source_contract"], "canonical_event_ledger")
            self.assertIn("careunit", loaded.attrs["physiology_encoder_excludes"])

    def test_partitioned_builder_preserves_future_labels(self):
        frame = canonicalize_event_frame(
            pd.DataFrame(
                [
                    {
                        "patient_id": patient,
                        "encounter_id": stay,
                        "event_time": time,
                        "variable": "glucose",
                        "kind": "measurement",
                        "value": value,
                    }
                    for patient, stay, time, value in (
                        ("p1", "s1", "2020-01-01 00:00", 100.0),
                        ("p1", "s1", "2020-01-01 06:00", 90.0),
                        ("p2", "s2", "2020-01-01 00:00", 110.0),
                        ("p2", "s2", "2020-01-01 06:00", 105.0),
                    )
                ]
            )
        )
        anchors = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "encounter_id": ["s1", "s2"],
                "anchor_time": pd.to_datetime(
                    ["2020-01-01 00:00", "2020-01-01 00:00"]
                ),
            }
        )
        result = build_partitioned_examples(
            frame, anchors, ("glucose",), (6.0,)
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(
            sorted(result["future_glucose"].tolist()),
            [90.0, 105.0],
        )


if __name__ == "__main__":
    unittest.main()
