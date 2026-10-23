import unittest

import pandas as pd

from osler_jepa.event_ledger import (
    CanonicalEvent,
    EventKind,
    EventLedger,
    TargetLabelPolicy,
)
from osler_jepa.event_dataset import build_event_examples, ledger_readiness_report


class EventLedgerTests(unittest.TestCase):
    def setUp(self):
        self.anchor = pd.Timestamp("2026-01-01 06:00:00")
        self.ledger = EventLedger(
            [
                CanonicalEvent(
                    patient_id="p1",
                    encounter_id="s1",
                    event_time="2026-01-01 05:00:00",
                    variable="glucose",
                    kind=EventKind.MEASUREMENT,
                    value=10.0,
                    unit="mmol/L",
                    database="mimiciv",
                    source_table="labevents",
                ),
                CanonicalEvent(
                    patient_id="p1",
                    encounter_id="s1",
                    event_time="2026-01-01 05:30:00",
                    variable="insulin",
                    kind=EventKind.INFUSION,
                    rate=2.0,
                    rate_unit="unit/hour",
                    database="mimiciv",
                    source_table="inputevents",
                ),
                CanonicalEvent(
                    patient_id="p1",
                    encounter_id="s1",
                    event_time="2026-01-01 11:30:00",
                    variable="glucose",
                    kind=EventKind.MEASUREMENT,
                    value=120.0,
                    unit="mg/dL",
                    database="mimiciv",
                    source_table="labevents",
                ),
                CanonicalEvent(
                    patient_id="p1",
                    encounter_id="s1",
                    event_time="2026-01-01 12:30:00",
                    variable="insulin",
                    kind=EventKind.INFUSION,
                    rate=1.0,
                    rate_unit="unit/hour",
                    database="mimiciv",
                    source_table="inputevents",
                ),
            ]
        )

    def test_anchor_view_is_causal_and_normalizes_units(self):
        view = self.ledger.anchor_view("s1", self.anchor, ("glucose",))
        self.assertAlmostEqual(view["values"]["glucose"], 180.182, places=3)
        self.assertAlmostEqual(view["measurement_age_hours"]["glucose"], 1.0)
        self.assertEqual(len(view["observed_treatment_context"]), 1)
        self.assertEqual(
            view["observed_treatment_context"][0]["event_time"],
            "2026-01-01T05:30:00",
        )
        self.assertNotIn("hospital_id", view["values"])

    def test_future_label_uses_actual_event_time_and_paired_anchor(self):
        label = self.ledger.future_label(
            "s1",
            "glucose",
            self.anchor,
            6.0,
            policy=TargetLabelPolicy(1.0, 2.0),
        )
        self.assertEqual(label["quality"], "strict")
        self.assertAlmostEqual(label["actual_delta_t_hours"], 5.5)
        self.assertEqual(label["training_weight"], 1.0)
        self.assertTrue(label["eligible_for_physiology_supervision"])

    def test_future_treatment_never_enters_anchor_context(self):
        view = self.ledger.anchor_view("s1", self.anchor, ("glucose",))
        times = {
            item["event_time"] for item in view["observed_treatment_context"]
        }
        self.assertNotIn("2026-01-01T12:30:00", times)

    def test_next_measurement_time_is_separate(self):
        self.assertAlmostEqual(
            self.ledger.next_measurement_time("s1", "glucose", self.anchor),
            5.5,
        )

    def test_event_dataset_emits_jepa_ready_strict_fields(self):
        frame = build_event_examples(
            self.ledger,
            pd.DataFrame(
                {
                    "patient_id": ["p1"],
                    "encounter_id": ["s1"],
                    "anchor_time": [self.anchor],
                    "database": ["mimiciv"],
                }
            ),
            ("glucose",),
            (6.0,),
            policies={"glucose": TargetLabelPolicy(1.0, 2.0)},
        )
        self.assertEqual(len(frame), 1)
        self.assertAlmostEqual(frame.iloc[0]["future_delta_t_hr_glucose"], 5.5)
        self.assertEqual(frame.iloc[0]["future_label_weight_glucose"], 1.0)
        self.assertEqual(frame.iloc[0]["next_measurement_time_hr_glucose"], 5.5)
        self.assertEqual(frame.iloc[0]["hist_insulin_last_rate"], 2.0)
        report = ledger_readiness_report(frame, ("glucose",))
        self.assertEqual(report["targets"]["glucose"]["strict_supervision_rows"], 1)


if __name__ == "__main__":
    unittest.main()
