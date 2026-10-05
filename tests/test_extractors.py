import unittest

from osler_jepa.event_ledger import CanonicalEvent, EventKind, EventLedger, EventQuality
from osler_jepa.urine_output import derive_timestamped_urine_rate


class ExtractorTests(unittest.TestCase):
    def test_extractor_modules_import(self) -> None:
        from extractors import canonical_event_examples  # noqa: F401
        from extractors import canonical_event_ledger_extract  # noqa: F401
        from extractors import eicu_body_system_transition_extract  # noqa: F401
        from extractors import mimiciv_observation_transition_extract  # noqa: F401
        from extractors import nwicu_observation_transition_extract  # noqa: F401

    def test_measurement_anchors_are_real_target_events_in_episode_window(self) -> None:
        import pandas as pd

        from extractors.canonical_event_examples import build_measurement_anchors

        events = pd.DataFrame(
            {
                "patient_id": ["p1"] * 5 + ["p2"],
                "encounter_id": ["s1"] * 5 + ["s2"],
                "event_time": pd.to_datetime([
                    "2026-01-01 00:00", "2026-01-01 05:00",
                    "2026-01-01 06:00", "2026-01-01 06:00",
                    "2026-01-05 12:00", "2026-01-01 00:00",
                ]),
                "variable": ["heart_rate", "creatinine", "creatinine",
                             "creatinine", "creatinine", "heart_rate"],
                "kind": ["measurement"] * 6,
                "quality": ["observed", "observed", "observed", "invalid", "observed", "observed"],
                "source_id": ["a", "b", "c", "d", "e", "f"],
                "hospital_id": ["h1"] * 5 + ["h2"],
                "care_unit": ["icu"] * 6,
                "database": ["test"] * 6,
            }
        )
        anchors = build_measurement_anchors(
            events, "creatinine", lookback_hours=6, maximum_episode_hours=96
        )
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors.loc[0, "encounter_id"], "s1")
        self.assertEqual(anchors.loc[0, "anchor_time"], pd.Timestamp("2026-01-01 06:00"))
        self.assertEqual(float(anchors.loc[0, "hours_since_onset"]), 6.0)

    def test_nwicu_units_are_canonicalized(self) -> None:
        import pandas as pd

        from extractors.nwicu_observation_transition_extract import clean_measurements

        frame = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "stay_id": [10, 10, 10],
                "charttime": ["2026-01-01"] * 3,
                "var": ["temperature_f", "body_weight_ounces", "crp"],
                "valuenum": [98.6, 3527.396, 20.0],
                "valueuom": ["degF", "oz", "mg/L"],
            }
        )
        result = clean_measurements(frame).set_index("var")
        self.assertAlmostEqual(float(result.loc["temperature", "valuenum"]), 37.0, places=4)
        self.assertAlmostEqual(float(result.loc["body_weight", "valuenum"]), 100.0, places=3)
        self.assertAlmostEqual(float(result.loc["crp", "valuenum"]), 2.0, places=4)

    def test_nwicu_map_prefers_same_time_arterial_pair(self) -> None:
        import pandas as pd

        from extractors.nwicu_observation_transition_extract import (
            ARTERIAL_DIASTOLIC,
            ARTERIAL_SYSTOLIC,
            NIBP_DIASTOLIC,
            NIBP_SYSTOLIC,
            _derive_map,
        )

        rows = []
        for itemid, value in (
            (ARTERIAL_SYSTOLIC, 120.0),
            (ARTERIAL_DIASTOLIC, 60.0),
            (NIBP_SYSTOLIC, 150.0),
            (NIBP_DIASTOLIC, 90.0),
        ):
            rows.append(
                {
                    "subject_id": 1,
                    "stay_id": 10,
                    "charttime": "2026-01-01 00:00:00",
                    "itemid": itemid,
                    "valuenum": value,
                }
            )
        result = _derive_map(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result.loc[0, "valuenum"]), 80.0)
        self.assertEqual(result.loc[0, "quality"], "derived")
        self.assertEqual(result.loc[0, "map_source"], "arterial_bp")

    def test_event_ledger_is_as_of(self) -> None:
        ledger = EventLedger(
            [
                CanonicalEvent("1", "a", "2026-01-01 00:00", "glucose", EventKind.MEASUREMENT, 100.0),
                CanonicalEvent("1", "a", "2026-01-01 06:00", "glucose", EventKind.MEASUREMENT, 120.0),
            ]
        )
        view = ledger.anchor_view("a", "2026-01-01 03:00", ["glucose"], max_age_hours=6.0)
        self.assertEqual(view["values"]["glucose"], 100.0)

    def test_interval_urine_rate(self) -> None:
        import pandas as pd

        raw = pd.DataFrame(
            {
                "stay": [1, 1],
                "time": ["2026-01-01 00:00", "2026-01-01 02:00"],
                "volume": [50.0, 100.0],
            }
        )
        result = derive_timestamped_urine_rate(
            raw,
            id_column="stay",
            time_column="time",
            value_column="volume",
        )
        self.assertEqual(float(result.iloc[0]["valuenum"]), 50.0)

    def test_interval_urine_quality_is_canonicalized(self) -> None:
        import pandas as pd

        from osler_jepa.event_adapters import canonicalize_event_frame

        frame = pd.DataFrame(
            [
                {
                    "patient_id": "1",
                    "encounter_id": "10",
                    "event_time": "2026-01-01T00:00:00",
                    "variable": "urine_output",
                    "kind": "measurement",
                    "value": 25.0,
                    "quality": "derived_interval_rate",
                }
            ]
        )
        result = canonicalize_event_frame(frame)
        self.assertEqual(result.loc[0, "quality"], "derived_interval_rate")
        event = CanonicalEvent.from_mapping(result.iloc[0].to_dict())
        self.assertEqual(event.quality, EventQuality.DERIVED_INTERVAL_RATE)

    def test_nat_end_time_becomes_absent(self) -> None:
        import pandas as pd

        event = CanonicalEvent(
            "1", "10", "2026-01-01", "glucose", EventKind.MEASUREMENT,
            value=100.0, end_time=pd.NaT,
        )
        self.assertIsNone(event.end_time)

    def test_unknown_future_label_policy_fails_closed(self) -> None:
        ledger = EventLedger(
            [CanonicalEvent("1", "10", "2026-01-01", "unknown_lab", EventKind.MEASUREMENT, 1.0)]
        )
        with self.assertRaisesRegex(ValueError, "no future-label timing policy"):
            ledger.future_label("10", "unknown_lab", "2026-01-01", 6.0)

    def test_urine_is_weight_normalized_only_from_prior_weight(self) -> None:
        import pandas as pd

        from osler_jepa.event_adapters import (
            _weight_normalized_urine_events,
            canonicalize_event_frame,
        )

        frame = canonicalize_event_frame(
            pd.DataFrame(
                [
                    {
                        "patient_id": "1",
                        "encounter_id": "10",
                        "event_time": "2026-01-01T00:00:00",
                        "variable": "body_weight",
                        "kind": "measurement",
                        "value": 80.0,
                        "source_id": "weight:1",
                    },
                    {
                        "patient_id": "1",
                        "encounter_id": "10",
                        "event_time": "2026-01-01T02:00:00",
                        "variable": "urine_output",
                        "kind": "measurement",
                        "value": 120.0,
                        "source_id": "urine:1",
                    },
                    {
                        "patient_id": "2",
                        "encounter_id": "20",
                        "event_time": "2026-01-01T02:00:00",
                        "variable": "urine_output",
                        "kind": "measurement",
                        "value": 120.0,
                        "source_id": "urine:2",
                    },
                ]
            )
        )
        result = _weight_normalized_urine_events(frame)
        self.assertEqual(result["encounter_id"].tolist(), ["10"])
        self.assertAlmostEqual(float(result.loc[0, "value"]), 1.5)
        self.assertEqual(result.loc[0, "unit"], "mL/kg/hour")
        self.assertEqual(result.loc[0, "quality"], "derived")


if __name__ == "__main__":
    unittest.main()
