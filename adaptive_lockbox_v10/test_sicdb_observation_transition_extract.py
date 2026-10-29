import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import sicdb_observation_transition_extract as sicdb


H = 3600


def write_root(folder: Path, labs: list[tuple] | None = None, signals: list[tuple] | None = None) -> Path:
    pd.DataFrame({"ReferenceGlobalID": [2, 3, 4, 5, 2291], "ReferenceValue": ["INIC", "CWIN", "INBD", "INID", "NCIN"]}).to_csv(
        folder / "d_references.csv.gz", index=False)
    pd.DataFrame({"CaseID": [10, 11, 12, 13, 14], "PatientID": [1, 2, 3, 1, 4], "ICUOffset": [H, 0, 0, 0, H],
                  "TimeOfStay": [31 * H, 30 * H, H, 30 * H, 30 * H], "HospitalUnit": [4, 4, 3, 3, 3],
                  "OffsetAfterFirstAdmission": [0, 0, 0, 100 * H, 0]}).to_csv(folder / "cases.csv.gz", index=False)
    # case 10: an ICU ward logged 30 s before admission, then two wards in the admission second (the later id counts);
    # case 14: no ward logged at its admission second -> excluded
    pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 7, 8], "CaseID": [10, 10, 10, 10, 11, 12, 13, 14], "PatientID": 0, "LogState": 0,
                  "Offset": [0, H - 30, H, H, 0, 0, 0, H + 60],
                  "HospitalUnit": [2291, 5, 4, 3, 4, 3, 2, 3]}).to_csv(folder / "unitlog.csv.gz", index=False)
    labs = labs if labs is not None else [
        (10, 367, 2 * H, 1.2), (10, 355, 2 * H, 60.056), (10, 688, 2.5 * H, 7.4), (10, 687, 2.5 * H, 40.0),
        (11, 463, H, 4.1), (10, 471, 0, 1.0)]                                     # the last lab precedes ICU admission
    pd.DataFrame([(k, *row, 0) for k, row in enumerate(labs)],
                 columns=["id", "CaseID", "LaboratoryID", "Offset", "LaboratoryValue", "LaboratoryType"]).to_csv(
        folder / "laboratory.csv.gz", index=False)
    signals = signals if signals is not None else [
        (10, 707, 2 * H, 80.0), (10, 725, 2 * H, 50.0), (10, 703, 2 * H, 70.0), (10, 701, 2 * H, 120.0), (10, 702, 2 * H, 60.0),
        (10, 701, 3 * H, 120.0), (10, 702, 3 * H, 60.0), (10, 707, 0.5 * H, 90.0)]   # the last hour starts before ICU admission
    pd.DataFrame([(k, *row, 60, "") for k, row in enumerate(signals)],
                 columns=["id", "CaseID", "DataID", "Offset", "Val", "cnt", "rawdata"]).to_csv(folder / "data_float_h.csv.gz", index=False)
    return folder


class SicdbExtractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = write_root(Path(self.folder.name))
        self.stays, self.counts = sicdb.load_stays(self.root, 6)

    def tearDown(self):
        self.folder.cleanup()

    def measurements(self, root=None):
        root = root or self.root
        return sicdb.measurements_for(sicdb.con(), root, sicdb.load_stays(root, 6)[0])

    def rewrite(self, **tables):
        folder = Path(self.folder.name) / "variant"
        folder.mkdir(exist_ok=True)
        return write_root(folder, **tables)

    def test_stays_units_and_patient_timeline(self):
        self.assertEqual(self.stays["stay_id"].tolist(), [10, 11, 13])   # 12 is shorter than the horizon; 14 has no admission unit
        self.assertEqual(self.stays["first_careunit"].tolist(), ["CWIN", "INBD", "INIC"])   # unit at ICU admission
        self.assertEqual(self.counts, {"cases": 5, "cases_without_admission_unit": 1, "cases_with_several_admission_units": 1})
        self.assertEqual(self.stays["last_careunit"].tolist(), ["INBD", "INBD", "CWIN"])    # retrospective only
        readmission = self.stays.set_index("stay_id")
        self.assertGreater(readmission.loc[13, "intime"], readmission.loc[10, "outtime"])   # same patient, later case

    def test_labs_are_converted_kept_inside_the_stay_and_derived(self):
        values = self.measurements().set_index("var")["valuenum"]
        self.assertAlmostEqual(values["bun"], 28.0134, places=3)
        self.assertNotIn("phosphate", values.index)
        self.assertAlmostEqual(values["bicarbonate"], 0.0307 * 40 * 10 ** (7.4 - 6.1), places=6)

    def test_hourly_aggregates_end_their_hour_and_must_lie_inside_the_stay(self):
        frame = self.measurements()
        heart = frame[frame["var"] == "heart_rate"]
        self.assertEqual(heart["charttime"].tolist(), [sicdb.BASE_TIME + pd.Timedelta(hours=3)])
        self.assertEqual(frame.loc[frame["var"] == "urine_output", "valuenum"].tolist(), [50.0])

    def test_map_is_the_measured_mean_only(self):
        frame = self.measurements()
        self.assertEqual(frame.loc[frame["var"] == "map", "valuenum"].tolist(), [70.0])      # SBP/DBP pairs give no MAP
        invalid = self.rewrite(signals=[(10, 703, 2 * H, np.inf), (10, 701, 2 * H, 120.0), (10, 702, 2 * H, 60.0)])
        self.assertFalse((self.measurements(invalid)["var"] == "map").any())

    def test_simultaneous_values_do_not_depend_on_row_order(self):
        rows = [(11, 463, H, 6.0), (11, 685, H, 4.0)]
        signals = [(11, 703, H, 70.0), (11, 706, H, 90.0), (11, 707, H, 80.0), (11, 707, H, 100.0)]
        first = self.measurements(self.rewrite(labs=rows, signals=signals))
        second = self.measurements(self.rewrite(labs=rows[::-1], signals=signals[::-1]))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.loc[first["var"] == "potassium", "valuenum"].tolist(), [5.0])
        self.assertEqual(first.loc[first["var"] == "map", "valuenum"].tolist(), [80.0])

    def test_derivations_use_only_validated_inputs(self):
        frame = self.measurements(self.rewrite(labs=[(11, 688, H, 6.7), (11, 687, H, 201.0)]))   # PaCO2 above 200
        self.assertFalse((frame["var"] == "bicarbonate").any())

    def test_build_keeps_every_anchor_with_label_offsets(self):
        frame, report = sicdb.build(self.root, 6, 6)
        self.assertEqual(report["stays"], frame["stay_id"].nunique())
        self.assertTrue({"heart_rate_t", "heart_rate_tp6", "heart_rate_tp6_offset_hr"} <= set(frame.columns))
        self.assertTrue(frame["heart_rate_tp6"].isna().any())                               # unlabelled anchors are kept
        self.assertEqual(set(frame["hospitalid"]), {"SICdb"})

    def test_missing_patient_timeline_fails_closed(self):
        cases = pd.read_csv(self.root / "cases.csv.gz").astype({"OffsetAfterFirstAdmission": float})
        cases.loc[0, "OffsetAfterFirstAdmission"] = np.nan
        cases.to_csv(self.root / "cases.csv.gz", index=False)
        with self.assertRaises(ValueError):
            sicdb.load_stays(self.root, 6)

    def test_every_mapped_variable_is_a_contract_state_variable(self):
        mapped = {var for var, _ in sicdb.LAB_ITEMS.values()} | set(sicdb.SIGNAL_ITEMS.values())
        self.assertFalse(mapped - set(sicdb.contract.STATE_VARS))


if __name__ == "__main__":
    unittest.main()
