import tempfile
import unittest
from pathlib import Path

import pandas as pd

import hirid_observation_transition_extract as hirid


T0 = pd.Timestamp("2150-01-01 08:00")


def at(hours: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=hours)


def write_root(folder: Path, rows: list[tuple] | None = None) -> Path:
    (folder / "reference").mkdir(parents=True, exist_ok=True)
    parts = folder / "raw_stage" / "observation_tables" / "parquet"
    parts.mkdir(parents=True, exist_ok=True)
    for old in parts.glob("*.parquet"):
        old.unlink()
    pd.DataFrame({"patientid": [1, 2, 3, 4], "admissiontime": [T0, T0, T0, T0], "sex": "M", "age": 60,
                  "discharge_status": "alive"}).to_csv(folder / "reference" / "general_table.csv", index=False)
    hr = [(1, 200, at(h), at(h), 0, None, 80.0) for h in (0.5, 10, 30)] + [(2, 200, at(3), at(3), 0, None, 90.0)]
    hr += [(1, 200, at(40), at(40), 2, None, 85.0)]                 # invalidated: does not extend the stay
    hr += [(1, 200, at(45), at(45), 0, None, -1e6)]                 # implausible: does not extend the stay either
    rows = rows if rows is not None else [
        (1, 20000600, at(8), at(9), 0, "F", 88.42),                  # creatinine 1 mg/dL
        (1, 20005110, at(8), at(9), 0, "F", 5.0),                    # glucose: not mapped (dictionary does not distinguish)
        (1, 20000900, at(8), at(9), 0, "F", 120.0),                  # haemoglobin 12 g/dL
        (1, 20004100, at(8), at(9), 0, "F", 10.0),                   # urea -> BUN 28.013
        (1, 20000500, at(12), at(12.2), 0, "P", 3.0),                # preliminary ...
        (1, 20000500, at(12), at(12.5), 0, "F", 4.0),                # ... replaced by the final result
        (1, 20000500, at(14), at(14.2), 0, "F", 4.0),
        (1, 20000500, at(14), at(15.0), 0, "C", 5.0),                # correction replaces the final result
        (1, 20000400, at(16), at(16), 2, "F", 150.0),                # invalidated
        (1, 20000400, at(17), at(17), 64, "F", 160.0),               # censored "bigger than"
        (1, 20000400, at(18), at(18), 1, "F", 140.0),                # out of range flag only: kept
        (1, 20000400, at(19), at(19), 0, "F", 141.0),
        (1, 20000400, at(19), at(19.5), 2, "F", 142.0),              # newest version invalidated: measurement withdrawn
        (1, 20000300, at(20), at(20), 0, "F", 7.4), (1, 20001200, at(20), at(20), 0, "F", 40.0),
        (1, 600, at(10), at(10), 0, None, 120.0), (1, 620, at(10), at(10), 0, None, 60.0),   # SBP/DBP give no MAP
        (1, 110, at(11), at(11), 0, None, 75.0),
        (1, 20000700, at(-1), at(-1), 0, "F", 9.0),                  # before admission
        (1, 20000110, at(31), at(31), 0, "F", 200.0),                # after the last heart rate
    ]
    columns = ["patientid", "variableid", "datetime", "entertime", "status", "type", "value"]
    frame = pd.DataFrame(hr + rows, columns=columns).assign(stringvalue=None).astype({"type": "string", "stringvalue": "string"})
    frame[frame["patientid"] <= 2].to_parquet(parts / "part-0.parquet", index=False)
    frame[frame["patientid"] > 2].to_parquet(parts / "part-1.parquet", index=False)
    pd.DataFrame({"patientid": [1, 2, 3, 4], "part": [0, 0, 1, 1]}).to_csv(
        parts.parent / "observation_tables_index.csv", index=False)
    return folder


class HiridExtractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = write_root(Path(self.folder.name))

    def tearDown(self):
        self.folder.cleanup()

    def measurements(self) -> pd.DataFrame:
        c = hirid.con()
        stays, _ = hirid.load_stays(c, self.root, 6)
        return hirid.measurements_for(c, self.root, stays)

    def value(self, frame: pd.DataFrame, var: str, hours: float) -> list[float]:
        return frame.loc[(frame["var"] == var) & (frame["charttime"] == at(hours)), "valuenum"].round(4).tolist()

    def test_stays_end_at_the_last_valid_heart_rate(self):
        stays, counts = hirid.load_stays(hirid.con(), self.root, 6)
        self.assertEqual(stays["stay_id"].tolist(), [1])            # 2 is shorter than the horizon; 3, 4 have no heart rate
        self.assertEqual(stays["outtime"].tolist(), [at(30)])
        self.assertEqual(counts, {"patients": 4, "patients_without_valid_heart_rate": 2})
        self.assertEqual(set(stays["first_careunit"]), {"ICU"})

    def test_units_are_converted_to_mimic(self):
        frame = self.measurements()
        self.assertEqual(self.value(frame, "creatinine", 8), [1.0])
        self.assertFalse(frame["var"].str.startswith("glucose").any())
        self.assertEqual(self.value(frame, "hemoglobin", 8), [12.0])
        self.assertEqual(self.value(frame, "bun", 8), [28.013])

    def test_flags_corrections_and_stay_window(self):
        frame = self.measurements()
        self.assertEqual(self.value(frame, "potassium", 12), [4.0])   # final replaces preliminary
        self.assertEqual(self.value(frame, "potassium", 14), [5.0])   # correction replaces final
        self.assertEqual(frame.loc[frame["var"] == "sodium", "valuenum"].tolist(), [140.0])
        self.assertFalse(frame["var"].isin(["wbc", "platelets"]).any())   # outside admission .. last heart rate

    def test_map_is_measured_and_bicarbonate_is_derived_from_validated_inputs(self):
        frame = self.measurements()
        self.assertEqual(frame.loc[frame["var"] == "map", "valuenum"].tolist(), [75.0])
        self.assertAlmostEqual(self.value(frame, "bicarbonate", 20)[0], 0.0307 * 40 * 10 ** (7.4 - 6.1), places=4)

    def test_simultaneous_items_are_averaged_order_invariantly(self):
        rows = [(1, 20000500, at(12), at(12), 0, "F", 6.0), (1, 24000520, at(12), at(12), 0, "F", 4.0)]
        first = self.measurements_of(rows)
        second = self.measurements_of(rows[::-1])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(self.value(first, "potassium", 12), [5.0])

    def measurements_of(self, rows):
        write_root(Path(self.folder.name), rows)
        return self.measurements()

    def test_build_keeps_every_anchor_with_label_offsets(self):
        frame, report = hirid.build(self.root, 6, 6)
        self.assertEqual(report["stays"], 1)
        self.assertTrue({"potassium_t", "potassium_tp6", "potassium_tp6_offset_hr"} <= set(frame.columns))
        self.assertTrue(frame["potassium_tp6"].isna().any())          # unlabelled anchors are kept
        self.assertEqual(set(frame["hospitalid"]), {"HiRID"})

    def test_selectable_rows_assemble_identical_transitions(self):
        dense = [(1, 200, at(h / 30), at(h / 30), 0, None, 60.0 + h % 17) for h in range(0, 30 * 30)]    # 2-min heart rate
        dense += [(1, 20000500, at(h), at(h), 0, "F", 3.5 + h / 10) for h in (5, 6.5, 13, 19, 23.5)]
        write_root(Path(self.folder.name), dense)
        c = hirid.con()
        stays, _ = hirid.load_stays(c, self.root, 6)
        anchors = hirid.contract.build_anchors(stays)
        full = hirid.measurements_for(c, self.root, stays)
        kept = hirid.selectable(full, anchors)
        self.assertLess(len(kept), len(full) / 10)
        for suffix, when, direction, tolerance in (("_t", "t", "backward", 6.0), ("_tp6", "t_plus", "nearest", 2.0)):
            pd.testing.assert_frame_equal(hirid.contract.assemble_state(anchors, full, suffix, when, direction, tolerance),
                                          hirid.contract.assemble_state(anchors, kept, suffix, when, direction, tolerance))

    def test_part_groups_give_the_same_transitions(self):
        dense = [(p, 200, at(h / 2), at(h / 2), 0, None, 60.0 + h % 13) for p in (1, 3) for h in range(0, 60)]
        dense += [(p, 20000500, at(h), at(h), 0, "F", 3.5 + h / 10) for p in (1, 3) for h in (5, 9.5, 13, 19, 23.5)]
        write_root(Path(self.folder.name), dense)
        previous = hirid.PARTS_PER_GROUP
        try:
            hirid.PARTS_PER_GROUP = 1
            grouped, _ = hirid.build(self.root, 6, 6)
            hirid.PARTS_PER_GROUP = 10
            together, _ = hirid.build(self.root, 6, 6)
        finally:
            hirid.PARTS_PER_GROUP = previous
        self.assertEqual(grouped["stay_id"].nunique(), 2)                  # stays 1 and 3 lie in different parts
        pd.testing.assert_frame_equal(grouped, together)

    def test_a_selected_stay_without_anchors_builds_an_empty_frame(self):
        frame, report = hirid.build(self.root, 25, 6)                         # 30 h stay: selected, but no anchor fits
        self.assertEqual((len(frame), report["selected_stays"], report["stays"]), (0, 1, 0))

    def test_version_policy(self):
        rows = [
            (1, 20000500, at(12), at(12), 0, "P", 3.0), (1, 20000500, at(12), at(12), 0, "C", 5.0),
            (1, 20000500, at(12), at(12), 0, "F", 4.0),                   # equal entry time: C > F > P
            (1, 20000400, at(18), at(18), 0, "F", 140.0),
            (1, 20000400, at(18), at(18.5), 0, "F", None),               # newer empty version withdraws
            (1, 20000400, at(19), at(19), 0, "F", 142.0), (1, 20000400, at(19), at(19), 0, "F", 141.0),   # tie: smaller
            (1, 20000500, at(14), at(14), 0, "F", 4.0),
            (1, 20000500, at(14), at(14), 2, "F", 4.0),                   # equally ranked, one flagged: withdrawn
            (1, 24000439, at(15), at(15), 0, "F", 100.0),
            (1, 24000439, at(15), at(15), 0, "F", None),                  # equally ranked, one empty: withdrawn
        ]
        for order in (rows, rows[::-1]):
            frame = self.measurements_of(order)
            self.assertEqual(self.value(frame, "potassium", 12), [5.0])
            self.assertEqual(self.value(frame, "potassium", 14), [])
            self.assertEqual(self.value(frame, "chloride", 15), [])
            self.assertEqual(frame.loc[frame["var"] == "sodium", "valuenum"].tolist(), [141.0])

    def test_the_stay_end_heart_rate_follows_the_version_policy(self):
        for rows in ([(1, 200, at(30), at(31), 2, None, 80.0)],
                     [(1, 200, at(30), at(30), 2, None, 80.0)][::-1] + [(1, 200, at(30), at(30), 0, None, 80.0)]):
            write_root(Path(self.folder.name), rows)
            stays, _ = hirid.load_stays(hirid.con(), self.root, 6)
            self.assertEqual(stays["outtime"].tolist(), [at(10)])

    def test_empty_inputs_keep_the_measurement_columns(self):
        frame = self.measurements()
        stays, _ = hirid.load_stays(hirid.con(), self.root, 6)
        anchors = hirid.contract.build_anchors(stays)
        self.assertEqual(list(hirid.selectable(frame.iloc[0:0], anchors).columns), hirid.MEASUREMENT_COLUMNS)
        self.assertEqual(list(hirid.selectable(frame, anchors.iloc[0:0]).columns), hirid.MEASUREMENT_COLUMNS)
        empty = hirid.measurements_for(hirid.con(), self.root, stays.iloc[0:0])
        self.assertTrue(empty.empty and set(hirid.MEASUREMENT_COLUMNS) <= set(empty.columns))
        frame, report = hirid.build(self.root, 1000, 6)                       # no stay is long enough
        self.assertEqual((len(frame), report["selected_stays"], report["stays"]), (0, 0, 0))

    def test_glucose_cells_are_not_evaluable(self):
        from external_value_conditional_interval_audit import evaluability
        from mimic_to_eicu_49_cell_external_audit import normalize_cohort

        rows = [(1, item, at(h), at(h), 0, "F", 6.0) for item in (24000585, 20005110, 24000523) for h in range(1, 30)]
        write_root(Path(self.folder.name), rows)
        frame, report = hirid.build(self.root, 6, 6)
        self.assertEqual((report["glucose_mapped"], report["pair_counts"]["glucose"], report["pair_counts"]["glucose_poc"]),
                         (False, 0, 0))
        for target in ("glucose", "glucose_poc"):
            self.assertTrue(frame[[f"{target}_t", f"{target}_tp6"]].isna().all().all())
            for view in (frame, frame.drop(columns=[f"{target}_t", f"{target}_tp6"])):
                cohort, _ = normalize_cohort(view, database="HiRID", target=target, horizon=6, max_rows=0)
                self.assertEqual(evaluability("HiRID", cohort), "insufficient_external_support")

    def test_selectable_keeps_exact_tie_and_tolerance_boundary_matches(self):
        anchors = pd.DataFrame({"stay_id": [1, 1, 1], "t": [at(10), at(30), at(50)]})
        anchors["t_plus"] = anchors["t"] + pd.Timedelta(hours=6)
        times = [1, 2, 3, 10,                 # anchor 10: exact match
                 15, 17,                      # label 16: equidistant
                 25, 34, 38,                  # label 36: both ends of the 2 h tolerance
                 44,                          # anchor 50: the 6 h look-back boundary
                 53.9, 58.1, 70]              # label 56: just outside the tolerance
        full = pd.DataFrame({"subject_id": "1", "stay_id": 1, "charttime": [at(h) for h in times], "var": "potassium",
                             "valuenum": [3.0 + i / 10 for i in range(len(times))]})
        kept = hirid.selectable(full, anchors)
        self.assertLess(len(kept), len(full))
        for suffix, when, direction, tolerance in (("_t", "t", "backward", 6.0), ("_tp6", "t_plus", "nearest", 2.0)):
            pd.testing.assert_frame_equal(hirid.contract.assemble_state(anchors, full, suffix, when, direction, tolerance),
                                          hirid.contract.assemble_state(anchors, kept, suffix, when, direction, tolerance))

    def test_every_mapped_variable_is_a_contract_state_variable(self):
        self.assertFalse({var for var, _ in hirid.ITEMS.values()} - set(hirid.contract.STATE_VARS))


if __name__ == "__main__":
    unittest.main()
