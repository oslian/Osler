import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from emory_challenge2019_transition_extract import BASE_TIME, COLUMNS, build, read_patient

HEADER = list(COLUMNS) + ["SBP", "DBP", "EtCO2", "BaseExcess", "SaO2", "Calcium", "Age", "Gender", "Unit1", "Unit2",
                          "HospAdmTime", "ICULOS", "SepsisLabel"]


def write_patient(folder: Path, name: str, hours: int, unit1=1.0, unit2=0.0) -> None:
    table = pd.DataFrame(np.nan, index=range(hours), columns=HEADER)
    table["ICULOS"] = np.arange(1, hours + 1)
    table["Unit1"], table["Unit2"] = unit1, unit2
    table["HR"] = 80.0 + np.arange(hours)
    table.loc[0, "FiO2"] = 0.4
    table.loc[0, "Calcium"] = 1.1
    table.loc[1, "Calcium"] = 8.5
    table.to_csv(folder / f"{name}.psv", sep="|", index=False, na_rep="NaN")


class EmoryExtractTests(unittest.TestCase):
    def test_rows_are_timed_at_the_end_of_their_hour_and_units_converted(self):
        with tempfile.TemporaryDirectory() as folder:
            write_patient(Path(folder), "p100001", 30)
            stay, events = read_patient(Path(folder) / "p100001.psv")
        self.assertEqual(stay["first_careunit"], "MICU")
        self.assertEqual(stay["outtime"] - stay["intime"], pd.Timedelta(hours=30))
        first_hr = events[events["var"] == "heart_rate"].sort_values("charttime").iloc[0]
        self.assertEqual(first_hr["charttime"], BASE_TIME + pd.Timedelta(hours=1))
        self.assertEqual(events.loc[events["var"] == "fio2", "valuenum"].item(), 40.0)
        self.assertEqual(events.loc[events["var"] == "ionized_calcium", "valuenum"].item(), 1.1)
        self.assertEqual(events.loc[events["var"] == "calcium", "valuenum"].item(), 8.5)

    def test_bicarbonate_is_calculated_only_where_not_measured(self):
        with tempfile.TemporaryDirectory() as folder:
            write_patient(Path(folder), "p100003", 30)
            table = pd.read_csv(Path(folder) / "p100003.psv", sep="|")
            table.loc[2, ["pH", "PaCO2"]] = [7.40, 40.0]
            table.loc[3, ["pH", "PaCO2", "HCO3"]] = [7.20, 25.0, 11.0]
            table.to_csv(Path(folder) / "p100003.psv", sep="|", index=False, na_rep="NaN")
            _, events = read_patient(Path(folder) / "p100003.psv")
        bicarbonate = events[events["var"] == "bicarbonate"].sort_values("charttime")["valuenum"].to_numpy()
        np.testing.assert_allclose(bicarbonate, [24.5, 11.0], atol=0.3)   # derived at hour 3, measured at hour 4

    def test_build_produces_mimic_contract_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            write_patient(Path(folder), "p100001", 30)
            write_patient(Path(folder), "p100002", 20, unit1=0.0, unit2=1.0)
            frame, report = build(Path(folder), 6.0, 6.0)
        self.assertEqual(set(frame["hospitalid"]), {"Emory"})
        self.assertIn("heart_rate_tp6", frame)
        row = frame.sort_values("t").iloc[0]
        # anchor at ICU admission + 6 h: as-of heart rate is the hour-6 row (85), label the hour-12 row (91)
        self.assertEqual(row["heart_rate_t"], 85.0)
        self.assertEqual(row["heart_rate_tp6"], 91.0)
        self.assertEqual(report["careunits"], {"MICU": 4, "SICU": 2})


if __name__ == "__main__":
    unittest.main()
