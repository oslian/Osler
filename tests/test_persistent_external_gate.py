import unittest

import pandas as pd

from scripts.validate_persistent_physiology_external import (
    _apply_target_delta_contract,
    _split_development_residual_gate,
)


class PersistentExternalGateTests(unittest.TestCase):
    def test_residual_gate_split_is_patient_disjoint_and_complete(self) -> None:
        frame = pd.DataFrame(
            {
                "subject_id": [
                    f"patient-{patient}" for patient in range(20) for _ in range(3)
                ],
                "value": range(60),
            }
        )
        rows = pd.RangeIndex(len(frame)).to_numpy()

        fit_rows, gate_rows = _split_development_residual_gate(frame, rows, seed=7)

        fit_patients = set(frame.iloc[fit_rows]["subject_id"])
        gate_patients = set(frame.iloc[gate_rows]["subject_id"])
        self.assertFalse(fit_patients & gate_patients)
        self.assertEqual(set(fit_rows) | set(gate_rows), set(rows))
        self.assertEqual(len(gate_patients), 4)

    def test_delta_contract_counts_only_effective_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "future_glucose": [110.0, 120.0, 130.0, 140.0],
                "future_delta_t_hr_glucose": [3.0, 3.0, 8.0, 8.0],
                "future_label_weight_glucose": [1.0, 0.0, 1.0, 0.0],
            }
        )

        contracted, report = _apply_target_delta_contract(frame, "glucose")

        self.assertEqual(report["labels_before"], 2)
        self.assertEqual(report["labels_retained"], 1)
        self.assertEqual(report["labels_rejected"], 1)
        self.assertEqual(report["future_values_already_zero_weight"], 2)
        self.assertTrue(pd.isna(contracted.loc[2, "future_glucose"]))
        self.assertTrue(pd.isna(contracted.loc[3, "future_glucose"]))
        self.assertEqual(contracted.loc[2, "future_label_weight_glucose"], 0.0)


if __name__ == "__main__":
    unittest.main()
