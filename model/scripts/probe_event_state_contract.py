"""Verify repaired input invariants with synthetic data, not promotion evidence."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from scripts.audit_event_aware_patient_state import replay_features, _treatment_matrix


def replay(currents, subjects=None):
    n = len(currents)
    times = pd.date_range("2020-01-01", periods=n, freq="1h")
    frame = pd.DataFrame(dict(
        subject_id=subjects or ["synthetic"] * n, stay_id=["synthetic"] * n,
        t=times, _time_hours=np.arange(n), careunit=["test"] * n,
        future_event_time_glucose=times + pd.Timedelta(hours=6),
        future_delta_t_hr_glucose=[6.] * n))
    current = np.asarray(currents, dtype=float)
    prepared = SimpleNamespace(frame=frame, current=current, base=current+10, truth=current+20)
    measurements = (current[:, None]/100, np.ones((n, 1)), np.zeros((n, 1)))
    return replay_features(prepared, measurements, np.zeros((n, 10)), 100., reference_center=100., max_age_hours=6.)


def run():
    _, alone = replay([100.])
    records, others = replay([100., 900., 900.], ["synthetic", "other1", "other2"])
    focal = int(np.flatnonzero(records.subject_id.eq("synthetic"))[0])
    repeated, _ = replay([100., 120.])
    treatment = pd.DataFrame(dict(hist_insulin=[1., 1.],
        hist_insulin_evidence_count=[1., 1.], hist_insulin_last_rate_drug_units_per_hour=[1., 10.]))
    encoded = _treatment_matrix(treatment)
    result = {
        "synthetic_only": True,
        "focal_center_alone": float(alone[0, 6]),
        "focal_center_with_others": float(others[focal, 6]),
        "repeat_measurement_flag": float(repeated.iloc[1].new_measurement_fraction),
        "rate_only_encoded_difference": float(np.max(abs(encoded[1]-encoded[0]))),
    }
    assert result["focal_center_with_others"] == result["focal_center_alone"]
    assert result["repeat_measurement_flag"] == 1
    assert result["rate_only_encoded_difference"] > 0
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
