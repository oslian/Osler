"""VitalDB-to-WholeBody-JEPA lockbox schema helpers."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_TRACKS = {
    "heart_rate": "Solar8000/HR",
    "respiratory_rate": "Solar8000/RR_CO2",
    "o2sat": "Solar8000/PLETH_SPO2",
    "map": "Solar8000/ART_MBP",
    "sbp": "Solar8000/ART_SBP",
    "dbp": "Solar8000/ART_DBP",
    "fio2": "Primus/FIO2",
    "peep": "Primus/PEEP_MBAR",
}

RAW_TRACKS = {
    "ECG II": "SNUADC/ECG_II",
    "PLETH": "SNUADC/PLETH",
    "RESP": "Primus/CO2",
    "ART": "SNUADC/ART",
}

REQUIRED_TRACKS = {
    NUMERIC_TRACKS["heart_rate"],
    NUMERIC_TRACKS["o2sat"],
    RAW_TRACKS["ECG II"],
    RAW_TRACKS["PLETH"],
}


def read_vitaldb_index(path: Path) -> pd.DataFrame:
    """Read the official API gzip CSV regardless of its local suffix."""

    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        return pd.read_csv(handle)


def eligible_case_ids(cases: pd.DataFrame, tracks: pd.DataFrame) -> np.ndarray:
    """Return deterministic cases with the minimum portable signal contract."""

    available = tracks.groupby("caseid", sort=False)["tname"].agg(set)
    eligible = available.index[available.map(REQUIRED_TRACKS.issubset)]
    case_duration = pd.to_numeric(cases.set_index("caseid")["caseend"], errors="coerce")
    eligible = [
        int(case_id)
        for case_id in eligible
        if float(case_duration.get(case_id, 0.0)) >= 1800.0
    ]
    return np.asarray(sorted(eligible, key=_case_order), dtype=np.int64)


def _case_order(case_id: int) -> bytes:
    return hashlib.sha256(f"vitaldb-lockbox|{int(case_id)}".encode()).digest()


def numeric_history_features(
    values: np.ndarray,
    anchor_seconds: float,
) -> dict[str, float]:
    """Create the exact MIMIC numeric feature names from 1-second tracks."""

    output: dict[str, float] = {}
    times = np.arange(len(values), dtype=float)
    for column, name in enumerate(NUMERIC_TRACKS):
        if name in {"fio2", "peep"}:
            continue
        array = values[:, column]
        for seconds in (300.0, 900.0):
            mask = (
                (times <= anchor_seconds)
                & (times >= anchor_seconds - seconds)
                & np.isfinite(array)
            )
            observed = array[mask]
            observed_times = times[mask]
            prefix = f"num_{name}_{int(seconds)}s"
            if not len(observed):
                output.update(
                    {
                        f"{prefix}_last": np.nan,
                        f"{prefix}_median": np.nan,
                        f"{prefix}_std": np.nan,
                        f"{prefix}_slope": np.nan,
                        f"{prefix}_count": 0.0,
                    }
                )
                continue
            slope = np.nan
            if len(observed) >= 2 and np.ptp(observed_times) > 0:
                slope = float(
                    np.polyfit(
                        (observed_times - anchor_seconds) / 60.0,
                        observed,
                        1,
                    )[0]
                )
            output.update(
                {
                    f"{prefix}_last": float(observed[-1]),
                    f"{prefix}_median": float(np.median(observed)),
                    f"{prefix}_std": float(np.std(observed)),
                    f"{prefix}_slope": slope,
                    f"{prefix}_count": float(len(observed)),
                }
            )
    return output


def last_observation(values: np.ndarray, column: int, anchor_seconds: float) -> float:
    end = min(int(np.floor(anchor_seconds)) + 1, len(values))
    observed = values[:end, column]
    finite = observed[np.isfinite(observed)]
    return float(finite[-1]) if len(finite) else np.nan
