"""Standard physiological derivations for measurements a site does not report directly.

Used by extractors to harmonize sites to the MIMIC-IV measurement definitions;
a derived value is only added where the site has no measured value at that time.

- Calculated bicarbonate (Henderson-Hasselbalch, as reported by blood-gas
  analyzers and in MIMIC-IV itemid 50803): HCO3- = 0.0307 * PaCO2 * 10**(pH - 6.1)
  (mmol/L; PaCO2 in mmHg; CO2 solubility 0.0307 mmol/L/mmHg, pKa 6.1).
- Mean arterial pressure from a same-time systolic/diastolic pair:
  (SBP + 2 * DBP) / 3, only when SBP > DBP.
"""

from __future__ import annotations

import numpy as np


CO2_SOLUBILITY = 0.0307
CARBONIC_PKA = 6.1


def calculated_bicarbonate(ph, paco2) -> np.ndarray:
    ph = np.asarray(ph, dtype=np.float64)
    paco2 = np.asarray(paco2, dtype=np.float64)
    return CO2_SOLUBILITY * paco2 * np.power(10.0, ph - CARBONIC_PKA)


def mean_arterial_pressure(sbp, dbp) -> np.ndarray:
    sbp = np.asarray(sbp, dtype=np.float64)
    dbp = np.asarray(dbp, dtype=np.float64)
    return np.where(sbp > dbp, (sbp + 2.0 * dbp) / 3.0, np.nan)
