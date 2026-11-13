"""Patient-level split of held-out calibration patients into selection and conformal halves.

Split-conformal validity needs calibration scores from patients that played no
part in any fitting or choice.  Held-out MIMIC patients therefore do one of two
jobs, never both:

- selection patients: choose among candidate policies (hurdle policy, JEPA
  coupling path) and fit nuisance scales (interval scale, point-correction
  scale);
- conformal patients: provide the calibration scores only.

The split is a fixed hash of the patient identifier, so every stage (recipe
fitting, JEPA path selection, deployment replay) agrees without passing seeds.
"""

from __future__ import annotations

import hashlib

import numpy as np


def selection_patients(subjects) -> np.ndarray:
    """True for rows whose patient belongs to the selection half."""

    subjects = np.asarray(subjects).astype(str)
    unique, inverse = np.unique(subjects, return_inverse=True)
    chosen = np.array([int(hashlib.sha256(f"selection:{s}".encode()).hexdigest(), 16) % 2 == 0 for s in unique], dtype=bool)
    return chosen[inverse] if len(unique) else np.zeros(0, dtype=bool)
