# Magnesium source exposure

Exact patient-uniform sampler replay, no fitting. Each fit contains437-453
distinct patients. Whole selected checkpoints have99.315-100%patient exposure,
71.166-97.785%row coverage and75.603-98.113%future-event coverage.
At epoch32 all patients,98.600-99.130%rows and98.856-99.528%future events have
been sampled. Own selected checkpoints range89.931-100%patient exposure.

This does not support wholesale omission of most training patients as the
explanation for the whole-input generalization gap. Patient-uniform sampling
also prevents long-stay row counts alone dominating draw probability.
It does not prove convergence, sufficient independent patient diversity,
balanced subgroup support, or that more data cannot help. Repeated gradients
from437-453patients are not equivalent to new patients. Earlier2250-patient
glucose experiments used a different task and do not settle this magnesium
cohort question. Do not extend epochs merely to see already-covered rows.

No runtime, calibration or promotion. Artifact magnesium_exposure_20260911.json.
