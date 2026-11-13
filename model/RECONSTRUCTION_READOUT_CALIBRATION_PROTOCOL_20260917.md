# Frozen Source Readout Calibration Contrast

Use all six existing V2 source prediction caches and their source evaluation
rows. These data have been inspected: this is development evidence, not a fresh
lockbox. Do not use hospital or strict-site evaluation for calibration.

For each supported target, freeze shared state and selected checkpoint. Fit on
source tune predictions only: (1) additive median residual; (2) nonnegative-
slope affine L1 calibration using scipy.optimize.linprog. No sign reversal,
hyperparameter sweep or outcome-based target selection. Apply identical methods
to real and process controls. Tune already selected checkpoints, so the source
evaluation, not tune improvement, is the reported comparison.

Retain all available targets/unsupported entries, require patient-disjoint
fit/tune/evaluation, reproduce cached MAEs and hashes. Compare calibrated real
to frozen real, population and equally calibrated process; report all mean
errors and nominal paired patient bootstrap intervals (2000 draws, seed20260917).
These intervals are exploratory and not multiplicity corrected. No promotion,
future forecast, interval-calibration or fresh-external claim.
