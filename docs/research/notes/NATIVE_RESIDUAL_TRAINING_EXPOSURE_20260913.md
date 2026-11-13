# Residual Training Exposure

Replayed the baseline sampler exactly through selected epoch25. All64 residual
fit patients had contributed baseline optimization;22/64 exact residual queries
had been sampled. Tune/evaluation/hospital each had zero sampled patients.
Tune patients did contribute baseline epoch selection, so are not untouched.

Residual versus current MAP least-squares slopes: fit-.16834, tune+.08898,
evaluation+.01441, hospital-.03628. Descriptive small samples, no confidence
intervals or causal claim. Original fit/tune MAEs reproduce float32 tolerance.

The new residual is trained on in-sample baseline errors, not cross-fitted
out-of-patient errors. This distribution asymmetry is verified; that it causes
all observed failure is NOT verified. Small samples, patient mix and baseline
generalization may also contribute. No external data were used to choose a fix.

Correct next stacking comparison needs baseline predictions on patients excluded
from baseline fitting AND selection, with residual fitting/selection separated
inside development. Shared encoder features must obey the same exclusion; merely
swapping predictions from another seed risks leaking through its training set.
Warm-start pretraining provenance also needs explicit qualification. Do not
call historical experiments leaked externally solely from this finding: heldout
fitting separation was preserved. No runtime/calibration/promotion changes.

Evidence: `artifacts/native_residual_training_exposure_20260913.json`.
