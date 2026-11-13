# Native Replay Latent Objective Diagnosis

Fixed epoch8 none/sparse checkpoints, same64fit/32tune patients. No optimizer
updates or external selection. Baseline teacher is frozen for primary losses.
Current EMA teacher shifts normalized vectors by~.032/.034, but predictions
also improve against the frozen teacher; a moving target alone does not explain
the loss improvement. This is descriptive, not a trained EMA ablation.

| Tune fixed-teacher loss | None | Sparse |
|---|---:|---:|
| Frozen baseline | .005628850 | .005628850 |
| Full learned residual | .001139607 | .001138320 |
| Fit-mean latent residual for every tune patient | .001168130 | .001167243 |

The constant is the mean of (trained predicted latent minus frozen baseline
latent) over FIT patients only, applied unchanged to tune baseline latents.
No tune truth is used to construct it. About99.36% of the observed loss reduction
can be reproduced by this shared additive correction. This control is derived
from a trained model, not an independently trained constant-only baseline.

Thus most latent-loss progress in this small pilot is not evidence for improved
patient-specific dynamics. Combined with worsening value MAE and near-invariant
state shuffling, generic latent alignment is a concrete objective shortcut.
It does not prove removal of the shortcut improves forecasts, nor explain all
earlier JEPA failures. Useful delta information in the teacher is not excluded.

A controlled follow-up should compare the existing latent objective with a
source-centered target/prediction objective, explicitly checking constant-only
and shuffled-patient controls alongside value MAE. Centering must be fit-only,
teacher coordinates fixed or centers recomputed legally during training; do not
change serving representations or claim promotion from latent loss alone.
No runtime/weights/calibrator changes. Reports:
`artifacts/native_organ_teacher_drift_20260913.json` and
`artifacts/native_organ_constant_latent_20260913.json`.
