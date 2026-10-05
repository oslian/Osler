# Nested Selection Matched Training

Freeze previous33training targets and536fit patients. New encoder selection uses
only90patients/4sites; readout89patients/7sites and evaluation178patients/11sites
are never loaded by this trainer. All39catalog support statuses retained; only
28can later pass the existing30patient requirement in every selection role.

Keep same seed19 initialization, inherited scaler, architecture, target-balanced
complete schedule, AdamW1e-4/default decay, SmoothL1, gradient clip1, eval-mode
dropout,8epochs and real/process/placebo arms as the previous strict-site fit.
Checkpoint score is original33target-equal MAE but on encoder-selection only.
Including targets with fewer30encoder-selection patients in an aggregate training
score does NOT make them eligible for per-target validation or promotion.

Record per-target score each epoch and save every epoch's state (private) so
later diagnoses do not require refitting. Initial epoch0 may win. New optimizer
training starts from same release initializer, never previous selected weights.
Check old/new per-epoch training draw loss identity; selection alone must not
change optimizer trajectory. No stop/restart based on heldout outcomes.

Warm-start current source identities/sites are disjoint by original crosswalk,
but historical exact-source hash remains unresolved. This is a development
comparison, not clean historical-pretraining or fresh external validation.
Future readout selection uses only the reserved readout group after encoder
freezing; no runtime/promotion follows training alone.
