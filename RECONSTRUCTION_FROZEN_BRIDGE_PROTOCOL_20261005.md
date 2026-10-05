# E34: fixed backbone versus joint bridge adaptation

## Question and unique change

E33 localized some held-out harm to the trained bridge at inference. It did
not isolate training coadaptation. Keep the SAME E29 original epoch10 initial
model, data, masking, order, seed20261004, two passes, AdamW lr1e-4/wd.01,
SmoothL1 and gradient norm clip1. Only bridge.projection.weight may update.
The body/state/readout and event encoder remain bitwise fixed. Do not detach
the recurrent forward graph: bridge gradients must traverse later updates.

All five folds and all three channels (immediate/delayed/body), all 53 target
inventory. Reuse verified E29 joint-trained models as the same-initialization
comparison, not a differently initialized or freshly tuned joint model.
Match starts/contracts, log optimizer steps, and test frozen weights exactly.
This is an optimization-scope ablation, not equal trainable capacity between
joint and fixed; the three fixed channels have equal trainable parameters.
Gradient clipping may act differently with fewer trainable weights.

## Data and controls

Same E29 fit2048/selection1024 per fold, E31 same_source1024 per fold. No new
patient selection and no fitting on selection/same_source. Retain all support
and unsupported targets. E31 may be original-encoder-exposed, not a lockbox.
Zero-start parity uses original state/readout plus binding organ proposals:
bound_start. Saved frozen uses native proposals, so retain BOTH references.
Keep E27 unbridged and full-fit raw unchanged; raw exists only for selection.
Preserve current-state target/sibling blocking and event availability times.

## Fixed assessment

Selection primary candidate fixed_immediate versus seven controls:
immediate (E29 joint), fixed_delayed, fixed_body, bound_start, unbridged,
frozen, raw. Require >=30 patients and >=3 source components, patient and
component paired bootstrap5000 upper CI<0, one-sided p<.05/(53*7), using
existing paired_summary. All seven must pass for the full development screen.
No change to old gates or runtime promotion. Shared encoder and repeated
patients/folds prohibit independent-fold claims. Average repeats per patient.
Report fit and same_source descriptively, without inventing missing raw.

If the fixed bridge also improves fit but loses on new roles, coadaptation is
not necessary for that failure. If fixed wins only joint but not strong raw,
it is not a complete repair. If neither improves fit, inability to fit this
budget/representation remains; do not diagnose all failures as domain shift.
No epoch/seed/lr sweep or posthoc target selection from this run.

## Validation and artifacts

Before training, synthetic tests cover zero-start parity, trainable gradient,
unchanged backbone, malformed identities and patient-equal scoring. Load
saved weights before inference. Compare normal forward and independent
explicit event schedule on ALL three roles for all15 saved models, atol1e-7.
Hash-check parent assessments, predictions, inputs and contracts; refuse
changed completed artifacts. Record private patient predictions and weights
only under local data; public JSON contains aggregate results/hashes.
Verify patient/component scalar aggregates, saved controls and statistics
after completion. Abort on nonfinite data, hash/parity failure, or <10GiB
free. The fixed branch is closed if it does not pass, unless new mechanism
evidence justifies a distinct intervention. No clinical, causal, future or
untouched-external claim; no release change in this experiment.
