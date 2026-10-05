# Availability-Aware Encoder Fine-Tuning Findings

## Decision

Completed three-seed paired pilot with REAL context-encoder updates and EMA
future-delta JEPA supervision. Not the preceding frozen311-feature ridge trial.
No full promotion, release or interval change. Eight-epoch warm-start pilot does
not establish that all possible end-to-end training has been exhausted.

## Fixed Comparison

Same availability-safe glucose inputs/labels and patient roles: development
7,140 pairs/386 patients, hospital6,077/364, MIMIC6,616/889. Four arms combine
own/full accepted labs with frozen/trainable context encoder. Existing neural
future and value components train in both frozen-context controls and end-to-end
arms. Inputs have real masks/ages throughout, not measurement-time leakage.

Same initialization, patient-balanced minibatches, optimizer and budget.
Seeds7/19/31; 8 epochs x16 steps x32 samples. Source-tune MAE selects weights,
including epoch0. No external checkpoint selection or hyperparameter sweep.
Glucose paired-delta teacher uses the existing stop-gradient target encoder,
EMA-updated after each step. Context never sees future labels or their actual
measurement schedule. The nominal request stays6h with actual5-7h truth.

This is a direct neural value-head endpoint, NOT the existing311-feature shared
head. Compare within this experiment, not against an older head as if identical.
Pretrained patient overlap remains unaudited; no new lockbox claim.

## Training Really Reached the Encoder

All six selected end-to-end checkpoints changed77 context tensors; all six
frozen-context controls changed0. Maximum observed context gradient norms were
2.54-8.61 in end-to-end arms,0 in controls. All selected epochs were nonzero.
Twelve fitted checkpoints were retained under local temporary storage with
SHA256 bindings in the aggregate report. These are candidate weights, not release.

## Results

Patient-equal MAE mg/dL, medians across three seeds:

| Model | eICU hospital | MIMIC |
|---|---:|---:|
| Persistence | 45.878 | 37.449 |
| Initial whole-input direct neural model | 42.749 | 35.300 |
| Own input, frozen context | 42.164 | 34.622 |
| Own input, end-to-end | 42.112 | 34.536 |
| Whole input, frozen context | 42.459 | 36.271 |
| Whole input, end-to-end | 42.386 | 36.091 |

Significantly better paired patient-bootstrap comparisons:

| Comparison | eICU hospital | MIMIC |
|---|---:|---:|
| Whole end-to-end vs whole frozen context | 1/3 | 1/3 |
| Own end-to-end vs own frozen context | 1/3 | 2/3 |
| Whole end-to-end vs own end-to-end | 0/3 | 0/3 |
| Whole end-to-end vs persistence | 3/3 | 2/3 |
| Whole end-to-end vs initial whole model | 1/3 | 0/3 |

For seed31, whole context effective rank remains16.61 hospital/13.55 MIMIC,
versus frozen15.96/14.22. State did not trivially collapse to a constant; nonzero
rank does not demonstrate physiological correctness or useful coupling.

## Interpretation

Corrected-input fine-tuning is feasible and creates small differences, but
does not solve the cross-database whole-body increment. It would be incorrect
to keep saying encoder retraining is entirely untested, or to infer the only
cause was old weights seeing new masks. This particular warm-start budget
failed to establish stable incremental gain.

Remaining uncertainties: limited pilot budget, frozen source normalization,
lab-support asymmetry, pretrained representation and the bounded direct value
head. The existing direct head bounds its normalized residual; this endpoint
is not equivalent to the old shared head. No loss-weight/learning-rate/head
sweep was performed. Value and JEPA losses have different numerical scales;
no claim that they contribute equal gradient strength.

Full treatment/bedside reconstruction, from-scratch training,7-seed formal
gate, other-target retention, shuffled-input placebo, care-unit/time testing
and conformal calibration remain untested here. This is not a rejection of
all JEPA or all physiological information, and no runtime source is promoted.

## Artifacts

- `artifacts/availability_encoder_20260910.json`: all training logs, changed
  context counts, checkpoint hashes, patient CIs and latent rank diagnostics.
- `scripts/train_availability_encoder.py`: bounded paired training and evaluation.
- `tests/test_availability_encoder.py`: context trainability, stop-gradient and
  equivalence to the existing target-specific future encoder.
