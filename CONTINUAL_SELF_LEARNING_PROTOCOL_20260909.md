# Continual self-learning diagnostic

Frozen before outcome evaluation on 2026-09-09. This is a bounded source-side
diagnostic for `creatinine@6h`; it cannot promote a runtime source.

## Question

After a forecast has been issued and its real future measurement later becomes
available, can a small continual adapter on the shared whole-body JEPA become
more accurate without erasing previously learned performance?

This experiment tests continual learning, not treatment choice, exploration or
causal effects. A label may update an arm only after its recorded future time.

## Data roles

- eICU exact creatinine-measurement anchors are the sole development source.
- eICU fit patients are deterministically divided into base-head training,
  initial adapter training and retention evaluation patients.
- eICU patient-test trajectories form the online stream. Predictions are saved
  before the corresponding future label can update an arm.
- eICU hospital-test patients measure transfer after online updates.
- MIMIC-IV exact measurement anchors are repeated external validation. MIMIC is
  already inspected and is not an untouched lockbox.
- All rows require observed anchor creatinine, creatinine age zero and an
  observed future creatinine 5-7 hours later.

## Fixed representation and prediction target

The frozen release JEPA produces the same 256 named latent features used by the
measurement-anchor diagnostic. Seven measured creatinine-history features are
appended. A fixed source-trained HistGradientBoosting head predicts the base
signed delta. Every continual arm learns only the residual of this same base
prediction from the same 263-dimensional feature contract.

This first screen updates a small residual adapter rather than the full JEPA
encoder. A positive result is required before exposing the shared encoder to
online gradients.

## Arms

1. `frozen_global`: fixed base head plus the initially trained residual adapter.
2. `frozen_patient_state`: frozen global prediction plus a shrunk running mean
   of that patient's previously resolved residuals. No weights change.
3. `naive_online`: adapter weights update from the newly resolved example only.
4. `dual_replay`: equal-budget updates mix the new example with a short FIFO
   memory and a long reservoir memory, following the useful part of ARROW.
5. `uniform_replay`: equal-budget updates mix the new example with uniform
   samples from a same-capacity history buffer. This is the ordinary replay
   control.
6. `persistence`: anchor creatinine unchanged.

The three weight-updating arms receive one batch of 16 examples per resolved
label. Naive online repeats the new example; dual replay uses 4 new, 6 short and
6 long samples; uniform replay uses 4 new and 12 uniform-history samples. Buffer
capacity, optimizer, initial adapter, labels and update count are otherwise
fixed. Buffers are seeded only with eICU initial-adapter training examples.

## Streaming and leakage rules

- Patient order is repeated under seven deterministic seeds.
- Within a patient, anchors are processed in time order.
- Before predicting an anchor, only earlier examples whose future measurement
  time is no later than that anchor may update the model or patient state.
- Remaining labels from a completed patient become available before the next
  patient, representing a sequential-patient stream.
- The primary personalization slice contains second-or-later predictions for a
  patient; the first prediction cannot benefit from that patient's feedback.
- No future treatment, future measurement time or future value enters features.

## Evaluation

- Report prequential stream MAE overall and for second-or-later predictions.
- Use patient-level paired bootstrap CIs against frozen global and persistence.
- After each seeded stream, evaluate the updated adapters on untouched eICU
  retention patients, eICU hospital-test patients and repeated MIMIC validation.
- Report whether an arm causes statistically detectable harm to retained or
  external performance.
- A source adaptation signal requires lower mean error than frozen global on
  the second-or-later stream slice across all seven patient orders, supported
  by a paired CI in at least four orders, without detectable eICU retention or
  hospital harm.
- A cross-database continual pass is stricter: all seven orders must
  significantly beat frozen global and persistence, and no order may cause
  detectable retention, eICU hospital or MIMIC harm.

Even a favorable result remains `candidate_only`. Full JEPA online training,
new interval calibration and a new untouched lockbox would still be required
for promotion. Runtime weights, registry, calibrators and serving behavior stay
unchanged.

Only aggregate reports and hashes may enter Git. Row-level predictions, patient
identifiers and fitted online models remain local.
