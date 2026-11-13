# Continual learning on a validated cell

Frozen before continual outcome evaluation on 2026-09-09.

## Question

Can continual residual learning improve an already validated release source,
rather than merely repairing a weak candidate, while preserving performance on
patients, hospitals and a second database?

## Fixed cell and cohort

- Cell: `glucose@6h`.
- Incumbent: the frozen release `unified_shared_jepa_target_head` selected for
  this cell before this experiment.
- Every row requires an observed anchor glucose with measurement age zero and
  a strict observed future glucose 5-7 hours later.
- The available event cache is anchored on chemistry-panel measurement events;
  this is therefore an exact glucose co-measurement cohort, not every glucose
  measurement in either database.
- eICU is the sole development source. Its fit patients are divided into
  initial replay-memory and retention roles. Separate patient and hospital
  roles come from the pre-existing deterministic role contract.
- MIMIC-IV is repeated external validation and is already inspected. It is not
  an untouched lockbox.

The source-selection screen establishing eligibility is descriptive and occurs
before continual updates: on this strict slice the incumbent significantly
beats persistence in both eICU and MIMIC-IV.

## Representation and update target

The incumbent point prediction remains frozen. Its existing 311-column feature
contract contains the shared context, patient belief, target future latent,
owner-organ latent, base point and measurement-pure state. A bounded linear
adapter learns only the incumbent's signed residual.

The adapter starts at exactly zero, so every arm's first prediction equals the
validated release prediction. Initial eICU replay-memory patients seed only the
memory buffers; their outcomes do not pre-fit a nonzero correction.

## Compared versions

1. `frozen_patient_state`: no weight updates; uses a shrunk running mean of the
   same patient's previously resolved incumbent errors.
2. `naive_online`: updates only from each newly resolved example.
3. `dual_replay`: equal-budget updates combine the new example, a short FIFO
   memory and a long reservoir memory.
4. `uniform_replay`: equal-budget updates combine the new example with uniform
   samples from a same-capacity history reservoir.

`frozen_release` and `persistence` are fixed references. The three
weight-updating versions each receive one 16-example update batch per resolved
label.

## Prequential timing

- Predictions are recorded before the future label is revealed.
- A label may update an arm only after its recorded future measurement time.
- Anchors within a patient are processed in time order.
- Unresolved labels at the end of one patient become available before the next
  patient's stream begins.
- The primary personalization slice contains second-or-later predictions for a
  patient.
- Seven deterministic patient arrival orders are evaluated.

## Gates

A continual arm passes this diagnostic only if it:

- significantly improves the frozen release on second-or-later predictions in
  all seven patient orders;
- significantly beats persistence in all seven orders;
- causes no significant harm on untouched eICU retention patients, eICU
  hospital-heldout patients or repeated MIMIC validation under any order.

Even a pass remains `candidate_only`. A new untouched database, source-bound
interval calibration and a separate release decision are required for runtime
promotion. This experiment does not modify release weights, registry,
calibrators or serving behavior. All outputs remain factual and
`causal_claim_allowed: false`.
