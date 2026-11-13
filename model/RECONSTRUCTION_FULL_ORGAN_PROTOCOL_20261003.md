# E25: Full-support readout of the same masked body and organ state

## Question and distinction

E24 established that E23's bounded raw reader did not represent the full-fit
raw control on identical queries. E25 tests organ-state utility with all of
E3's fixed fitting/selection support, not another bounded concatenation sweep.
It does not retrain the encoder, change the runtime, or evaluate forecasting.

## Fixed scope

- Original frozen epoch10 checkpoints; fold2 uses the documented source fold0.
- Five E3 source folds, all their fit and selection feature keys, 53 variables
  inventoried including unsupported targets. E3 has 47 supported targets.
- Original target and sibling masking, actual times, source-only scaling and
  unique patient-target query identities. Patient roles stay unchanged.
- MIMIC/NWICU are reused development data, not an untouched external lockbox.
- No outer evaluation, no model choice or settings based on selection results.

## Readout experiment

One prespecified HGB100 family uses `train_nonlinear_probe.PARAMETERS` and
the existing bounded component weights on identical fitting rows for all arms:

1. body64 plus the same159 process columns;
2. body64 plus final organ compartments plus the same159 process columns;
3. arm2 with same-target organs cyclically assigned from another patient,
   separately within each role, ordered by patient identity;
4. arm2 replacing organs by a fixed random linear expansion of body64;
5. raw265, plus the original frozen native prediction as a reference.

Expansion seed is 20261003+fold, normal(0,1)/8, width equal to flattened organs.
No additional Ridge family or parameter sweep. The original E3 body64 reader
cannot replace arm1 because it lacks the159 process columns. Saved full raw
readers can be reused only after exact data, weights, parameters and prediction
parity checks; otherwise stop and investigate, do not silently refit.

Five primary contrasts compare arm2 against body, shuffled, expanded, raw and
native. For each target, average absolute error over repeated folds per patient
first, then compute patient-equal and source-component-equal paired summaries.
Use the existing paired_summary, 5000 bootstrap repetitions, seed20261003 plus
target_index*1000 plus contrast_index (component adds100). Require at least30
unique selection patients and3 components, both upper95% CI<0 and both p_less
<0.05/(53*5). Report fit errors and actual tree leaves separately, all targets
and failures, not just winners. These are research screens, not release gates.

True organs must beat body and both width controls to support incremental organ
utility; beating raw and native is also required for the full comparative claim.
If not, do not declare absence of organ information. No future, self-learning,
causal, clinical, calibration or runtime-promotion claim is permitted.

## Extraction and cost gate

Use the original NativeBodyReconstruction.forward and capture its trajectory
output state; never recompute an approximate state or unmask target history.
For every reference query verify raw265, normalized truth, body64 (absolute
tolerance1e-6), and native prediction (absolute tolerance1e-7). Store only final
organ tensors, reference digest and parity diagnostics in a private SQLite
cache; do not duplicate raw/history/body data. Bind inputs, source and protocol.
Transactions allow resumption, but completion requires exact reference-key
coverage and payload integrity. Role, schema, weight, time or hash violations
stop the run; do not skip invalid rows or turn them into unsupported silently.

Initial engineering batches are fold0 fit and selection, first512 missing keys
in reference patient/target order. This is a throughput/parity test, not a
representative accuracy study. Record support counts, elapsed time and bytes;
estimate full cost before proceeding. Disk reserve10GiB, no deletion of existing
data. Later finite batches extend the same full scope; a partial cache never
counts as completed E25. Resume may change batch budget but not scientific
contract. Stop before low disk and retain committed valid rows for resumption.

## Required completion evidence

All ten role/fold caches complete with per-query parity, all full-support arms
trained and scored, independent saved-output and paired-statistic verification,
all unsupported targets reported and findings linked in the root-cause ledger.
Engineering success alone does not resolve the model root cause.
