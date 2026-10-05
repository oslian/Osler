# E27: Paired Organ Assignment Intervention

## Question and prior evidence

E26 proves additive organ pooling loses the association between variable identity
and numeric tuple. The body branch distinguishes those swaps. This local mechanism
does not yet explain real masked-reconstruction error. E21 trains a different
content objective; E25 adds frozen organ coordinates to readers. Neither tests
the present intervention. The established pre-aggregation element pattern is
described in [Deep Sets](https://arxiv.org/abs/1703.06114).

## Fixed comparison

Use all five existing E19/E21 role sets: 2048 fitting and 1024 selection contexts
per fold, original outcome-blind patient/context hashes, all 53 variables inventoried.
Keep source-fold mapping (fold2 uses fold0 epoch10 initialization), target/sibling
masking, available timestamps, scaler, event body representation and state path.
This is a bounded reused-development mechanism experiment, not full-cohort or
untouched-external validation. Repeated people across folds are averaged before
statistics. Fitting and selection are disjoint within each fold.

Three trained arms, no additional learned parameters:

- native: mean(phi(numeric) + identity), original proposals.
- binding: mean(SiLU(phi(numeric) + identity)), pre-pooling binding from E26.
- postpool: SiLU(mean(phi(numeric) + identity)), still assignment-invariant.

Only observed variables contribute. Absence remains absence; observed zero is
valid. Body proposals are identical across arms. Postpool is a nonlinearity and
scale control, not an exact distribution match. If binding beats only native,
do not attribute gain specifically to value/identity binding.

Start each arm from identical original epoch10 state layer and masked readout.
Freeze all event-JEPA parameters. Train only state layer and existing masked
readout, SmoothL1 on actual masked current truth, exactly two passes, AdamW
lr=1e-4, weight_decay=.01, global gradient clipping=1, seed=20261004. Identical
context order, eval-mode networks (no dropout), deterministic CPU operations.
No auxiliary content loss, extra heads, early stopping or selection-based tuning.
This tests state/readout adaptation to repaired proposals, not full encoder
retraining. Failure cannot disprove all binding architectures or larger training.

Cache frozen event proposals once, verify original body/organs/presence/times,
original final state and prediction against E21; regenerate binding from the
same original masked token stream. Never overwrite old inputs or weights.
Bind source, original data roles, cache digests, protocol and code to every output.
Retain original frozen native prediction and the existing full-fitting raw HGB
reader on the identical selection contexts. Do not replace the strong raw model
with a new weak small-sample fit. That raw control has greater fitting support;
comparisons isolate trained-arm changes, not a universal model ranking.

## Evaluation and falsification

Report fit and selection normalized MAE for every supported target; do not drop
harmful targets. Primary evidence: selection patient-equal and source-component-
equal paired bootstrap, 5000 repetitions, averaged repeated fold observations.
At least30 people and3 source components, CI95 upper<0, one-sided p<.05/(53*4)
for each of four prespecified binding contrasts: native, postpool, frozen, raw.
Report unsupported variables explicitly. Full descriptive screen requires all
four contrasts; mechanism-increment screen requires native and postpool.
These screens are not runtime promotion or future-forecast validation.

Binding improves both paired trained controls: supports usefulness of repaired
local association for this task. Improvement over native only: nonlinearity or
scale remains a confound. Fit improves but selection reverses: transfer/adaptation
problem remains. Neither improves: local structural correction is insufficient
for this fixed model/budget. Strong raw still wins: full reconstruction repair
not achieved. Do not sweep hyperparameters after results.

Regression tests must check unchanged body and parameter counts, original parity,
pre/post-pool swap distinction, zero/missing, masking, gradients, role isolation,
saved prediction replay and patient-equal scoring. Independent saved-model replay
and direct raw prediction verification are required before final findings.

## Boundaries and execution

Data rows, identities, proposal caches and candidate weights stay in ignored
private storage. Aggregate evidence only is eligible for commit. Keep10GiB free;
stop on role/hash/nonfinite/parity errors and investigate, not reinterpret them as
physiology. Resume only completed hash-bound stages, never restart a live process.
Run all five folds, not only successful folds. No serving, release, interval,
causal, clinical or complete-root-cause claim. The full research objective remains
open even if this mechanism increment passes.
