# Independent Encoder Training: Stage 1 Complete

## 2026-09-21 Update: All Training And Outer Evaluation Complete

All80required arm-epoch states available:72executed,8verified equivalent reuse.
Fold4real/process each finished8epochs/283008updates,zero skips; finalsampled
loss.869209/.930542. Alltencheckpoint hashes/finite tensors and per-fold paired
RNG/medians/dependencies verify. No best-epoch selection. All5outerfolds inferred
and scored,247611pairs/247610eligible. Fifteen regression tests pass.

Full result/limitations: `docs/research/findings/RECONSTRUCTION_ENCODER_CROSSFIT_FINDINGS_20260921.md`.
All/broad4descriptive screens (albumin,BUN,Hct,totalprotein),selected0/both0.
No promotion or runtime modification. All earlier pending counts below are
historical and superseded by this completion note.

## 2026-09-21 Update: Fold 3 Training Complete

Fold3 real/process each finished all8fixedepochs and283008updates, with zero
empty-history skips. Training uses6299fitpatients and its fit-only normalization
and observation vocabulary. This fold does not have source anion-gap support;
that predeclared input restriction remains intact. It was not tuned to results.

| Epoch | Real sampled fit loss | Process-only sampled fit loss |
|---|---:|---:|
| 1 |0.950698|0.959735|
| 2 |0.903747|0.936199|
| 3 |0.870121|0.917992|
| 4 |0.861457|0.917820|
| 5 |0.870867|0.930709|
| 6 |0.799168|0.867374|
| 7 |0.766209|0.829104|
| 8 |0.841241|0.911205|

Retain fixed epoch8, not the lower sampled epoch7 loss. No outer predictions
have been scored. FinalRNG equality, medians and dependencies match; every
floating model tensor is finite; bound files and private checkpoint/report
hashes verify. Jobs exited normally. This is training progress, not efficacy.

Status:64/80required arm-epoch states available,56actuallyexecuted and8verified
equivalent reused. Fold4 training and complete outer evaluation remain. No
runtime, registry, intervals or promotion changed.

Evidence: `artifacts/native_reconstruction_encoder_crossfit_fold3_real_20260920.json`
and `artifacts/native_reconstruction_encoder_crossfit_fold3_process_20260920.json`.

## 2026-09-21 Update: Fold 2 Training Complete

Fold2 real and process arms each completed all8fixedepochs/283008updates,
with no skipped empty histories or nonfinite losses. This fold fits5513patients
and uses different fit-only scalers than fold0/1. It is not reused training.

| Epoch | Real sampled fit loss | Process-only sampled fit loss |
|---|---:|---:|
| 1 |0.687487|0.698334|
| 2 |0.592517|0.619828|
| 3 |0.565555|0.599197|
| 4 |0.518926|0.563252|
| 5 |0.546632|0.591974|
| 6 |0.540076|0.586825|
| 7 |0.506614|0.556468|
| 8 |0.491907|0.545704|

Do not compare these fold-normalized sampled objectives directly with the
other folds' objectives or call their difference a held-out gain. Outer
predictions have not been scored. Epoch8 is retained, with no best-epoch search.
Full model finite-tensor checks, final RNG equality between arms, matching
readout medians/dependencies, bound file hashes and aggregate/private checkpoint
hashes all pass. Both jobs exited normally, not interrupted.

Protocol progress:48/80required arm-epoch states available,40executed and8
verified equivalent reused. Folds3-4 and all outer evaluation remain pending.
No runtime, registry, interval or promotion change.

Evidence: `artifacts/native_reconstruction_encoder_crossfit_fold2_real_20260920.json`
and `artifacts/native_reconstruction_encoder_crossfit_fold2_process_20260920.json`.

## Update: Fold 1 Complete Through Verified Computational Equivalence

Fold1 real independently ran all8epochs/283008updates. Its fitting population
is exactly the same ordered4969patient shards as fold0: the two large outer
components exchange test versus unused inner-selection roles. Source scalers,
all8training logs, full model tensors, optimizer tensors, NumPy/Torch RNG,
initial medians and dependencies are **bitwise/equivalently identical**.

After verifying that equality, fold1 process control reuses fold0 process
weights with explicit source/destination provenance and zero additional process
optimizer updates. It is not a separately executed replicate. Two regression
tests reject changed values, input order or mapping support. Aggregate evidence:
`artifacts/native_reconstruction_encoder_fold_equivalence_20260920.json`.

Status:32/80 required arm-epoch states available, **24/80 actually executed**,
8equivalent reused. Folds2-4 still require training. Outer test populations for
fold0/1 differ and still need separate predictions, but their fitted model is
the same: do not count them as independent model replications. No outer scores
have been used to make this computational decision. No runtime/promotion change.

This also verifies deterministic retraining at full8epoch length rather than
merely asserting that seeded code should be deterministic. It is reproducibility
evidence, not an efficacy result or a solution to all cross-hospital failures.

## Update: Fold 0 Training Complete

Both fold0 arms have now finished all eight fixed epochs, 283,008 successful
updates each, with zero empty-history skips. Full scope progress is16/80
arm-epochs; folds1-4 and all outer evaluation remain pending. No inner or outer
score selected an epoch; final epoch8 is retained exactly as specified.

| Epoch | Real sampled fit loss | Process-only sampled fit loss |
|---|---:|---:|
| 1 |1.119969|1.131107|
| 2 |0.969936|1.004880|
| 3 |0.933296|0.983506|
| 4 |1.037267|1.097014|
| 5 |1.051793|1.122486|
| 6 |0.863830|0.931574|
| 7 |0.900039|0.969537|
| 8 |0.922825|0.999242|

These are sampled training objectives, not held-out MAE or a generalization
claim. Replacement schedules differ across epochs; nonmonotonic epoch means
cannot alone establish optimization instability. Both arms used matched plans.
The apparent real-input advantage is not yet evidence that it transfers.

Verified after completion: epoch counts, 283008updates per arm, finite model
tensors, identical final NumPy/Torch RNG states, identical initial medians and
dependency bindings, all bound file hashes still current, both private
checkpoint hashes match their aggregate reports. Jobs exited normally; no
training was interrupted. Runtime/promotion remains unchanged.

The following section preserves the earlier stage1 history, superseded by the
completion counts above.

The fixed protocol is five folds, two arms and eight epochs per arm. Only
fold0/epoch1 of both arms is complete: 2 of 80 planned arm-epochs. This is
actual end-to-end current-reconstruction encoder training, not another frozen
latent readout experiment. It is not a finished accuracy test.

Both arms initialize the same architecture from fresh random parameters.
Release weights are never loaded; a regression test mutates every checkpoint
tensor except the three structural schema buffers and verifies bitwise identical
new initialization. Source-fold preprocessing and observation support are fixed
before training. The prior one/two-site inner selection is not used to pick
epochs. No outer predictions or labels have been scored to select a checkpoint.

| Fold 0, epoch 1 | Real numbers | Process-only |
|---|---:|---:|
| Scheduled updates | 35376 | 35376 |
| Completed updates | 35376 | 35376 |
| Empty-history skips | 0 | 0 |
| Mean sampled training SmoothL1 | 1.119969 | 1.131107 |

Training loss is in new fold-local normalized units and is not comparable with
the previous pooled-scale macro MAE. The small training difference is NOT
evidence of generalization, physiological coupling or promotion. Matching final
NumPy and PyTorch RNG states, initial readout medians and all dependency hashes
are verified. 98 state tensors differ after training, confirming the two arms
did not accidentally reuse a single trained state. Optimizer/RNG/model state
is saved privately for deterministic epoch-boundary continuation.

Eleven tests pass across independent initialization, equal target exposure,
fold isolation, source-only scaling and unsupported-input handling. The first
stage completes without nonfinite loss or interrupted training. Warning about
Transformer nested-tensor optimization is not a training failure.

Remaining: epochs2-8 for both fold0 arms, all stages of folds1-4, then matched
outer inference and patient/site evaluations including abstentions and every
target. No target or checkpoint may be selected using partial outer results.
Do not silently reduce the protocol to fold0 or declare epoch1 a negative result.
The experiment remains reused source development, not a new external lockbox.

No release weight, runtime registry, interval or promotion changed.

Evidence:
`artifacts/native_reconstruction_encoder_crossfit_fold0_real_20260920.json`
and `artifacts/native_reconstruction_encoder_crossfit_fold0_process_20260920.json`.
