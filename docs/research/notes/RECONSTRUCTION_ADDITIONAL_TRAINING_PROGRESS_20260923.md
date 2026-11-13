# Additional-Domain Training Progress

Fixed full experiment: five folds, two arms, eight epochs,47targets.
Execution started; this is not a completed efficacy experiment.

## Latest: All Fixed Training Complete (2026-09-24)

All5folds x2arms x8epochs are now available:64executed stages and16certified
equivalent reused stages. Every independently trained arm has403,072updates.
Fold4final sampled fit loss0.6539388real/0.7108092process;243,475unique fit pairs.

All ten checkpoint states were audited together: epoch/fold metadata correct,
paired seen sets/RNG/fit medians identical, every seen pair belongs to that
fold's fit role, tensors finite and dependency hashes current. Fold2destination
hashes match the explicit equivalence certificate. No held-out results were
used to choose epochs or the model. No outer inference has yet been run.

Final fit losses by fold(real/process):0 .5558083/.6010830;
1 .5469040/.5939681;2sameas0(reused);3 .5549454/.6073397;
4 .6539388/.7108092. These are sampled training objectives, not MAE or evidence
of generalization. Next implement/run all-target outer assessment under the
fixed protocol, retaining raw units, fit-normalized aggregation and support
limitations. Calibration and promotion remain unproven; runtime unchanged.

## Latest: Fold 3 Both Arms Complete (2026-09-24)

Both47target arms completed8epochs/403,072updates each and230,194unique fit
pairs. Final sampled fit loss0.5549454real/0.6073397process. Seen sets, RNG,
medians and dependency bindings match; all state tensors finite and seen pairs
fit-only. No settings or training code changed while the processes ran.

64of80states now available:48actually trained plus16equivalent reused.
Only fold4real/process training remains before full outer assessment. No
accuracy, interval or runtime promotion follows from these fit losses.

## Latest: Fold 2 Equivalence / Fold 3 Real Complete

Fold2fit identities, all scaler arrays, every ordered eight-epoch weighted
sampling schedule, fit medians and final numpy RNG match fold0exactly. Source
checkpoints and shared-cache hashes verify. Training code uses fold only to
select fit membership/scaler and name outputs; identical seeded architecture
and optimization therefore permit exact computational reuse. Copied fold2
states are explicitly marked reused, not independent replicas; calibration
and test remain fold2's distinct patients. Certificate:
`artifacts/native_reconstruction_additional_fold_reuse_20260923.json`.

Fold3real independently completed8epochs/403,072updates/230,194unique fit
pairs; final sampled fit loss0.5549454. All tensors finite, all seen pairs
fit-only, all dependency hashes match. Process counterpart still pending.

56of80scheduled states are now available:40actually trained and16certified
equivalent reused. Fold3process and both fold4arms remain. No outer assessment,
calibration, promotion or runtime change. Fit loss is not generalization.

## Latest: Fold 1 Both Arms Complete

Independently trained fold1real/process through all8epochs:403,072updates each,
236,632unique fit pairs. Final sampled fit loss0.5469040real/0.5939681process.
Checkpoint seen sets, RNG, medians and bindings match; all tensors finite,
every seen pair belongs to fit. No outer outcomes inspected.32of80states now
complete; folds2-4 and full outer assessment remain pending.

Important correction: equal fit COUNTS in folds0/1 did not mean equal patients.
Direct inspection found29,527shared people plus15different in each, with2
different scaler locations and6different scales. No checkpoint was reused.
The earlier partition prose was corrected, not the actual fixed partitions.
Folds0/2 have matching fit identities but complete training equivalence has
not been certified. This correction concerns the NEW additional-domain folds,
not the previously verified old crossfit model equivalence.

## Latest: Fold 0 Both Arms Complete

Both arms resumed the epoch1 optimizer/RNG states and completed fixed epochs2-8
without changing settings. Each has403,072updates and236,509unique fit pairs
(33.85% of698,766available pairs). Seen sets, RNG states, fit medians and
dependency bindings match; all state tensors finite and every seen pair is fit.
Final sampled fit loss is0.5558083real and0.6010830process. Epoch6 increased in
both arms; shuffled replacement samples differ across epochs, so these are not
fixed-cohort validation curves. No checkpoint selected by minimum training loss.

16of80scheduled states now complete. Other folds and outer evaluation remain
pending. This supersedes the2of80progress below, not a promotion or evidence
of cross-hospital improvement.

## Fold 0 / Epoch 1

| Arm | Updates | Mean sampled fit SmoothL1 | Unique fit pairs seen |
|---|---:|---:|---:|
| Physiological values |50,384|0.6253965|43,132|
| Process-only |50,384|0.6330501|43,132|

29,542fit patients provide698,766available patient-target pairs. One epoch
samples with replacement and covers about6.17% of those pairs; it is NOT one
complete pass through the dataset. The budget follows the fixed protocol,
not a convergence claim. Selection/calibration/test outcomes were not evaluated.
The small fit-loss difference does not establish transferable physiology.

Both arms used fresh seeded weights, identical architecture, fit baselines and
sampling rules; process-only removes normalized numeric values but keeps the
same event masks/ages/times. On completed checkpoints, seen-pair sets, numpy
RNG states, torch RNG states, median baselines and dependency bindings match
exactly. Every seen pair belongs to fit. All state tensors are finite, current
dependency hashes match, and98state tensors differ between arms after training.
No frozen old encoder parameters were used as initialization.

Two of80scheduled epoch/arm/fold states are complete. Continue from the saved
optimizer/RNG checkpoints, not a fresh restart; do not select a winner using
these fit losses. Full outer evaluation, calibration and external validation
remain pending. No runtime weight, registry, interval or promotion change.

Reports:
`artifacts/native_reconstruction_additional_training_fold0_real_20260923.json`
and `artifacts/native_reconstruction_additional_training_fold0_process_20260923.json`.
