# E33: trained bridge on/off localization

## Fixed question

E29 adapts the recurrent state, readout and bridge together. E32 cannot tell
whether failure belongs to the bridge at inference or the remaining coadapted
weights. This audit does not train, search, promote or change release weights.

Use all five saved E29 folds, all three channels (immediate, delayed, body),
all 53 target inventory, E29 fit/selection and the E31 same-source query role.
Expected query occurrences: 10240 + 5120 + 5120 = 20480. Do not select targets
or remove small components after inspecting outcomes. Roles remain separate.
E31 is adaptation-unseen, mostly epoch10-exposed, not an external lockbox.

## Intervention and controls

For each trained checkpoint, deep-copy and zero only bridge.projection.weight.
Verify every other parameter/buffer is identical; never save over weights.
Compare on, off, saved E27 unbridged and frozen references; retain raw where
already available. On must reproduce saved predictions (absolute tolerance
1e-7). Off must match native advance/update replay with the same trained
layer/readout (1e-7). Preserve times, masks, target/sibling masking and inputs.
Hash-check existing independent verifications and all bound dependencies first.

Per query, with absolute error L:

    L(on)-L(unbridged) = [L(on)-L(off)] + [L(off)-L(unbridged)]

First term is the total inference-path effect, including later recurrent
consequences, not just the last correction. Second term contrasts remaining
coadapted weights with a separately trained unbridged model. Neither is an
isolated causal training effect: off can be out of training distribution.
Positive terms indicate harm; zero is not a significant null result.

## Reporting and stopping

Report each role/target/channel with patient-equal and source-component-equal
means, support, plus fold-level results. Average repeated folds within patient
before averaging patients; do not claim independent fold replications.
Compare fit-to-query changes in both terms, retaining missing support. This is
descriptive diagnosis, not a new significance/promotion screen. No threshold
selection, refitting, label-policy changes or release update is allowed.

Stop on changed hashes, role/identity mismatches, nonfinite outputs, parity
failure or fewer than 10 GiB free. Save row predictions only in private data;
public artifacts contain aggregate results and hashes. Check summaries from
saved predictions independently, plus synthetic unchanged-weight, nonzero
bridge, repeated-patient and cancellation counterexamples. Current masked
reconstruction does not establish future forecast or clinical causality.
