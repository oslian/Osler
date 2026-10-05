# Raw Observation Block Diagnostic: Completed

All47 targets/five folds,235 paired target-fold groups,940 HGB fits completed.
Same E17 outcome-independent bounded samples: at most512 fit/256 selection
patients per group;59629 query occurrences,5703 unique patients. This is
current-value masked reconstruction on reused development, not forecasting,
a full-cohort repair, a new lockbox, or release promotion.

## What Was Actually Changed

After target/history/sibling masking, raw265 contains latest53, mean53,
presence53, lag53 and logcount53. Remove one specified block in both fit/query,
refit the same HGB, keep every other setting and sample fixed. Presence remains
in every arm; removing lag/count does NOT remove all observation-process cues.
All inputs are265-wide with constant-zero removed columns, not equal information.

Raw reproduction agreed with E17 at absolute tolerance1e-12 in all235 groups.
E17 body comparator is body64 PLUS the same159 process features; it is not a
body-only or native-head comparator. It was reused unchanged, not newly fitted.

## Results

Normalized target-macro MAE, lower is better; these are NOT accuracy percentages:

| Input | Patient equal | Source-component equal |
|---|---:|---:|
| Full raw |0.869179|0.871693|
| Latest + presence/lag/count (remove mean) |0.873490|0.878645|
| Mean + presence/lag/count (remove latest) |0.876621|0.877506|
| Latest + mean + presence (remove lag/count) |0.870159|0.867923|
| E17 body + process |0.885501|0.880512|
| E17 process only |0.949792|0.947550|

All seven prespecified comparisons have0 corrected diagnostic screens. This
is not evidence of equality and is not an equivalence/noninferiority test.

Descriptive patient-mean counts: removing mean worsens29/47 against raw;
removing latest worsens34/47; removing lag/count worsens20/47 and improves27/47.
The last arm still beats body in34/47 patient means and28/47 component means.
These signs are NOT corrected significant wins.

Examples of patient-equal normalized MAE (all included in the fixed family):

| Target | Full raw | Remove mean | Remove latest | Remove lag/count | Body+process |
|---|---:|---:|---:|---:|---:|
| BUN |.614154|.620824|.613307|.603504|.630306|
| Creatinine |.806442|.813007|.817131|.797238|.824523|
| Glucose |.793373|.794169|.788062|.791300|.792428|
| Bicarbonate |.599738|.593723|.606838|.593154|.614139|
| Ferritin |.956009|.958070|.951304|.949284|.957711|
| Troponin I |3.772151|3.834112|3.808853|3.760085|3.821802|
| INR |.814182|.810567|.818568|.813035|.808980|

## Diagnosis Update, Not A Claimed Repair

The observed average raw advantage does not vanish when explicit lag/count
features are removed. Patient macro retains about94% of the raw-versus-body
mean gap; component macro improves further. This ratio is descriptive, not
information retained, explained variance, causal attribution or significance.
It weakens the specific explanation that explicit lag/count features alone
account for the raw mean advantage in this fixed diagnostic.

Removing latest or mean has a larger patient macro penalty than removing
lag/count, but target-level patterns differ and no corrected comparison passes.
Therefore neither a universal latest-only nor mean-only state objective is
justified. Presence, ordering selection and sampling still encode workflow;
this does not establish a pure physiology signal or remove coordinate effects.

E15 already decoded raw coordinates from body without a meaningful repair.
E17 already tested event pooling. Do not repeat either under a new name.
The remaining question is whether the same observable numeric content can be
retained/read before versus after persistent aggregation under matched probes.
This experiment does not answer that question or authorize an unplanned sweep.
No automatic encoder retraining follows this negative/limited diagnostic.

## Verification

- Exact E17 samples, role separation, components, labels, cache hashes and
  reference prediction hashes checked. Every feature payload checksum checked.
- Post-run recheck: all235 saved prediction/model artifacts and fold/assessment
  bindings match. All940 saved models have265 inputs and100 fitted iterations;
  all predictions and recorded fit errors finite.
- Seventeen tests pass (five new block tests plus twelve prior stage tests),
  covering removal boundaries, retained presence, no input mutation, invalid
  schemas, pair ordering/truth and repeated-fold patient equalization.
- Runtime, weights, registry, intervals and promoted cells unchanged.
  Saved diagnostic heads/private rows remain ignored under data, not published.
- The process exited0. Arrow CPU-discovery permission warnings and loky core
  discovery fallback did not interrupt execution or baseline reproduction.

Evidence: `RECONSTRUCTION_RAW_BLOCK_PROTOCOL_20261003.md`,
`artifacts/native_reconstruction_raw_blocks_fold{0..4}_20261003.json`, and
`artifacts/native_reconstruction_raw_blocks_assessment_20261003.json`.
