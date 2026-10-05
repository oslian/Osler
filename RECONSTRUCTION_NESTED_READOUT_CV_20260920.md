# Conditional Nested Readout CV

## Why This Is Not Another Independent-Head Experiment

The 2026-09-17 nested-selection and independent-head confirmation experiments
already separated encoder selection from head confirmation. They did not
establish a general repair. Do not describe independent head selection as an
entirely untested idea. The present diagnostic instead isolates the reliability
of the latest bounded7 anchored-head selection procedure on its cached states.

All 33 trained targets, all existing source-fit patients, same frozen encoder,
same original heads and four fixed anchoring penalties. For each of five outer
hospital folds, choose a penalty (or unchanged) using at most four inner
hospital folds of the outer training set only. Refit on that outer training
set, then predict the outer fold. All scaling and correction fitting are local
to the relevant training fold. Score groups equally and hospitals equally
within each group, matching the preceding audit. No label-derived exclusions.

Original head and encoder have already seen these source-fit hospitals.
Therefore this is CONDITIONAL nested validation of an additional correction,
not end-to-end independent encoder validation. It cannot settle the latter.

## Results

| All 33 targets | Ordinary selected CV | Nested selection CV |
|---|---:|---:|
| Mean normalized MAE change versus original | -0.0225194 | -0.0061255 |

Nested outer predictions improve 14 targets numerically, worsen 17 and leave
two unchanged. Absolute mean benefit is about 73% smaller. This is descriptive,
not a confidence statement or proof that all that difference is selection bias:
inner model selection also uses fewer hospitals than the original procedure.
Different outer folds can choose different penalties; unchanged in the old
full-source choice does not guarantee unchanged in every nested fold.

Examples:

| Target | Ordinary CV change | Nested CV change | Prior encoder-unseen source change |
|---|---:|---:|---:|
| BUN | -0.03800 | -0.03800 | -0.06328 |
| Glucose | -0.04944 | -0.04528 | +0.06079 |
| PTT | -0.02580 | -0.02637 | +0.07841 |
| Troponin I | -0.11796 | +0.03942 | +0.27436 |

The last column is the prior fixed full-source head's two-group/site-equal
change, NOT deployment of this nested fold ensemble on new patients. No new
model was picked using these outer outcomes. These are current-value masked
reconstruction errors, not future forecast changes or clinical units.

## Diagnosis And Repair Boundary

The previous minimum-CV score was not a reliable performance estimate for its
selection procedure. Outer evaluation exposes regressions that minimum-CV
scoring cannot show when unchanged is among the choices. Use outer predictions
to assess that procedure; never report its chosen inner minimum as independent
evidence of benefit. This evaluation correction is now implemented and tested.

However, glucose and PTT retain internal nested gains while reversing on the
previous encoder-unseen source hospitals. Selection optimism alone cannot
account for their transfer failure. Encoder reuse, case composition, hospital
shift and finite support are still not separated. A full encoder cross-fit
would be a distinct, more expensive test; repeating the old independent-head
confirmation alone is not a new remedy. Do not tune a cutoff or delete failed
targets based on the present outcomes and call it promotion.

Two tests pass: changing an outer fold's labels does not change predictions
for that fold; a perfect original head remains unchanged. Parent hashes,
full body-cache digest and original scores reproduce. Per-target resumable
results are private; aggregate artifact contains no patient identifiers.
No runtime weights, calibrators, registry or gate changed. No new promotion.

Evidence: `artifacts/native_reconstruction_nested_readout_cv_20260920.json`.
