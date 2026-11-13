# Readout CV Versus Encoder-Unseen Source Hospitals

## Fixed Comparison

Use the completed anchored-readout experiment, without fitting another model
or selecting a new penalty. Compare chosen-minus-original MAE under identical
group-equal, then hospital-equal weighting. The original report's patient-equal
macro MAEs use a different aggregation; they must not be directly substituted.
Check parent dependency hashes before deriving this aggregate-only report.

Of 33 targets, source-fit hospital CV chose a changed head for 24 and left nine
unchanged. The encoder and original head had already trained on those inner-CV
hospitals. Only the additional correction was fitted without each held fold.
The outer source-selection hospitals were absent from encoder fitting, but
were previously used for checkpoint selection and repeatedly inspected. They
are not an untouched final lockbox.

## Results

For the 24 changed heads:

| Quantity | Result |
|---|---:|
| Mean anticipated CV MAE change | -0.0309641 |
| Mean realized source-held-hospital MAE change | -0.0014300 |
| Targets with realized mean improvement | 18 |
| Targets with realized mean regression | 6 |
| Targets improving numerically in both source groups | 12 |
| Spearman correlation of anticipated and realized changes | 0.22435 |

Values are normalized-error changes, not percentages or clinical units.
Across-target correlation is descriptive: targets are dependent, sample
support differs, and CV selected a minimum among candidates. No significance
or causal interpretation is attached to the correlation.

The six negative transfers in the group/site-equal average are glucose, INR,
PTT, respiratory rate, troponin I and WBC. Glucose, PTT, respiratory rate and
troponin I worsen in both groups. This list is diagnostic, NOT permission to
remove those heads retrospectively and claim a validated new selector.

The existing incremental gate result is unchanged: no selected-group passes,
broad BUN only, and zero common passes. Twelve numerical two-group gains are
not twelve promotions.

## What This Locates

Internal readout CV substantially overstates aggregate benefit in this run.
The remaining failure is not explained solely by different evaluation weights:
the weighting is matched here. Nor is it simply universal absence of signal:
18 changes retain the favorable mean sign, but gains are offset by regressions.

This cannot separate three explanations: selection optimism from minimizing
noisy CV scores, representations/anchor heads already exposed to inner-CV
hospitals, and changed hospital/case composition. A new prediction head would
not resolve that ambiguity.

The next valid selection repair must separate encoder fitting from readout
selection at the hospital/patient level, or cross-fit the entire encoder and
head inside each fold. Compare the full selection procedure on the same outer
patients, with all transforms and checkpoint choices kept inside training
roles. Reused source/MIMIC/NWICU results remain development evidence; a genuinely
untouched final cohort is still needed. Do not tune penalties or exclusions
against this diagnostic's failed targets.

One focused test verifies equal-group/site aggregation is not replaced by
patient-count weighting. Audit exit 0; release validation remains 41 catalog,
27 move, 14 fallback. No weights, intervals, registry or serving changes.
This is masked-current reconstruction, not future forecast validation.

Evidence: `artifacts/native_reconstruction_readout_cv_transfer_20260920.json`.
