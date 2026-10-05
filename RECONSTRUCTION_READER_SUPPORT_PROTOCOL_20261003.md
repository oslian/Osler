# E24: frozen full-fit versus bounded-fit raw readers

## Question and scope

E23 direct organ readout failed, but its raw control was also weaker than
native. Test whether an existing full-fit reader recovers the raw advantage
on exactly the same E23 selection queries. This is masked-current inference,
not forecasting, organ repair, runtime promotion or untouched external testing.

## Fixed comparison

- Reuse E3 raw265 HGB100 and E23 raw265 HGB100 checkpoints, no training.
- All five E23 folds and all eligible E23 variable-folds; inventory all53.
- Same frozen native point, query patient, masked target, normalized truth,
  input raw265 and source component. No selection-dependent exclusions.
- Verify both fit and selection sampled features against full caches;
  sampled fitting patients must be a subset of full fitting patients.
- Full fitting identities must be disjoint from selection and assigned fit.
- Verify bound metadata, checkpoint hashes, parameters and saved predictions.
- Stop on mismatch; do not silently intersect incompatible populations.

## Assessment

Three contrasts: full_raw-small_raw, full_raw-native, small_raw-native.
Average absolute loss over repeated folds per patient before patient-equal
and component-equal means. Paired bootstrap5000, one-sided paired t statistic
via the existing paired_summary helper. Descriptive screen requires at least
30 distinct patients and3 components, CI95 upper<0 and p<.05/(53*3) at both
levels. This is not the release gate. Keep unsupported targets explicit.
Report all three arms, support counts and target-fold directional results.

## Interpretation and stopping

Full-fit superiority would show E23's bounded raw control is inadequate to
represent full-fit raw performance. Larger fitting support also changes
population composition and component weights; it does not isolate sample
count as a causal factor. No difference refutes this particular explanation
on the fixed query set, not the existence of organ information.
Do not compare E3 body64 with E23 body64+159 as a support-only experiment.
No parameter search, no organ head retraining, no runtime changes.
Private rows and model files stay ignored; publish aggregate evidence only.
