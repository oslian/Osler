# E31: same-source, adaptation-unseen diagnostic

E30 showed that E29 has no same-model overlapping fit/selection sources.
This diagnostic adds a third role to the SAME frozen E29 weights. It does not
repeat a bridge/learning-rate/epoch experiment or train any new model.

## Fixed population and comparisons

- All five original folds, all 53 targets in reporting, maximum 1024 people per
  fold with one context each, using the existing E19 hash-order `contexts` rule.
- Eligible keys come from the verified full fitting-feature cache. Exclude all
  E29 adaptation patients, not just their trained target. Require component in
  that fold's E29 fitting components. Never use truth or predicted error to select.
- Record original epoch10 update exposure by mapping `(row,target)` in `seen`
  through the bound scaling `people` vector. Do not compare integers to patient IDs.
- Most eligible people were seen during epoch10 training; preliminary inventory
  found only 69-110 unexposed people per fold before source filtering. Scalers
  were fitted on the original fitting pool, including these people. This is
  NOT a completely unseen-patient test for the whole encoder.
- Compare immediate vs unbridged primarily; preserve delayed, body-control and
  frozen predictions. No raw comparator: full-fit raw readers trained on this
  population, and the question is incremental adaptation generalization.
- Same masks, scaler, anchor, chronological replay, binding proposals, weights
  and inference mode as E29. Verify frozen prediction against the original
  cached value and manual native organ pooling against the native API.

## Reporting and falsification

Use patient-equal mean absolute error within target, then component-equal MAE;
keep repeated folds aggregated per patient and identify occurrences separately.
Report all targets and original-update-exposed/unexposed counts. No new
significance screen, model selection or promotion; this is a descriptive
diagnostic on previously used development data.

For each original E29 fit-gain/selection-reversal target: if same-source unseen
adaptation patients also lose mean advantage, source switching alone is not a
sufficient explanation. If the gain persists, it is compatible with a source
transfer problem but not proof: patient mixture, measurement support and prior
encoder exposure differ. Unsupported targets remain unsupported, not successes.

Do not fit any correction using these outcomes. Stop on hash, role, target mask,
nonfinite prediction or native-parity mismatch. Save private row predictions
locally and only aggregate counts, scores and hashes publicly. Original E29
artifacts and runtime remain unchanged. Successful replay is not model repair.
