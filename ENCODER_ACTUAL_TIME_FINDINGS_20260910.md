# Actual-Time Encoder Training Result

All28 matched2250fit/136tune64epoch fits completed with actual5-7h source
supervision and actual-time source-tune selection. Same original data hashes,
optimizer, sampling and initialization as nominal training. Real context
encoder training, not just fixed-weight querying. Protocol:
ENCODER_ACTUAL_TIME_PROTOCOL_20260910.md.

## Primary Fixed6h Query

Median patient-equal MAE mg/dL; original identical5-7h truth tolerance:

| Model | Hospital nominal-trained | Hospital actual-trained | MIMIC nominal-trained | MIMIC actual-trained |
|---|---:|---:|---:|---:|
|Own identity|36.310|36.319|31.209|31.061|
|Whole identity|36.796|36.799|32.037|32.067|

Paired significant improvement/harm versus nominal-trained samearm:
own1/7improve,2/7harm hospital;4/7improve,2/7harm MIMIC.
Whole0/7improve,1/7harm hospital;0/7improve,5/7harm MIMIC.
Whole-vs-own remains0/7both. Tiny median shifts do not negate per-seed harms.

## Separate Retrospective Actual-Time Query

Actual-trained models queried at actual observed label time: own36.296hospital,
31.052MIMIC;whole36.764hospital,32.022MIMIC. Descriptive scores only; not a
matched actual-query improvement comparison against the old model, and not
forecasting the next measurement time. No promotion from this alternate task.

## Diagnosis and Limits

Within this5-7h tolerance, actual elapsed-time training does not repair the
whole-context gap. This moves the question beyond the prior fixed-weight
negative diagnostic. It does not establish that precise time never matters,
especially for wider horizons or other variables. Source-tune selection also
uses the actual-time query, so training and selection semantics change together.

Candidate_only. No runtime weights, registry, intervals or deployment changes.
No conformal, retention, causal or fresh-lockbox validation.303tests pass.
All predictions/weights private. Aggregate artifacts under artifacts:
encoder_actual_time_20260910.json, encoder_actual_time_comparison_20260910.json,
encoder_actual_time_strata_20260910.json.
