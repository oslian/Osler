# Expanded sequence diagnostic: support repaired, incremental gain not repaired

Completed 21 fixed-capacity fits on frozen seven patient partitions; 1822 fit
patients per seed, 608 source evaluation, 567 hospital-heldout. Supported
sequence evaluation now269-281 patients and184 hospital patients, not1-10.
Cross-hospital patient source rows purged. No tuning on evaluation outcomes.

Median patient-equal MAE across seeds:

| Role/group | Own history | Plus insulin | Shuffled | Persistence |
|---|---:|---:|---:|---:|
| Source/all | .424584 | .422354 | .425334 | .529313 |
| Source/sequence | .402279 | .386444 | .398169 | .436196 |
| Hospital/all | .365562 | .366079 | .365974 | .464061 |
| Hospital/sequence | .336886 | .341000 | .336804 | .400376 |

Real-vs-own significant gain1/7source all/sequence,0/7hospital all/sequence.
Real-vs-shuffled gain0/7source all,1/7sequence,0/7hospital both. Hospital sequence
real-vs-own numerical harm7/7, significant harm1/7; not stable added value.
Real-vs-persistence hospital7/7both groups. Forecasting itself works; the
additional treatment-summary correction does not improve this comparator.

This weakens the hypothesis that insufficient evaluation support alone explains
the previous failure. It does not establish no treatment information, adequate
power for every small effect, or a unique physiological cause. Cohort and anchor
composition also changed. Fixed summary trees are an information diagnostic,
not end-to-end shared JEPA. Donor placebo availability is not exactly matched.
Treatment event time is not independent knowledge-time, no integrated delivered
dose or other simultaneous therapies. No new lockbox, interval or runtime change.

Artifact: `artifacts/expanded_insulin_potassium_20260911.json`.
Do not repeat simple sample expansion as an untested guaranteed solution.
