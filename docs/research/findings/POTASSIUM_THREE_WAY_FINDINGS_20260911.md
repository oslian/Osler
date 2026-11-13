# Whole-source three-way result

All14 fixed64epoch fits complete. Each seed uses785 fit,262 tune,262 evaluation
patients, disjoint within the trial. Checkpoints use tune only; evaluation
predictions are generated once after selection. No external database read.

| Seed | Evaluation whole minus own MAE | Patient-paired CI95 |
|---|---:|---|
|7|-.004764|[-.017214,.007378]|
|19|.001162|[-.014860,.016081]|
|31|.010207|[-.008254,.028040]|
|43|-.003846|[-.017045,.009120]|
|59|.021427|[.005629,.037864]|
|71|-.005608|[-.020723,.009395]|
|83|-.012529|[-.025582,-.001263]|

Both own and whole beat persistence7/7 on evaluation; whole beats own1/7,
harms1/7, numerically improves4/7. Median evaluation MAE own.377129,
whole.374244,persistence.437907. Tune whole-own0/7. Median across seeds is
descriptive and does not replace within-split paired comparisons.

This repairs the prior source diagnostic's lack of a separate evaluation
partition and incorporates expanded patients into every role. It does not
repair incremental whole-context reliability: failure is still visible inside
source data, not only in cross-database transfer. The reduced fitting count and
different partition population prevent old/new score comparison from isolating
a causal split effect. Selection bias or small old tune support alone cannot
be presented as a demonstrated explanation for the entire context gap.

Patients and seeds overlap across earlier experiments; warm-start pretraining
overlap is not excluded. This is trial-disjoint evaluation, not a new external
lockbox. No multiple-testing-adjusted promotion, calibration, organ-edge
attribution or whole-body shared-state validation is claimed. No runtime change.

Protocol: `POTASSIUM_THREE_WAY_PROTOCOL_20260911.md`.
Artifact: `artifacts/potassium_three_way_20260911.json` (complete=true).
Command: `python -m scripts.train_potassium_matched_encoder --three-way`.
