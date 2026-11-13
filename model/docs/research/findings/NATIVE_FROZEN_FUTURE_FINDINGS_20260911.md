# Frozen shared-future training result

14 matched32epoch fits complete. Shared latent_predictor weights exactly
preserved in all fits; encoder and private paths train, EMA updates normally.
Same source roles, labels, sampling and source-tune selection as protocol.

| Glucose MAE | Source evaluation | Hospital |
|---|---:|---:|
| Original whole | 34.8408 | 42.4007 |
| Frozen-future whole | 33.9582 | 42.7113 |
| Frozen-future own | 34.1927 | 42.1914 |
| Original own | 33.9762 | 41.4025 |

Whole versus original whole gain1/7source,0/7hospital (hospital harm1/7).
Whole versus matched own0/7source(harm1),1/7hospital; versus original own0/7both.
Frozen-future own versus original own hospital gain1/7,harm5/7. Persistence
still beaten7/7both roles by whole; not an incremental repair.

Actual other-target hospital outcomes versus original whole: Mg gain3/harm1,
phosphate4/1,Ca4/3,BUN3/2,creatinine3/0. All supported targets were evaluated;
no consistent all-target preservation. Source-evaluation rollback findings did
not establish trained repair: co-adaptation changes the effect of freezing.
Do not infer that this module alone causes all failures or freeze it in runtime.

No runtime, calibration, new lockbox, causal or promotion change. Base-model
and cohort/pretraining limitations unchanged; overlapping seeds exploratory.
Artifacts: `artifacts/native_frozen_future_20260911.json` and
`artifacts/frozen_future_retention_20260911.json`.
