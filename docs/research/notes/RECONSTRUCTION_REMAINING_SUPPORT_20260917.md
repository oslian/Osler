# Remaining Observation Support After Masking

All4637 strict-site evaluation pairs matched hash-verified input shards and
exact truths. Same target/sibling masking helper, all33 real/process MAEs
reproduce to1e-10; six unsupported catalog targets retained. No fitting.

| Target | N | Latest remaining event within1h | Median recent variable count | Within1h real-minus-process MAE |
|---|---:|---:|---:|---:|
| Lipase |62|61|17.5|+0.2018|
| Triglycerides |30|29|16|+0.3875|
| Creatinine |177|177|12|-0.1372|
| Bilirubin |137|137|18|-0.0663|
| Glucose |176|99|3|-0.0726|

Lag is event availability lag, not specimen age. Recent event support does not
certify every historical input is recent or clinically informative. Nonetheless,
all independent inputs being stale/missing is not a sufficient explanation for
lipase/triglyceride failures: almost all have recent input but still harm.

Glucose differs:60patients have latest event1-6h old, mean increment-0.0696;
17patients have latest event>6h old, mean increment+0.1043. These are descriptive
small subgroup means, not significance or permission to pick a deployment
cutoff using evaluation. Any source-trained policy requires separate evaluation
and must retain older-input patients in reporting. No patient was excluded.

This narrows the failure hypotheses but does not prove identifiability, a unique
cause or repaired accuracy. No weights, runtime or promotion changed.
Artifact: `artifacts/native_reconstruction_remaining_support_20260917.json`.
