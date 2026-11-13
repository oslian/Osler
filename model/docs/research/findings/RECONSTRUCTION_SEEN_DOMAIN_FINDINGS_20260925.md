# Matched Seen/Unseen Domain Correction Diagnosis

All47targets/five calibration roles processed. Query people are fixed20%
hash-priority within component, shared across targets. Seen and unseen fits
exclude ALL query patients, use equal target-specific training-row counts,
and score exactly the same query people. Huber and preprocessing unchanged.
No outer evaluation or new hyperparameter/model selection.

## Results And Coverage

75of235target-folds have all query-component fits supported/converged with
actual same-component training exposure, and>=3querycomponents of>=30people.
These75are the ONLY full comparisons. The other160are not counted as successes
or biological failures. Across component queries:1029paired,259incompletefit,
81no_seen_exposure. A small component whose only person is held out cannot
support a same-domain new-patient contrast. Do not silently delete that group
to make a target-fold complete.

Within75complete comparisons, numeric Huber:

| Versus unchanged frozen model | Number |
|---|---:|
| Both seen and unseen improve |17|
| Seen improves, unseen does not |7|
| Unseen improves, seen does not |4|
| Neither improves |47|

Seen numeric correction has lower MAE than unseen in60/75. Thus exposure to
other people in the query component often helps this correction, but does not
usually make it beat the unchanged model. Counts describe means, not corrected
significance tests or independent replication.

Example raw-unit, component-equal MAE for identical query sets:

| Target/fold | Unchanged | Seen numeric | Unseen numeric |
|---|---:|---:|---:|
| Albumin0 |0.34121|0.34469|0.38925|
| Glucose0 |41.12382|42.43291|44.69432|
| Creatinine0 |0.66250|0.65792|0.65627|
| MAP0 |12.50765|11.79530|11.96258|
| MAP1 |14.60452|15.45831|15.59245|

## What Changes In The Diagnosis

The explanation "the correction is useful except at unseen hospitals" is not
sufficient for these complete comparisons. There is also new-patient failure
despite seen-domain exposure. This points toward residual overfitting, limited
conditional information, or misspecification, rather than just external site
harmonization. It does NOT isolate which one is the unique root cause.

Matched row counts do not equalize component composition, weighting, input
support, or case mix; those change along with domain exposure. Therefore60/75
is not a clean causal hospital effect. One fixed split and incomplete160groups
limit generalization. This is current masked reconstruction, not forecasting.

Do not repeat another domain-only normalization as if it necessarily fixes the
model. A further correction must establish incremental generalization to new
patients WITHIN the source domains before asking it to transfer across them.
The unchanged shared state remains the reference; the latest corrective heads
are not deployed. Two partition tests pass (outcome/order independence and
equal-size, same-query, patient-disjoint exposure contract). No runtime,
interval, weights, registry or promotion changed.

Artifact: `artifacts/native_reconstruction_seen_domain_20260925.json`.
