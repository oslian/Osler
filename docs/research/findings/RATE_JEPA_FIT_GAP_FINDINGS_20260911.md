# Frozen numerical increment across patient roles

Seven saved support models re-inferred; own/whole cached predictions hash
verified. All evaluation/hospital MAEs reproduce to1e-7. No fitting or model
selection. Aggregate: `artifacts/rate_jepa_fit_gap_20260911.json`.

| Median patient-equal MAE | Own | Whole | Support-only |
|---|---:|---:|---:|
| Fit | .418641 | .353430 | .377526 |
| Tune | .410518 | .390664 | .395441 |
| New source patients | .417130 | .384183 | .394297 |
| Held-out hospitals | .402915 | .386973 | .400350 |

Whole versus support significant gain/harm: fit5/7 and1/7; tune1/7 and0/7;
new source1/7 and0/7; hospital2/7 and0/7. Whole versus own gains7/7fit,3/7tune,
5/7source,1/7hospital. Support versus own7/7fit,2/7tune,5/7source,0/7hospital.
These are descriptive paired bootstrap comparisons, not independent seed
replications; fit and tune are not unbiased tests. Unequal role sizes affect
significance. Differences between median MAEs are not median paired effects.

The numerical increment already attenuates on new source patients, before
hospital shift. This supports partial fit-to-new-patient generalization loss,
not a unique overfitting mechanism. Unlike the older glucose experiment, the
latest whole model still has lower median error than own in every role.
Do not claim wholesale reversal or absence of shared signal.

There are two distinct limitations: numbers have a weaker incremental benefit
on unseen source patients, while support-only benefit further loses hospital
significance. Support, case mix, optimization, and measurement practices are
not individually isolated by this audit. Previously tested dropout, extra
epochs, and scalar rescaling are not new untested repairs.

No runtime, intervals, promotion or self-learning update changed. This narrows
failure location but does not certify a successful repair.
