# Output retention: real benefit and a clear tradeoff

Seven native whole fits complete. Fixed weight1 target-balanced retention of
initial predictions on source-supported, anchor-observed non-glucose targets.
Protocol: `NATIVE_RETENTION_REPAIR_PROTOCOL_20260911.md`.
Aggregate: `artifacts/native_retention_repair_20260911.json`;63bindings verified.
Original glucose predictions reproduce before fitting; two loss tests pass.

| Hospital MAE (native units) | Glucose-only | Retention | Initial |
|---|---:|---:|---:|
| Glucose | 42.4007 | 42.2208 | 46.0267 |
| Magnesium | 0.3168 | 0.2073 | 0.2074 |
| Phosphate | 1.5838 | 0.9821 | 0.9907 |
| Calcium | 0.6294 | 0.5018 | 0.5072 |
| Hematocrit | 4.3128 | 3.5721 | 3.5940 |
| Potassium | 0.4765 | 0.4364 | 0.4391 |
| BUN | 5.2409 | 7.1220 | 7.2587 |
| Creatinine | 0.3092 | 0.3962 | 0.3928 |

Retention versus glucose-only hospital significant gain/harm counts:
Mg7/0, phosphate7/0, calcium5/0, Hct6/0, K5/0; BUN0/7, creatinine0/7.
Glucose1/0; compared with own-history glucose1/1, not stable whole benefit.
New-source glucose34.8408->34.2384, gain1/7,harm0; own comparison gain0/harm1.
Source Mg gain7/7, but BUN harms7/7 and creatinine harms5/7.

This intervention shows output retention can reverse particular downstream
regressions without significant primary harm in this test. It does not prove
primary noninferiority, complete repair, or all-organ preservation. It also
suppresses useful changes from poor initial BUN/creatinine predictions.
Initial references were deliberately not asserted as validated experts.

The next repair must distinguish preservation of useful knowledge from freezing
initial mistakes. Any reference-competence selection must be source-only and
prespecified, not choose targets from these hospital results. Measured auxiliary
supervision and frozen-output retention are different objectives; neither
uniform policy has established full-body success.

These are median patient MAEs across overlapping seeds. Multiple target tests
are exploratory, not multiplicity-adjusted promotion. BUN/creatinine remain
worse than persistence even before retention. No new clinical or causal state
claim, no untouched database, no runtime/interval/registry changes.
