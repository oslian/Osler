# Clock correction accuracy

All14 matched fits complete,50 report bindings verified. Original native
checkpoints re-inferred on original inputs reproduce source evaluation/hospital
MAEs within1e-7 before comparisons. Source roles/labels/query inputs unchanged;
only two history clock coordinates unbounded, including negative offsets.
Aggregate: `artifacts/rate_anchor_jepa_clock_20260911.json`.

| Median patient-equal MAE | Native real | Corrected real | Corrected support | Snapshot whole |
|---|---:|---:|---:|---:|
| Source evaluation | .387409 | .388720 | .388724 | .384183 |
| Source bracket-after | .399871 | .393103 | .393156 | .393479 |
| Hospital | .389022 | .387887 | .387864 | .386973 |
| Hospital bracket-after | .426119 | .425551 | .425549 | .413968 |

Corrected-real versus original-native significant gain/harm: source all1/7,
1/7; source bracket1/7,0/7; hospital both0/7,0/7. Versus corrected support:
source all1/7,1/7; bracket1/7,0/7; hospital both0/7,0/7. Versus snapshot whole
gain0/7allgroups, hospital bracket harm1/7. Versus own gain5/7source all,
6/7source bracket,1/7hospital both. Persistence benefit7/7allgroups remains.

Removing clock collisions is an input-representation repair, but not a stable
accuracy repair. The hospital median improvement .389022->.387887 is small
and no paired real-model hospital CI establishes benefit. The time defect is
not supported as the principal explanation under this finite matched test.
Do not confuse differences of medians with paired effects or overlapping seeds
with independent replications. Keep this as a completed negative/partial result,
not an untested clock-repair proposal.

Remaining boundaries: event time provenance is still retrospective for rates;
warm-start pretraining independence is unverified; no new external database,
conformal, retention or persistent-organ architecture gate. No runtime changes,
promotion or causal claim. Candidate checkpoint metadata records unbounded
history clocks; query context retains the original matched inputs.
