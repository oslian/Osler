# Same-cadence trained bedside control

Fourteen fits completed: lab control and bedside-support-only, seven original
partitions. All five bedside numbers are neutralized in source-centered
coordinates during BOTH training and evaluation, while masks, ages, event
times and capacity remain as in the preceding source-rescaled real-number
trial. Labels, partitions, initialization and32epoch settings unchanged.
All seven lab prediction controls reproduce exactly. Current run bindings and
historical prediction hashes checked; historical source hashes not rewritten.

| Median patient-equal glucose MAE | Lab | Real numbers | Support-only |
|---|---:|---:|---:|
| New source patients |34.884272|34.618132|34.875907|
| Held-out hospitals |42.079848|42.328418|41.658324|

Paired95%CI gains/harms out of7:
- Support-only vs lab: source1/0; hospital2/0.
- Real numbers vs support-only: source1/0; hospital0/1.
- Median paired real-minus-support effect: +0.02404source,+1.08453hospital.

Differences of medians are not median paired differences; use the paired
artifact for significance. Overlapping seeds are not independent replications.
Neither version establishes consistent new-context benefit. Dense update
cadence is not uniformly harmful: the trained support version has some gains.
Real numbers do not reliably add to it under this model. This localizes the
remaining dependency but does not prove all predictive signal is hospital
workflow: observation support can also reflect severity and patient state.

No promotion, runtime, calibrator or release change. This is one glucose task,
not all organs; it does not prove raw bedside values contain no information.
Further work should separate whether the numerical history encoder loses
useful within-patient changes from whether those changes add predictive
information beyond own laboratory history, using source-only development.

Artifacts: native_bedside_support_training_20260911.json and
native_bedside_support_comparison_20260911.json under artifacts.
