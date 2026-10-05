# Recorded Pressor Context

Private snapshot cache built for all55,468queries from145,510matched source
rows, retaining invalid-rate records so they block stale values. Five named
drugs, three separate rate dimensions, rate masks, recorded ages and status.
No drug-equivalence summation, inferred delivery or active-duration claim.

Among33,477MAPeligible queries, latest compatible uncontested recorded support:
norepinephrine8,125; epinephrine860; phenylephrine1,793; dopamine733;
vasopressin2,082. Counts overlap. Lower than the inventory because latest
invalid/contradictory rows now block older compatible records. Across drug-query
cells there are63conflicting-latest and145invalid-latest cases.

Eight tests cover minute units, documented zero, prefix invariance, invalid
latest, same-time contradiction, source-ID conflict, cross-patient isolation,
and identical-valued remeasurement freshness. Records lacking independently
verified availability remain `online_eligible: false`. This is strictly
recorded-time retrospective context, not a legal live-input certification.

No model fitted and no accuracy gain, runtime or promotion claim. Future work
must not silently relabel infusionoffset as known availability. Cache remains
private; aggregate evidence: `artifacts/map_recorded_pressors_20260912.json`.
