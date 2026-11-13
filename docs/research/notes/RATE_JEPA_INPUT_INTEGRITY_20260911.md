# Rate-query comparison input integrity

Read-only verification on the current caches completed:

- Body array hash matches the body-input report.
- Query and label hashes match the original rate-query report.
- All seven partition hashes match the original partition manifest.
- Fit/tune/evaluation/hospital patient-set intersections are zero in every seed.
- Duplicate stay/query timestamps among paired queries: zero.

This supports the matched-current-data interpretation of the own/whole/support
contrasts. It does not establish original warm-start pretraining independence:
the current disjoint partitions do not certify unseen patients for the inherited
checkpoint. It also does not verify clinical availability of infusion records.
Keep these qualifications on all reported source and hospital results.

No fitting, weights, runtime, intervals, or promotion changed. These checks rule
out inspected cache mismatch and current partition overlap, not forecast error
or the unique cause of numerical-increment attenuation.
