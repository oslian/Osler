# MAP Active Update Count Audit

The new native-query own/full comparison does not hold recurrent update
cadence constant. `NativeEventFilter` activates when any allowed measurement
or the observed-rate flag is present. Own removes non-MAP masks and treatment
features; full retains them. Identical history indices therefore do not imply
identical state-transition counts.

The input-only audit verifies original training bindings and measures every
eligible query in all seven partitions. No outcome magnitudes or prediction
errors enter the counts. Seed7 patient-equal means per query:

| Role | Own updates | Full updates | Extra updates |
|---|---:|---:|---:|
| Fit | 25.64 | 74.04 | 48.40 |
| Tune | 24.74 | 73.75 | 49.01 |
| New source patients | 26.08 | 75.22 | 49.14 |
| Held-out hospitals | 16.69 | 54.19 | 37.50 |

At least one extra update occurs in 99.74% of source evaluation queries and
99.15% of hospital queries under patient-equal weighting. These are not
percentages of unique patients. Padding activates neither arm; all own-active
events remain active in full.

This quantifies a known confound on the newly rebuilt task, not a new proof
that cadence causes the accuracy gap. The earlier MAP update-scope replay
already failed to establish a repair, and changing evaluation masks alone
would mismatch training. Removing other-organ observations is not a solution
to the requested shared-state model. A mechanistic distinction still requires
a training-matched content/cadence control; this audit does not promote one.

Evidence: `artifacts/map_native_update_counts_20260912.json`.
No training, release weight, interval, registry or runtime changes.
