# Expanded Confirmation Event Cache

Completed extraction for3433additional source-confirmation patients/4131stays.
Full original patient histories lie within the fixed readout-confirmation role;
current cohort and current ancestral patients/sites remain excluded.

Durable Git-ignored cache contains2,154,036canonical laboratory/nursing events.
Existing extractors verify units, source versions and available/measured time.
Nursing invalid finite observations are retained with quality_valid=false rather
than silently replaced. Lab filtering follows existing canonical-unit/plausibility
policy. All source IDs, units, times and quality remain in private files only.

| Bedside target | Canonical events | Quality-valid events | Stays |
|---|---:|---:|---:|
| Heart rate | 360483 | 360426 | 4096 |
| MAP | 314290 | 314182 | 4089 |
| O2sat | 358180 | 358087 | 4082 |
| Respiratory rate | 362754 | 361303 | 4094 |
| Temperature | 141042 | 141007 | 4043 |

This closes the prior laboratory-only inventory gap for the five bedside targets.
It does not establish final target-by-site support: reconstruction still must
handle simultaneous conflicts, target/sibling removal, independent surviving
history and one-per-patient-target sampling. No model predictions or fitting
occurred. This is reused-source expansion, not new external lockbox.

Fourteen raw replay and bedside regression tests passed. Private paths checked
Git-ignored. Aggregate report binds inputs, extractor code, checkpoint schema and
cache files: `artifacts/native_reconstruction_expanded_confirmation_events_20260917.json`.
No release weights, registry, intervals, promotion or serving changes.
