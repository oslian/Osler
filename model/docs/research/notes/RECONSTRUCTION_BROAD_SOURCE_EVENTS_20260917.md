# Broad Source Event Extraction

Repaired the inherited treatment-selected source boundary at the extraction
stage: additional patients are chosen solely by full-table patient/site roles,
not insulin rates, target outcomes or prediction errors. Existing current cohort,
ancestral patients/sites, confirmation, evaluation and reserved roles excluded.

| Role | Patients | Canonical events | Quality-valid events |
|---|---:|---:|---:|
| Fit | 10201 | 5721709 | 5717062 |
| Encoder selection | 3323 | 2487270 | 2485141 |

Fifteen hospital shards use the same lab mapping/units/availability and bedside
identity/time/quality functions as expanded confirmation. Invalid finite nursing
events stay explicitly quality-marked. Streaming raw SQL into private Parquet
and per-hospital canonicalization bounds memory. All individual data Gitignored;
aggregate report binds sources, code, checkpoint schema and event index.

Initial attempt failed before event output due to DuckDB COPY placeholder
binding order; only provenance existed. Terminal failure verified, original
provenance preserved under failed_copy_binding directory. Replaced COPY with
relation.write_parquet, reran successfully to completion. No clinical/model
failure inferred. Fourteen shared replay/bedside regression tests passed.

This is event extraction, not accepted reconstruction samples, encoder retraining
or evidence of accuracy repair. Next construct the same one-per-patient-target
fresh block-masked examples and freeze a comparable training protocol. Broader
source changes both N and composition; an equal-size source comparison is needed
to isolate composition rather than attributing every gain to cohort repair.

Evidence: `artifacts/native_reconstruction_broad_source_events_20260917.json`.
No runtime, interval, weights or promotion changes.
