# Additional Domains: Canonical Lab Extraction

## Completed

Read the local eICU lab table only for the73944previously inventoried candidate
patients. Reuse existing LAB_TO_STATE mapping, canonical unit whitelist,
PLAUSIBLE/EXTRA_PLAUSIBLE bounds and canonical event identity handling. Require
explicit revised offset; use the established greatest(measured,revised) result
availability rule. Reject simultaneous latest-draw conflicting values; fresh
label support requires age0-1h. Preserve real zero and distinct remeasurements.

This does not revise the historical observation contract or assign candidate
patients to train/calibration/test. Existing reserved domains remain excluded.

| Extraction stage | Count |
|---|---:|
| Candidate patients |73944|
| Mapped raw lab rows |13067972|
| Unit/quality/canonical rows |12458563|
| Fresh unambiguous event rows |10070389|
| Patients with at least one fresh lab |72817|
| Schema variables with fresh lab support |42|
| Variables with>=30patients in>=6components |41|

Fresh event rows are not unique patients or independent training pairs.

## Per-Target Domain Support

| Target | Unique patients | Components with>=30patients |
|---|---:|---:|
| Creatinine |70003|30|
| BUN |69915|30|
| Glucose |71312|30|
| Albumin |51454|29|
| Total protein |49794|27|
| Hematocrit |71212|30|
| Bicarbonate |68244|28|
| Potassium |69931|29|
| Troponin I |20075|20|
| TSH |10753|19|
| Cortisol |1369|11|
| INR |47714|28|

These are pooled unassigned-domain support, NOT proof of the previous selected
cohort's eligibility or of>=30in every prospective role. Sparse-marker current
observations do not establish future repeats or forecasting ability.

## Remaining Before Training

Bedside HR/MAP/O2sat/RR/temperature are not lab-table targets; their zero counts
here are not evidence that local data are absent. ck_mb, CRP/hsCRP, ionized Ca,
prealbumin and urine output also have no accepted lab support under this exact
mapping/unit contract. Do not silently invent mappings/units to fill them.

Next extract bedside observations with their existing entry-time and quality
rules; build target-masked histories and verify nonempty independent inputs.
Then preserve/explicitly qualify the selected-cohort definition and construct
patient/component-disjoint roles before fitting. The pool now has substantially
more observed lab-domain support than the old subset, but no accuracy gain,
representation generalization or promotion has been demonstrated.

Private raw/canonical shards and patient-target support are git-ignored, with
checksum-bound indices. Aggregate report contains no patient identifiers.
Three tests pass: conflicting latest draws excluded, newest zero retained,
and stale labels not counted. All extraction completed normally. No model,
runtime, interval, gate or promotion changes.

Report: `artifacts/native_reconstruction_additional_domain_labs_20260921.json`.
