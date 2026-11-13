# Additional Domains: Bedside Observations

## Question And Contract

Can the unassigned additional eICU development domains supply availability-safe
bedside histories alongside the already extracted laboratory observations?
Reuse the established nursing label/unit mappings, explicit entry offsets,
canonical source-event identity, and plausible-value flags. Entry must be at or
after measurement. No periodic-vital channel with unverified availability is
substituted. Invalid numerical values remain quality-flagged, not imputed.

Candidate provenance and raw inputs are checksum-bound. The 105 hospital event
shards and patient support tables remain private and git-ignored. No training,
calibration, or test roles have been assigned to the additional candidates.

## Extraction Results

| Stage | Count |
|---|---:|
| Candidate people |73,944|
| Numeric mapped raw nursing rows |40,865,099|
| Canonical time-valid events |40,775,882|
| Canonical quality-valid events |40,714,118|
| Fresh, unambiguous, quality-valid event rows |35,697,038|
| People with fresh bedside observations |70,770|

Fresh support requires age 0-1 hour. For simultaneously available values,
choose the latest draw and reject conflicting latest values BEFORE selecting
quality-valid rows. An invalid latest draw cannot revive an older valid draw.
Distinct equal-valued remeasurements retain their event identities.

| Variable | People | Components with at least 30 people |
|---|---:|---:|
| Heart rate |67,437|28|
| MAP |66,224|28|
| O2 saturation |63,037|28|
| Respiratory rate |62,569|27|
| Temperature |69,937|28|

Union with the checksum-bound laboratory support table: 73,812 unique people,
47 distinct variables; 69,775 people have both lab and bedside support. These
are observation counts, NOT independent target-masked training pairs, future
label support, model accuracy, or newly promoted cells.

## Reproduced And Repaired Extraction Failure

The first extraction stopped on a hospital with no matching nursing rows.
The existing helper's DataFrame.apply preserved string dtype on an empty
frame, so the boolean reduction failed. The subsequent empty numeric frame
also retained string dtype and failed isfinite. Explicit per-column boolean ID
validation and float numeric conversion repair the empty-frame contract.
The empty-hospital regression reproduces both issues and now passes.

This is a data-pipeline boundary bug, not evidence that it caused historical
model generalization failures. The interrupted run produced no final report.
Its generated raw cache was removed and extraction reran to normal completion
with current dependency hashes. All 105 final shard hashes were verified.
Historical research artifacts are not silently rebound to the changed helper.

Six new tests plus ten existing bedside/lab tests pass (16 total), including
real-zero preservation with invalid-vital marking, equal remeasurements,
availability delay, impossible entry rejection, invalid-latest behavior, and
missing provenance rejection.

## Decision

Bedside coverage is no longer the missing extraction step in this new pool.
Next build independent target-and-sibling-masked histories, verify nonempty
support by target/cohort/component, explicitly qualify the selected cohort,
and assign source/calibration/test roles without using model errors. Increased
observation support does not establish transferable calibration or solve the
remaining representation failure. No release weights, registry, forecast,
interval, promotion gate or causal claim changed.

Aggregate report:
`artifacts/native_reconstruction_additional_domain_bedside_20260921.json`.
