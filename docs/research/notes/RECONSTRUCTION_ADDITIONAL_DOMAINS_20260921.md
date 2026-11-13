# Additional Development-Domain Inventory

## Purpose And Exclusions

Address the latest limited-domain calibration failure by identifying additional
source candidates BEFORE looking at target values or prediction outcomes.
Read the full local eICU patient/stay/hospital crosswalk. Exclude all known
original role patients, query patients, current crossfit patients, broad-source
and confirmation provenance, and current ancestral source patients. Include
all hospitals in their complete raw histories in the exclusion set.

Then compute connected hospital components using ALL raw patients, not merely
the bounded research cohort. If any component touches an excluded site/person,
exclude that whole component. This prevents an apparently new hospital entering
through transitive cross-hospital patient bridges.

This is deliberately conservative pipeline provenance, not proof of no exposure
anywhere in the project's entire past. No existing test roles are reassigned.

## Counts

| Quantity | Count |
|---|---:|
| Raw patients |139367|
| Raw hospitals in local patient table |208|
| Raw whole-history hospital components |57|
| Known patients excluded |22011|
| Known complete-history hospitals |71|
| Patients directly disjoint before transitive exclusion |86447|
| Further excluded through connected-domain bridges |12503|
| Candidate patients |73944|
| Candidate stays |110083|
| Candidate hospitals |105|
| Candidate independent components |38|

The208count is a direct count of this local table's hospital IDs, not a claim
about a publication's study-site inclusion criteria.38new candidate components
are distinct from the8components computed inside the previous bounded source
subset; both analyses use complete patient histories but different patient
universes for graph connectivity.

## What Can Be Repaired Next

The local download contains substantially more candidate domain diversity than
the currently extracted training subset. Thus repeating the old one-component
calibration is not the only available option, and no new download is necessary
to investigate expansion.

These are NOT yet valid reconstruction examples. Next apply the existing unit,
plausibility, result-availability, conflict and target/sibling-mask contracts;
count actual per-target patient/component support. The selected-cohort definition
must be independently preserved or explicitly revised before a new comparison,
not assigned according to which patients/models perform well. Patient count
alone does not establish the earlier3cal/3fit per-group support requirement.

No model trained, no cohort roles assigned, no new external-lockbox claim, no
promotion or runtime change. Candidate provenance is private/ignored parquet;
aggregate report contains counts and anonymous component ordinals only.
Three tests pass: transitive exclusion, person-based component exclusion and
order-invariant candidate selection.

Report: `artifacts/native_reconstruction_additional_development_domains_20260921.json`.
