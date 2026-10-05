# Additional Domains: Frozen Candidate Anchors

## Fixed Selection

All extracted lab and bedside event shards were checksum-verified. Candidates
require the existing variable schema, a finite quality-valid measurement,
entry-time availability, and entry-minus-measurement age between zero and one
hour. Selection is one anchor per patient-target over ALL that person's stays
and sites, not one anchor per hospital or visit.

The deterministic priority is MD5 of
`20260921|patient|target|stay|ISO timestamp`. Lowest priority wins, with stable
identity tie-breaking. Numerical values, future outcomes, model errors,
insulin flags, and eventual data roles never enter priority calculation.
Per-site minima are reduced to the same global minimum; this is tested.
MD5 here is a sampling key, not an integrity/security guarantee. SHA256 binds
artifacts. Existing samples/roles are not rewritten.

After subsequent target-and-sibling masking or conflict replay, rejection MUST
NOT trigger another draw. The candidate table remains the denominator.

## Completed Inventory

| Quantity | Count |
|---|---:|
| Candidate people in provenance |73,944|
| People with a proposed anchor |73,812|
| Proposed patient-target anchors |1,810,110|
| Targets represented |47|
| Duplicate patient-target selections |0|

| Target | Proposed pairs | Components with at least 30 people |
|---|---:|---:|
| Creatinine |70,012|30|
| BUN |69,920|30|
| Glucose |71,314|30|
| Albumin |51,459|29|
| Total protein |49,802|27|
| Heart rate |67,437|28|
| MAP |66,319|28|
| TSH |10,760|19|
| Cortisol |1,370|11|

These counts can exceed the prior fresh-unambiguous support counts: candidate
selection deliberately precedes full source-version/latest-draw conflict
replay. They must not be reported as accepted labels or masked-history support.
Every person's proposed anchors remain within one full-history connected
hospital component. No fit/calibration/test assignment has happened yet.

## Verification And Remaining Work

Four tests pass: row-order/duplicate invariance, numerical-value independence,
site-local versus global selection equivalence, and empty-site schema.
The complete build exited normally; query hash, duplicate prohibition and
patient-component consistency were independently checked. Private candidate
tables remain ignored. Only the checkpoint variable schema is read, no learned
encoder weights or scaler are used to choose anchors.

Next replay exact known event versions at these fixed anchors, mask the target
and all related siblings, reject empty independent histories, and count accepted
support by target/component. This has NOT been completed in this artifact.
Do not shortcut the existing source-conflict policy or substitute an easier
anchor. The selected-cohort support limitation from
`RECONSTRUCTION_ADDITIONAL_COHORT_SUPPORT_20260921.md` still applies.

No training, accuracy result, promotion, interval or runtime change.
Report: `artifacts/native_reconstruction_additional_domain_anchors_20260921.json`.
