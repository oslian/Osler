# Multi-Domain Calibration Feasibility

## Question

Can the current fixed source subset support a stronger domain-calibration
experiment without stealing outer-test patients or relaxing patient/hospital
independence? This is preparation for a repair, not another performance claim.

Use existing source manifest identities/support and the complete raw patient
hospital crosswalk. No numerical measurements, prediction errors or outcomes
choose a partition. Preserve each outer test role. Enumerate ALL subsets of
remaining independent hospital components leaving at least3components for
both calibration and fitting (including calibration sets larger than3).

Proposed feasibility requirement: each target/group must have>=30patients in
at least3calibration AND3fit components. This is a prospective diagnostic
design requirement, NOT a newly imposed promotion threshold or a theorem that
fewer hospitals can never validate a model.

## Actual Support

Current subset:10737patients,14hospitals,8whole-history connected components.
Each existing inner calibration role contains just ONE independent component
(one or two physical hospitals). The completed calibration experiment therefore
did not test whether corrections could be estimated consistently across several
independent calibration domains.

| Target | Selected patients | Selected components with>=30 | Broad patients | Broad components with>=30 |
|---|---:|---:|---:|---:|
| Creatinine |530|3|9662|8|
| BUN |529|3|9664|8|
| Glucose |513|3|9541|8|
| Albumin |295|3|6002|8|
| Total protein |262|3|5712|8|

No target has six sufficiently supported components in BOTH groups, even
before retaining an outer test. Enumerated candidate role partitions per outer
fold:70,70,70,20,0. Zero of165target-fold combinations meet the proposed full
role-support requirement. Fold4 already has only5development components, so
it cannot even allocate3+3regardless of patient numbers.

## Consequence

The latest repair's calibration-domain limitation is real and localized to
the CURRENT SUBSET and its splitting design. This is not a claim that the
full downloaded eICU database has only14hospitals, that no other design is
possible, or that domain support explains every prediction failure.

Do not launch the same full training again while expecting a new scalar
calibration head to provide cross-domain stability. Before further fitting,
inventory additional source-eligible hospitals/patients under the SAME
observation and cohort contracts, excluding every reserved role and using
complete patient history to avoid cross-hospital patient leakage. Support
expansion must be based on cohort/provenance, never which hospital scored well.
Keep MIMIC/NWICU as previously inspected development validation, not new lockboxes.

Four tests pass, including exhaustive3/4calibration choices with7components,
disjointness, insufficient-domain rejection, and distinguishing patient count
from domain count. An initial implementation enumerated only3-component
calibration sets; it was corrected before reporting these results, and the
regenerated report covers all legal component subsets. No cohort reassignment,
model training, runtime, interval or promotion changes.

Evidence:
`artifacts/native_reconstruction_calibration_domain_support_20260921.json`.
