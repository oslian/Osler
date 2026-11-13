# Patient-Linked Hospital Support Preflight

## Newly Located Evaluation Boundary

The original reconstruction roles are patient-disjoint, but a patient's complete
cohort stays can span hospitals. The194 original hospital-role patients touch18
hospitals; three also occur in source patients' cohort stays.646source patients
touch those hospitals. Consequently this reconstruction cohort's role name
`hospital` does not establish strictly unseen hospitals across complete histories.
This does not prove every selected reconstruction anchor overlaps, nor diagnose
all historical experiments or active release validation. Prior numerical scores
remain unchanged; the strict hospital-independence interpretation is weakened.

The new preflight uses SciPy connected_components on hospitals linked by the
same patient's stays. All hospitals used by a patient must stay in one component.
Components touching the old hospital-role patients are reserved. This leaves893
source patients and quarantines676additional source patients, including indirect
links. Quarantine is for this proposed strict contract, not deletion of original
data or evidence that those patients are invalid.

## Support Without Looking At Errors

The cohort has47 connected hospital components. For each of all39 source-eligible
targets, count accepted patient-target labels per remaining component. A simple
three-role experiment is feasible when two distinct components each have at
least30 target patients and the remaining fitting components have at least30.

33targets meet that per-target feasibility test. Amylase, BNP, LDH, lipase,
triglycerides and TSH do not. This is NOT proof those six are untestable: pooling
multiple small components within a role may restore support. Nor does33mean
one common partition supports all33; no common assignment or training has yet
been selected. The labels' values and prediction errors do not select components.

For bilirubin there are six source components with30or more patients after strict
quarantine; creatinine has seven. This makes a cleaner source-site experiment
possible, but reduces the fitting population substantially. Do not compare a new
strict fit against the old model without accounting for that changed cohort.

## Next Required Implementation

Create one source-only component-level fit/tune/evaluation assignment based on
support counts, retain every target's explicit support status, and verify both
patient and hospital disjointness across all histories. Train matched real and
process controls on that same assignment, preserving all-target reporting.
The original hospital-role patients remain untouched by new fitting. They have
already been inspected and cannot be called a fresh lockbox.

This step fixes evaluation design, not demonstrated numerical accuracy. No new
weights, promotion, calibrators, serving behavior or original artifacts changed.
Machine report: `artifacts/native_reconstruction_site_support_20260917.json`.
