# Independent Selection Support Preflight

Implemented a deterministic site-component split of ONLY the old tune group.
Old fit, evaluation and reserved patients/sites remain exactly assigned. Largest
tune components first, assigned to the smaller role with encoder-first ties.
No target values, counts or prediction errors enter assignment. All original
partition/input hashes checked; complete hospital histories cannot cross roles.

| Role | Patients | Hospitals |
|---|---:|---:|
| Fit |536|11|
| Encoder selection |90|4|
| Readout selection |89|7|
| Evaluation |178|11|
| Reserved |870|21|

28of39catalog targets have at least30patients in each of the first four roles.
Unsupported targets remain explicit: amylase, anion gap, direct bilirubin, BNP,
CPK, fibrinogen, LDH, lipase, triglycerides, troponin I, TSH.

New shortfalls among the prior33 targets:

| Target | Encoder selection | Readout selection |
|---|---:|---:|
| CPK |30|22|
| Fibrinogen |42|18|
| Lipase |28|34|
| Triglycerides |25|16|
| Troponin I |34|13|

This is partition/support repair, NOT an accuracy result. Old checkpoints used
both new selection groups for checkpoint choice and cannot be called selection-
independent after relabeling. New training and separate head selection remain
required. Warm-start pretraining independence is still uncertified; MIMIC/NWICU
and this evaluation are not fresh lockboxes.

The support drop is a real cost of splitting scarce site-qualified labels.
Do not lower the30patient threshold, choose a split using prediction errors or
silently drop the11targets from the project scope. They require more eligible
source labels or a separately fixed validation design, not a fabricated pass.

Three tests verify unchanged other roles, whole-component isolation, row-order
invariance and rejection of invalid/insufficient components. Full build validates
all shard identities and aggregate support. No training, runtime or promotion.

Artifact: `artifacts/native_reconstruction_nested_selection_20260917.json`.
Private split is kept in Git-ignored durable storage.
