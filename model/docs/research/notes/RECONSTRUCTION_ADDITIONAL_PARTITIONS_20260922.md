# Additional Domains: Independent Data Roles

## Fixed Design

Reuse sklearn GroupKFold and the project's full-history patient/hospital
connected-component construction. Include all73,944 candidate people, even
those with no accepted anchor. Outcome values, residuals, insulin flags and
target support counts do not determine roles.

Five outer folds; within each development set, the first of four grouped folds
is calibration. Within the remainder, the first of three grouped folds is
selection, and the rest fit. No split retry after inspecting support. Patients
and their complete hospital components cannot cross roles in a fold. Every
person is outer-test exactly once.

Selection may choose a candidate under its future fixed protocol; calibration
may ONLY calibrate that frozen candidate, not choose epochs/models/features.
Outer test cannot choose any of these. This separates previously overloaded
inner roles. It does not turn repeatedly inspected eICU into a new external
lockbox or replace MIMIC/NWICU with untouched evidence.

## Confirmed Counts

| Fold | Fit people | Selection | Calibration | Test |
|---|---:|---:|---:|---:|
|0|29,542|14,862|14,772|14,768|
|1|29,542|14,862|14,768|14,772|
|2|29,542|14,772|14,768|14,862|
|3|29,634|14,772|14,768|14,770|
|4|29,632|14,772|14,768|14,772|

Fit has13-18 independent components; selection5-7; calibration7-8; test5-10.
The role labels and all accepted-example support tables are saved with hashes.
Correction after direct identity verification: folds0/1 share selection but
NOT fit identities. They share29,527fit people and each has15distinct people;
two normalization locations and six scales differ. Equal fit counts do not
permit checkpoint reuse. Folds0/2 do share fit identities, but any future reuse
still requires verified equality of all training inputs/settings. Overlapping
folds must not be called independent model replications.

## Target Feasibility, Not Model Passing

36/47targets have >=30 accepted people in >=3 components in EVERY role of
EVERY fold. These include creatinine, BUN, glucose, albumin, total protein,
heart rate, MAP, INR, PTT, troponin I and the common electrolyte/CBC targets.
This is a prospective support condition, not a changed promotion criterion.

Eleven fail that support condition somewhere:
anion_gap, cortisol, ESR, ferritin, fibrinogen, free T4, LDH, myoglobin, pH,
TSH and uric acid. Their examples remain in the inventory; they have not
failed an accuracy test or been silently dropped to improve a pass rate.

Examples distinguish population size from domain diversity:
- Anion gap fold0 calibration:5,792people but only one component with>=30.
- Cortisol fold0 calibration:302people, only one component with>=30.
- TSH fold0 test:2,402people, only two components with>=30.
- Myoglobin fold0 test and pH fold4 test have no accepted examples.

Do not reshuffle hospitals to conceal these gaps. Unsupported target-role
claims must remain unavailable; fit-only vocabulary checks and corresponding
abstention rules are still required before any model evaluation.

## Verification / Remaining Work

Three tests verify identity completeness, hospital/bridge isolation, exactly-once
outer coverage, order invariance and rejection of insufficient component
inventory. The full run checks original acceptance hashes and counts before
assigning example roles. Private patient assignments remain git-ignored.

No encoder, normalizer, interval calibrator or model selector has been fitted
on this new split. Next compute fit-only observation support and normalization,
prepare exact masked histories, and train the matched physiology/process
controls under a fixed protocol. The old selected-cohort mismatch remains
unresolved; this broad population is not a replacement proof for that cohort.
No runtime weights, registry, intervals or promotions changed.

Report: `artifacts/native_reconstruction_additional_domain_partitions_20260922.json`.
