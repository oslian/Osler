# Where Point-Teacher Imitation Loses Real Accuracy

## Scope

Completed retrospective decomposition of the fixed state-teacher experiment:
47 targets, all five source folds, 1,797,884 query occurrences. No new fitting,
predictions, external queries, clipping, feature selection or runtime changes.
These are masked CURRENT-value estimates, not future trajectories. Repeated
fold occurrences are not independent patients. Average losses within each
patient first, then equally across patients or provenance components, and
finally equally across all 47 targets. Component does not necessarily mean
hospital. All reported losses use the existing target normalization.

Let B be the original body readout, T the masked-raw teacher, S its state-only
student and Y the measured truth. The diagnostic partitions the change
|S-Y|-|B-Y| exactly. Positive contributions are harm; negative contributions
are benefit. Categories use eventual truth and are NOT available for routing.

## Result

| Target-equal macro | Patient equal | Component equal |
| --- | ---: | ---: |
| Gross harm | 0.057177 | 0.058103 |
| Gross benefit | 0.060528 | 0.057824 |
| Net truth error change | -0.003351 | +0.000279 |
| Net teacher-imitation error change | -0.033526 | -0.036443 |

The small net difference hides substantial cancelling improvements and harms.
The parent experiment still has zero corrected truth-accuracy increments over
the body readout and zero full truth screens. This diagnostic adds no efficacy
test or promotion.

### Direction Versus Overshoot

93.59% of patient-equal gross harm and 92.58% of component-equal gross harm
comes from moving from B toward S AWAY from Y. Crossing Y accounts for the
remaining harm. This is estimator correction direction, not a patient's
clinical direction of change. Moving beyond T is not the same as overshooting
Y: T can itself underestimate the needed correction.

Thus merely limiting correction magnitude would not address most observed
harm. This decomposition does not show how to identify wrong-direction cases
before their truth is known.

### Two Distinct Failure Modes

Teacher-worse cases account for 70.42% / 72.40% of gross harm (patient /
component equal). The remainder occurs despite a better teacher. The detailed
patient-equal contributions below use the ENTIRE target population denominator,
not the mean loss among category members.

| Retrospective teacher quality and student movement | Population share | Gross harm | Gross benefit | Net truth change |
| --- | ---: | ---: | ---: | ---: |
| Better teacher, student moves away from teacher | 17.40% | 0.016528 | 0 | +0.016528 |
| Better teacher, student stays inside B-to-T segment | 26.63% | 0 | 0.033223 | -0.033223 |
| Better teacher, student moves beyond teacher | 9.64% | 0.000387 | 0.015808 | -0.015421 |
| Worse teacher, student moves away from teacher | 15.17% | 0.002139 | 0.009790 | -0.007651 |
| Worse teacher, student stays inside B-to-T segment | 21.96% | 0.021138 | 0.001707 | +0.019431 |
| Worse teacher, student moves beyond teacher | 9.20% | 0.016986 | 0 | +0.016986 |

There are two separate obstacles: faithfully approaching an unsuitable teacher
can hurt, and a better teacher's correction is not always recovered by the
student. Neither follows from a single global offset or overshoot explanation.
The second obstacle alone is NOT proof of irreversible encoder information loss;
student approximation, fit/query transport and conditional heterogeneity remain
possible contributors. The first does not show that teacher quality is
prospectively identifiable.

## Representative Targets

| Target | Gross harm | Gross benefit | Net truth change | Net imitation change |
| --- | ---: | ---: | ---: | ---: |
| BUN | 0.053144 | 0.050316 | +0.002828 | -0.022840 |
| Creatinine | 0.064714 | 0.058262 | +0.006452 | -0.021923 |
| Hemoglobin | 0.044681 | 0.039717 | +0.004964 | See aggregate report |
| Glucose | 0.037330 | 0.036749 | +0.000581 | See aggregate report |
| Albumin | 0.038531 | 0.037014 | +0.001517 | See aggregate report |
| Troponin | 0.115894 | 0.146998 | -0.031104 | See aggregate report |

These are descriptive values, not newly significant target improvements.
Troponin's student improvement over body still does not establish superiority
over the original native predictor or raw teacher. All 47 targets, not just
these examples, are present in the machine-readable report.

## What This Changes

Confirmed: better scalar-teacher imitation is not a valid substitute objective
for better truth reconstruction here. A new teacher-only distillation head or
global correction shrinkage should not be presented as an untested repair.

Still unresolved: whether source-only training on real truth can learn which
corrections are useful on new patients/components. Existing support selectors,
readout blending, early stopping and raw observation skips already tested parts
of this question; they did not establish full incremental shared-state value.
A further experiment must distinguish those previous approaches, predefine its
truth-based objective and matched controls, and not route using these hindsight
categories. No new training experiment is claimed in this report.

## Verification and Artifacts

- Exact row-level absolute-loss identity and exhaustive synthetic sign cases.
- Every partition and its quality-by-position intersections reproduce original
  total patient/component truth and imitation differences.
- All 239 bound files rehashed successfully; parent bindings were checked during
  diagnostic generation. No patient rows or identifiers in the public artifact.
- Eleven geometry/teacher tests pass.
- Protocol: `RECONSTRUCTION_TEACHER_ERROR_GEOMETRY_PROTOCOL_20261003.md`.
- Script: `scripts/diagnose_teacher_error_geometry.py`.
- Report: `artifacts/native_reconstruction_teacher_error_geometry_20261003.json`.
- Parent: `RECONSTRUCTION_STATE_TEACHER_FINDINGS_20261003.md`.

Release weights, runtime registry, intervals and promotion gates are unchanged.
`promotion_allowed: false`; all interpretation remains observational.
