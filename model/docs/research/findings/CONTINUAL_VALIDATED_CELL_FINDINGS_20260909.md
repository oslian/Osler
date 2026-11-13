# Continual learning on validated glucose@6h

## Decision

`candidate_only`. Continual learning was tested on a release cell that already
has real cross-database predictive value. A causal patient-specific residual
state improved repeat predictions, but every method that updated the global
311-feature residual adapter degraded the patient stream and caused significant
retention, hospital and MIMIC-IV harm. Runtime weights, registry, intervals and
serving behavior remain unchanged.

## Why this cell was selected

The fixed target was `glucose@6h`, whose incumbent release source is
`unified_shared_jepa_target_head`. Rows required a freshly observed anchor
glucose and an observed future glucose 5-7 hours later.

On the complete strict co-measurement slice, before any continual updates, the
frozen source beat persistence in both databases:

| Database | Patients | Frozen MAE | Persistence MAE | Patient-equal delta (95% CI) |
|---|---:|---:|---:|---:|
| eICU | 1,706 | 38.49 | 40.71 | -2.22 (-3.40, -1.03) |
| MIMIC-IV | 996 | 32.20 | 33.80 | -2.18 (-3.86, -0.70) |

This is therefore a stronger test than adapting a weak candidate that already
loses to persistence.

## Fixed protocol

- eICU alone supplied development data.
- The existing deterministic patient and hospital roles were preserved.
- MIMIC-IV was repeated external validation, not an untouched lockbox.
- Predictions were recorded before labels became available.
- A label could update state or weights only after its true future measurement
  time.
- The adapter began at exactly zero, so its first prediction equaled the frozen
  release prediction.
- Seven patient arrival orders were tested.
- All weight-updating arms used the same 16-example update budget.

Support was 947 memory rows from 460 patients, 1,015 retention rows from 460
patients, 979 patient-test rows from 414 patients, 797 hospital-test rows from
372 patients and 2,349 MIMIC rows from 996 patients. The repeat-patient primary
slice contained 565 predictions from 241 eICU patients.

## Results

The primary comparison used each patient's second-or-later eligible prediction.
Negative delta is better than the frozen release.

| Method | Mean delta vs frozen across orders | Significant improvement | Beats persistence | External/retention harm |
|---|---:|---:|---:|---:|
| Frozen weights + patient residual state | -2.27 mg/dL | 7/7 | 0/7 | 0/7 |
| Naive online weight update | +9.33 mg/dL | 0/7 | 0/7 | 7/7 in every role |
| Short/long replay | +2.48 mg/dL | 0/7 | 0/7 | 7/7 in every role |
| Uniform replay | +2.50 mg/dL | 0/7 | 0/7 | 7/7 in every role |

The patient state result was deterministic across arrival orders because it
updates only from that same patient's previously resolved errors. Its paired
bootstrap CI against the frozen source was `[-3.81, -0.88]` mg/dL. However, on
this smaller repeat-only slice its delta against persistence was -1.99 mg/dL
with CI `[-4.92, 1.25]`, so it did not earn promotion.

Replay reduced the immediate stream damage relative to naive updates, but it
did not make global weight adaptation safe. Across patient orders, mean harm
after adaptation was:

| Method | eICU retention | eICU hospital heldout | MIMIC-IV |
|---|---:|---:|---:|
| Naive online | +25.22 mg/dL | +25.10 mg/dL | +40.82 mg/dL |
| Short/long replay | +10.91 mg/dL | +10.72 mg/dL | +23.27 mg/dL |
| Uniform replay | +10.64 mg/dL | +10.45 mg/dL | +25.78 mg/dL |

## Failure location

The updates did not fail primarily because the correction was clipped. The
adapter limit was 197.0 mg/dL, and only about 0.4-0.9% of repeat-stream replay
predictions reached it. The more important difference was correction scale:

| Method | Mean absolute correction | 90th percentile |
|---|---:|---:|
| Patient residual state | 11.49 mg/dL | 27.13 mg/dL |
| Naive online | 41.85 mg/dL | 93.03 mg/dL |
| Short/long replay | 29.59 mg/dL | 61.16 mg/dL |
| Uniform replay | 29.17 mg/dL | 59.70 mg/dL |

A few resolved labels are useful for estimating a low-dimensional bias for the
same patient. They are not sufficient to rewrite a 311-feature population
mapping. Replay remembers earlier examples but does not prevent the adapter
from converting a patient-local error into a large global correction that is
unstable across patients and databases.

## Conclusion

The tested self-learning hypothesis splits into two parts:

1. **Personal state adaptation has real signal.** A frozen shared model plus a
   causal running patient-error state improved repeat glucose predictions in
   all seven orders without changing predictions for unrelated patients.
2. **Global online weight adaptation is not safe.** Naive, uniform replay and
   short/long replay all failed the predeclared gate and caused strong external
   harm.

The next defensible design is therefore not a larger replay memory. It is a
frozen shared JEPA with an explicitly patient-scoped, low-dimensional
predict-update state whose influence decays with age and whose interval is
calibrated separately. Any future trial must compare that state against the
current simple residual mean and persistence on a new untouched database.

Authoritative aggregate report:
`artifacts/continual_validated_glucose_6h_20260909_v2.json`.
