# Patient-state failure localization

## Decision

`diagnostic_only`. The four prespecified diagnostics completed. They locate a
transfer failure in patient residual correction and a weaker calibration
opportunity, but do not justify changing runtime points, intervals, source
selection or serving behavior.

## Evaluation integrity correction

The previous findings said the time-aware state failed all seven wrong-patient
placebos. Its JSON actually recorded 7/7 significant comparisons in every
role. The discrepancy arose because the primary placebo was built on the full
role frame while the seven-seed placebo was rebuilt after filtering to
personalized rows. The two controls therefore did not have the same donor
population.

The corrected audit builds every donor assignment on the full role support,
then evaluates the same personalized anchors. Donor state must come from an
equal or earlier relative stay time. Under this consistent contract, the
simple level passed 1/7 placebo comparisons on eICU hospital-heldout and 0/7
on MIMIC; decay and slope passed 0/7 on both. The original `candidate_only`
decision remains correct, but the old placebo explanation is superseded.

Repeated forecasts resolving to the same future measurement were also
deduplicated. This removed only 3 hospital and 6 MIMIC updates. Eight MIMIC
patients spanned multiple stays. Neither explains the main result. Cached
examples contain measurement event time, not laboratory result availability
time; that timing limitation remains explicit.

## Component ablation

eICU selected a residual-level prior of 3, a 48-hour decay with prior 1, and a
48-hour level-plus-slope state with prior 1 and slope penalty 16.

Patient-equal personalized MAE:

| Method | eICU selection | eICU hospital | MIMIC-IV |
|---|---:|---:|---:|
| Persistence | 46.83 | 40.24 | 30.13 |
| Frozen release | 45.81 | 36.27 | **28.94** |
| Patient residual level | **43.13** | **35.88** | 29.22 |
| Decayed residual level | 43.44 | 37.65 | 31.82 |
| Decayed level plus slope | 43.31 | 37.46 | 30.69 |

The plain patient level significantly improved the frozen release in the eICU
selection role by 2.68 mg/dL, CI `[-4.55, -0.90]`. Its hospital improvement
was only 0.39, CI `[-1.98, 1.20]`, and MIMIC was 0.27 worse, CI
`[-1.14, 1.61]`. Thus the first transfer failure is already present in the
simple personal bias, before temporal complexity is added.

Decay was significantly worse than the plain level by 1.77 mg/dL in eICU
hospital-heldout, CI `[0.73, 2.91]`, and 2.60 in MIMIC, CI `[1.31, 4.53]`.
Slope recovered part of that loss but remained 1.47 worse than the plain level
in MIMIC, CI `[0.67, 2.31]`. The transferable object is therefore not a fixed
exponentially decayed error trajectory.

## Uncertainty-only personalization

All uncertainty comparisons held the point forecast exactly equal to the
frozen release.

| Scale | eICU hospital coverage row / patient | Width | MIMIC coverage row / patient | Width |
|---|---:|---:|---:|---:|
| Constant | 0.886 / 0.910 | 182.86 | 0.951 / 0.953 | 182.86 |
| Patient level | 0.894 / 0.911 | 172.11 | 0.942 / 0.952 | 170.01 |
| Decayed patient scale | 0.886 / 0.905 | **170.46** | 0.938 / 0.944 | **158.67** |

All three passed aggregate eICU hospital coverage. All over-covered on MIMIC
with the source-calibrated radius, so cross-database interval promotion fails.
After explicitly recalibrating inside MIMIC, mean width across seven patient
splits was 122.75 for constant scale, 120.40 for level scale and 119.92 for
decayed scale. Coverage passed 5/7, 2/7 and 4/7 splits respectively. Personal
error history contains useful error-magnitude information, but the current
scale estimator is not more reliably calibrated than the constant reference.

## Information location

Equal-capacity readouts were fit on the same 469 eICU memory patients.

| Input | eICU patient | eICU hospital | MIMIC-IV |
|---|---:|---:|---:|
| Target history only | 40.03 | 39.75 | 33.03 |
| Current cross-variable state | 38.91 | 39.31 | 32.86 |
| Causal cross-variable history | **38.48** | 39.32 | **32.74** |
| Frozen JEPA latent | 38.77 | **39.06** | 32.88 |
| Wrong-patient JEPA latent | 39.59 | 39.87 | 33.58 |

Current cross-state and cross-variable history significantly improved target
history on the eICU patient role, but neither improvement remained significant
in eICU hospital-heldout or MIMIC. The JEPA latent significantly beat its
wrong-patient latent in MIMIC by 0.70 mg/dL, CI `[-1.25, -0.15]`, yet did not
significantly beat target history there, CI `[-0.80, 0.52]`.

This does not support a simple encoder-loss diagnosis: raw cross-variable
history also failed to add stable external value. The latent retains real
patient-specific information, but for `glucose@6h` that information is mostly
redundant with target history or not transportable as future error reduction.
This conclusion is cell-specific and is not a universal information ceiling.

## Failure localization and next repair

The best-supported diagnosis is:

1. residual bias is partly personal in the source database but changes with
   illness and care regime;
2. exponential decay and linear slope assume a continuity that does not
   transfer across hospitals;
3. personal error magnitude is more learnable than personal signed trajectory,
   but still requires a better conditional scale model;
4. other tabular body variables and the existing latent do not reliably reveal
   the regime switch before it occurs for this cell.

The next model experiment should therefore keep the frozen release point and
predict whether a previously estimated patient bias is still valid, using only
anchor-known freshness, treatment and physiological-regime evidence. It must
compare against always-apply, never-apply and wrong-patient state. Interval work
should remain a separate fixed-point task and use disjoint target-domain
calibration before a new untouched database is evaluated.

Authoritative aggregate report:
`artifacts/patient_state_diagnostics_glucose_6h_20260909.json`.
