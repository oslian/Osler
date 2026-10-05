# Source-Only Numerical Rescaling

Completed28 matched32-epoch fits. Source-fit legal history snapshots determine
non-glucose center/scale; no MIMIC statistics fit anything. Same seven seeds,
source support, optimizer, JEPA/value loss and source-tune selection. All14
own-only training logs AND predictions exactly reproduce the prior controls.
This verifies the intervention did not silently change target normalization.

Median patient-equal glucose6h MAE:

| Arm | Hospital original | Hospital rescaled | MIMIC original | MIMIC rescaled |
|---|---:|---:|---:|---:|
| Whole bounded | 42.388 | 42.125 | 35.572 | 35.590 |
| Whole identity | 38.202 | 37.708 | 33.368 | 33.572 |
| Own identity control | 36.559 | 36.559 | 31.238 | 31.238 |

Rescale-vs-original paired CI improve/harm counts: whole bounded3/0 hospital,
1/2 MIMIC; whole identity3/0 hospital,3/1 MIMIC. Source rescaling is not a
consistent cross-domain repair. Whole-vs-own benefit remains0/7 both modes,
both domains. Whole identity still beats persistence7/7 pooled, but MIMIC
TSICU0/7,CCU1/7,MICU4/7,MICU/SICU5/7; CVICU/SICU7/7. Period2014-2016 remains0/7.

## Interpretation

The hypothesis that inherited affine scaling alone explains the remaining
full-context transfer gap is not supported by this test. This does not negate
the earlier proven unsupported-calcium defect, nor exclude method/unit errors,
nonlinear feature distributions, optimization sensitivity or conditional shift.
This is warm-start retraining, not a from-scratch invariance proof.

Source fit statistics use unique snapshot rows, not unique raw events or
patient-equal weights. Constant or unsupported features keep identity transform.
The inherited glucose scaling and all masks/ages are unchanged. This is a
numerical coordinate transform, not replacing measured values by artificial
centers. It does not normalize MIMIC into looking like eICU using test statistics.

## Artifacts And Decision

`artifacts/encoder_rescale_20260910.json`: training and pooled metrics.
`artifacts/encoder_rescale_comparison_20260910.json`: matched control check and
paired original-vs-rescaled outcomes.
`artifacts/encoder_rescale_strata_20260910.json`: descriptive group outcomes.
Checkpoints store numeric_centers/numeric_scales plus inherited scaler binding,
source support and activation metadata. These are mandatory candidate inputs.

298 tests pass. No runtime, registry, calibration or promotion changes.
MIMIC repeatedly inspected; subgroup and paired CIs exploratory, not a new
untouched lockbox or multiplicity-adjusted validation. Do not repeat simple
source affine normalization as an untested solution to the remaining gap.
