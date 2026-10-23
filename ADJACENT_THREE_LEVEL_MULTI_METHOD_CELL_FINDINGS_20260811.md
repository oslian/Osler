# Adjacent Three-Level Multi-Method Cell Findings

Date: 2026-08-11

## Scope

This audit attempted to promote additional target-horizon cells under the
formal adjacent hierarchy:

```text
validated system expert -> anatomical organ layer -> whole-body layer
```

Every deployable path remained path-specific and fail-closed. A cell had to
pass seven patient splits plus hospital, care-unit, forward-time, conformal,
module-balanced, teacher, persistence, and matched path ablations. No test
labels were used for training or calibration selection.

## Methods tested

1. Path-specific candidate scoring (`direct_organ` or `body_mediated`).
2. Stronger provenance-domain invariance.
3. Hospital- or care-unit-balanced residual-adapter sampling.
4. Distributional shape, asymmetric, and calibration-only adaptive
   conformal intervals.
5. Rarest-observed-module balancing as a bounded experiment. It was rejected
   and removed from the implementation after failing to improve the external
   module-balanced gate.

## Results

### `heart_rate@3h`

- Direct-organ candidates passed all seven patient splits.
- Increasing hospital invariance and equal-hospital sampling did not make the
  hospital bootstrap or direct-path ablation significant.
- Conclusion: the external failure is not explained by hospital sample-count
  imbalance alone. The cell remains candidate-only.

### `urine_output@12h`

- Calibration-only asymmetric adaptive conformal passed all seven patient
  splits.
- External point and path gates passed.
- External interval coverage remained unstable: hospital coverage was about
  `0.933`, while forward-time coverage fell to about `0.841`.
- Conclusion: residual shape changes over time; one global calibrator is not
  transportable. The cell remains candidate-only.

### `o2sat@3h`

- The body-mediated candidate passed seven patient splits, teacher,
  persistence, matched system-only ablation, and conformal calibration.
- Hospital and forward-time gates passed.
- Care-unit heldout failed point, path, module-balanced, and conformal gates.
- Adding care-unit invariance and equal-care-unit sampling did not change the
  failure pattern.
- Conclusion: the missing factor is not care-unit identity or sample weight.
  The likely gap is unobserved respiratory support, device, or measurement
  context that differs by ICU type. The cell remains candidate-only.

### `respiratory_rate@1h`

- Hospital-balanced sampling strengthened the body-mediated candidate:
  seven of seven patient splits passed, and hospital, care-unit, forward-time,
  conformal, teacher, persistence, and matched path gates passed.
- The only remaining failure was the care-unit module-balanced gate:
  `7/11` anatomical subgroups improved, below the required `0.80` pass rate.
- Heart, blood-vessel, lung, kidney, liver, pancreas, and bone-marrow subsets
  improved. Thyroid, adrenal, brain, and skeletal-muscle subsets did not.
- Rarest-observed-module balancing did not improve the `7/11` pass rate and
  worsened several sparse subgroups, so that method was removed.
- Conclusion: this cell is the closest candidate, but sparse subgroups appear
  to have distinct or missing respiratory determinants rather than a simple
  gradient-imbalance problem. It remains candidate-only.

## Decision

No new runtime cell is promoted by this audit. Existing validated experts and
persistence remain active for all affected cells. The results reject further
blind tuning of domain weights, equal-group sampling, and global conformal
calibration for these specific failures.

The next valid improvement for respiratory cells requires richer observed
respiratory context (oxygen device and flow, ventilator mode/settings,
sedation, airway events, and signal-quality provenance) before another model
architecture experiment. This is still factual forecasting;
`causal_claim_allowed` remains `false`.
