# Other-Cell Expansion Findings (2026-08-02)

## Objective

Try multiple bounded methods to promote additional target-horizon cells beyond
the validated `glucose@1h` and `glucose@3h` cells, without weakening the
seven-seed, external-holdout, uncertainty, or matched-ablation gates.

All experiments used the same connected 53-target, 11-organ, two-level JEPA
backbone on 1,000 eICU stays. Failed cells remain on their existing router or
persistence fallback.

## Methods tested

1. **Focused near-miss target training**
   - Nineteen near-miss targets, extra adapter training, unchanged data splits.
   - Result: zero cells passed point plus conformal gates in all three screening
     seeds. Removing the other targets also removed useful regularization.

2. **Adaptive patient-cluster conformal**
   - Calibration-patient-only nominal-coverage selection.
   - Result: zero complete seven-seed cells; rejected.

3. **Current-state Mondrian distributional conformal**
   - Calibration-only current-value strata with patient-cluster fallback.
   - Result: generally increased overcoverage and reduced the number of passing
     splits; the three-seed screen was negative, so a seven-seed run was not
     justified.

4. **Target-specific projection residuals**
   - Small zero-initialized target adapters above shared whole-body, target
     future, and own-organ latents.
   - A matched disable-projection path tested whether the new parameters added
     information rather than capacity alone.
   - Result: no complete seven-seed cell. `albumin@1h`, `MAP@48h`, and
     `respiratory_rate@24h` passed all point and matched projection gates, but
     conformal coverage passed only 1/7, 3/7, and 2/7 splits respectively.

5. **Asymmetric distributional conformal**
   - Calibration patients separately estimated lower-tail and upper-tail
     normalized errors. Test labels were not used.
   - Result: useful on some splits and harmful on others; zero complete
     seven-seed cells.

6. **Per-cell interval-shape selection**
   - For each target-horizon cell, calibration patients were divided into a
     shape-selection set and a disjoint final-calibration set.
   - Inner grouped validation selected symmetric or asymmetric intervals; the
     untouched final-calibration patients built the interval.
   - Result: `heart_rate@3h` passed teacher, persistence, no-expert, and
     conformal gates in all seven patient splits.

7. **Target-horizon-balanced adapter loss**
   - Each horizon contributed equally within each target, preventing dense
     horizons from dominating the residual and scale losses.
   - Combined with per-cell interval-shape selection.
   - Result: several 6/7 near-misses, but zero complete seven-seed cells.
     `hematocrit@3h` passed all point gates but conformal was 6/7;
     `respiratory_rate@3h` passed conformal 7/7 but point gates were 6/7.

## External result for heart rate at 3 hours

The internally successful `heart_rate@3h` candidate was retrained on hospital,
care-unit, and forward-time holdouts.

- Point accuracy significantly beat teacher and persistence in all three
  external splits.
- Hospital conformal passed at 0.9295 patient-equalized coverage.
- Forward-time conformal passed at 0.8724.
- Care-unit conformal was 0.9310, just above the fixed 0.93 upper boundary.
- The forward-time matched no-expert bootstrap failed.

Therefore `heart_rate@3h` remains candidate-only. The result is evidence of a
real point-prediction signal, but not of a sufficiently stable sparse-expert
contribution and uncertainty contract.

## Decision

No new target-horizon cell is added to the validated runtime registry. The
validated set remains:

```text
glucose@1h
glucose@3h
```

The negative results are not treated as missing evidence. They show that:

- several other cells contain reproducible point signal;
- interval calibration is target-, horizon-, and care-setting-specific;
- three-seed screens substantially overestimate promotion readiness;
- target-specific capacity and horizon balancing alone do not remove the
  external distribution shift.

## Artifacts

- `teacher_anchored_two_level_focused_nearmiss_1000_7seed.json`
- `teacher_anchored_two_level_all_targets_adaptive_conformal_1000_7seed.json`
- `teacher_anchored_two_level_all_targets_mondrian_3seed.json`
- `teacher_anchored_two_level_target_projection_1000_7seed.json`
- `teacher_anchored_two_level_target_projection_asymmetric_1000_7seed.json`
- `teacher_anchored_two_level_shape_adaptive_1000_7seed.json`
- `teacher_anchored_two_level_shape_adaptive_external_1000.json`
- `teacher_anchored_two_level_horizon_balanced_shape_1000_7seed.json`

All reports are aggregate-only and stored under
`/Users/chunyouchang/Documents/symbolic engine/artifacts/reports/`.
