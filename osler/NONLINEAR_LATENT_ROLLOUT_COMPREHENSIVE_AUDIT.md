# Nonlinear Latent Rollout Comprehensive Observation Audit

Audit date: 2026-07-12.

This audit is observation-only. It does not estimate treatment effects, does
not use treatment as a causal intervention, and grants no clinical or runtime
authority.

## Contract

For each module, the candidate was trained on separate fit groups, calibrated
on separate groups, and evaluated on patient-heldout test groups. A separate
hospital-heldout fit/calibration/test split was also run. The audit checked:

- 7 patient-heldout seeds;
- persistence and horizon-specific Ridge comparisons;
- subject-cluster bootstrap confidence intervals;
- split-conformal 90% interval coverage;
- 168h finite/bounded rollout stability;
- future-column leakage.

The raw aggregate reports are:

- `/private/tmp/nonlinear_latent_rollout_aki_comprehensive_audit.json`;
- `/private/tmp/nonlinear_latent_rollout_sepsis_comprehensive_audit.json`;
- `/private/tmp/nonlinear_latent_rollout_respiratory_comprehensive_audit.json`.

All three leakage checks passed: future target columns were present only as
labels and none were included in the input contract.

## Median Delta Versus Persistence

Negative means lower MAE than persistence. Values are medians over the
available target cells and 7 patient splits.

| Module | 1h | 3h | 6h | 12h | 24h | 48h |
|---|---:|---:|---:|---:|---:|---:|
| AKI | +0.137 | -0.048 | -0.087 | -0.157 | -0.211 | -0.453 |
| Sepsis | +0.079 | -0.050 | -0.073 | -0.184 | -0.188 | -0.380 |
| Respiratory | +0.096 | -0.021 | -0.031 | -0.061 | n/a | n/a |

The observation-level pattern is consistent: the model is not useful for the
shortest 1h horizon, then begins to add information as the physiologic time
scale becomes longer. This does not mean every target improves. Unsupported
target-horizon cells must still fall back to persistence.

## Conformal Coverage

The number of target-horizon-split cells whose heldout coverage landed inside
the 0.87--0.93 gate was:

| Module | Calibrated cells / evaluated cells |
|---|---:|
| AKI | 95 / 98 at 1h, 85 / 98 at 3h, 90 / 98 at 6h, 86 / 98 at 12h, 87 / 98 at 24h, 87 / 98 at 48h |
| Sepsis | 107 / 119 at 1h, 91 / 117 at 3h, 100 / 118 at 6h, 89 / 115 at 12h, 101 / 119 at 24h, 95 / 117 at 48h |
| Respiratory | 91 / 105 at 1h, 84 / 105 at 3h, 75 / 98 at 6h, 78 / 105 at 12h |

Cells outside the gate remain `needs_calibration_audit`; they do not receive
a displayed interval.

## Stability

AKI passed the 168h bounded rollout check on all 7 patient splits, with a
maximum absolute normalized output of 6.03. Sepsis passed on 6/7 splits, with
one split reaching 13.37. Respiratory passed on 6/7 splits, with one split
reaching 27.65. The hospital-heldout stability checks passed for all three
modules, but that does not erase the patient-split failures.

Those failures are important: a model can improve average MAE at supported
horizons and still be unsafe to recursively roll out without a stability
guard. The long-horizon cells for the failed split remain candidate-only.

## Decision

Status: **candidate-only; no production promotion**.

The candidate has demonstrated a promising observation-only pattern for
3h--48h in AKI and sepsis, and 3h--12h in respiratory data. However, target
cells were discovered while inspecting this audit, so they require a locked,
independent confirmation before validation. No cell is promoted merely because
its point estimate is negative.

The production policy remains:

```text
validated target × horizon cell -> candidate forecast + calibrated interval
otherwise                         -> persistence or missing
```

No causal, counterfactual, clinical or runtime claim is allowed.
