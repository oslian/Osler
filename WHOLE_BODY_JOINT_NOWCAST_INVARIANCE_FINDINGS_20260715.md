# Whole-Body Joint JEPA: Nowcast Pretraining and Measurement Invariance

Date: 2026-07-15

## Question

Can a shared body latent become more stable when it first learns same-anchor
physiology and is trained to ignore random missingness patterns?

## Changes

- Added masked same-anchor nowcast pretraining to `fit_joint`.
- Exposed the body latent for audit and optional consistency training.
- Added an optional consistency loss between two independent missingness views
  of the same patient-time row.
- Passed both options through the strict whole-body gate.
- Kept patient-heldout, hospital-heldout, conformal, and cross-system gates
  unchanged.

The consistency term is disabled by default. This is deliberate: the first
smoke result showed that combining it with nowcast pretraining can damage
future forecast stability.

## Matched smoke audit

All runs used the same 9,123 aligned rows, 5 modules, 7 patient seeds, one
hospital split, and the same masking/cross-forecast settings. The baseline
used two forecast epochs so its total update count matched the nowcast-1 plus
forecast-1 run. The consistency-only and combined runs were diagnostic
one-forecast-epoch runs and are not the primary compute-matched comparison.

| Run | Main result | Glucose@6h patient value pass | Bicarbonate@12h patient value pass | MAP@1h median delta |
|---|---|---:|---:|---:|
| Baseline, 2 forecast epochs | candidate-only | 3/7 | 6/7 | +2.125 |
| Nowcast pretrain 1 + forecast 1 | candidate-only | 4/7 | 6/7 | +0.592 |
| Consistency only, weight 0.05 | candidate-only | 3/7 | 6/7 | +0.291 |
| Nowcast + consistency | candidate-only | 0/7 | 6/7 | +0.614 |

The nowcast-only branch is the least harmful and gives a small glucose
improvement in this smoke cohort. It does not pass the full gate. The combined
branch is rejected because glucose patient stability collapses to 0/7 and its
cross-system performance also falls.

## Decision

- Keep the nowcast pretraining implementation as an opt-in candidate path.
- Keep measurement consistency opt-in with default weight `0.0`; do not promote
  the combined branch.
- Do not change the current production research fallback behavior.
- Do not add another heterogeneous router: that family was already rejected by
  prior matched experiments.

## Interpretation

The result supports the narrower hypothesis that learning same-anchor
physiology before sparse future labels can help the shared representation.
It does not support the stronger claim that missingness-invariant latent
consistency solves cross-organ forecasting. The remaining bottleneck is still
target/horizon-specific future signal and label density, not merely latent
capacity.

All outputs remain factual research artifacts. No causal or clinical claim is
enabled, and no target-horizon cell moves unless every existing gate passes.
