# Nonlinear Latent Rollout Candidate Findings

Audit date: 2026-07-12.

## Question

Can a recurrent latent state that evolves through bounded nonlinear transitions
represent longer patient trajectories better than the current local linear
router and persistence?

## Candidate architecture

`train_nonlinear_latent_rollout.py` combines the eICU AKI transition cohorts at
1h, 3h, 6h, 12h, 24h and 48h. The candidate uses:

- a GRU history encoder over same-stay observations;
- value, observation mask and measurement age for each state variable;
- an onset-time context feature;
- a nonlinear latent transition repeated in 6h-bounded steps;
- a bounded residual update around the current state;
- horizon-specific Ridge and persistence comparisons.

The model is trained only on the past/current contract. Future columns are
targets, not input features. Patient and hospital splits are group-disjoint.

## Preliminary run

Run artifact: `/private/tmp/nonlinear_latent_rollout_aki_candidate.json`

Configuration:

- AKI eICU cohorts: 1/3/6/12/24/48h;
- 280,025 rows, 14,202 subjects, 15,202 stays;
- 3 patient-heldout seeds;
- 1 hospital-heldout split;
- 3 epochs, 90,000 training rows per seed after per-horizon capping;
- latent width 48, history length 5.

The candidate effective rank was 25.9--28.9 out of 48, so this run did not
show latent collapse.

## Median target deltas versus persistence

Negative values are better. The values below are medians across the 14 AKI
targets for each patient-heldout seed; they are exploratory point estimates,
not promotion evidence.

| Horizon | Seed 0 | Seed 1 | Seed 2 | Hospital-heldout |
|---:|---:|---:|---:|---:|
| 1h | +0.126 | +0.165 | +0.155 | +0.160 |
| 3h | -0.050 | -0.032 | -0.041 | -0.038 |
| 6h | -0.094 | -0.069 | -0.083 | -0.074 |
| 12h | -0.173 | -0.159 | -0.150 | -0.143 |
| 24h | -0.149 | -0.192 | -0.243 | -0.237 |
| 48h | -0.378 | -0.392 | -0.381 | -0.370 |

The qualitative pattern is useful: the candidate is not helpful at 1h, but
begins to outperform persistence on the longer horizons in this capped AKI
experiment. That is consistent with a state-evolution model, but it does not
prove generalization or clinical usefulness.

## Decision

Status: **candidate-only**.

It is not connected to the production whole-body router. Promotion still
requires all of the following:

1. 7 patient-heldout seeds;
2. hospital-heldout improvement with confidence intervals;
3. comparison against the existing router, not only persistence;
4. per-target/per-horizon bootstrap intervals and conformal calibration;
5. rollout stability and non-collapse checks on an uncapped or independently
   held-out cohort;
6. no causal, counterfactual, clinical or runtime authority.

The first result supports replacing a direct long-horizon linear extrapolation
experiment with a nonlinear latent transition experiment. It does not justify
promoting the candidate yet.
