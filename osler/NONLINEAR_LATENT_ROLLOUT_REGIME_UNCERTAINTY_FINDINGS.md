# Nonlinear Latent Rollout: Regime and Uncertainty Findings

Audit date: 2026-07-12.

This is an observation-only candidate. It has no treatment-effect,
counterfactual, clinical or runtime authority.

## Changes tested

The second candidate version adds all four engineering controls requested for
longer factual trajectories:

1. Each body-system cohort is trained as its own dynamics model; no single
   transition network is shared blindly across organs.
2. The latent transition is a three-regime mixture with a learned state and
   horizon-dependent gate.
3. Transition dropout provides Monte Carlo predictive uncertainty. A
   calibration-only 75th-percentile uncertainty gate falls back to
   persistence when the candidate is not confident.
4. History length was increased from 5 to 9 same-stay observations so the
   latent state can use more intermediate measurements.

## Core 7-seed audit

Configuration: history 9, 7 patient-heldout seeds, hospital-heldout,
2 training epochs, 4 MC-dropout samples, grouped bootstrap and split-conformal
calibration.

The values below are median MAE deltas over available targets. Negative is
better. `candidate` is the raw nonlinear rollout; `gated` uses uncertainty
abstention and persistence fallback.

| Module | Horizon | Candidate | Gated | Median abstention |
|---|---:|---:|---:|---:|
| AKI | 1h | +0.097 | +0.074 | 25% |
| AKI | 3h | -0.042 | -0.039 | 25% |
| AKI | 6h | -0.079 | -0.068 | 25% |
| AKI | 12h | -0.158 | -0.132 | 25% |
| AKI | 24h | -0.203 | -0.154 | 25% |
| AKI | 48h | -0.418 | -0.313 | 25% |
| Sepsis | 1h | +0.082 | +0.069 | 25% |
| Sepsis | 3h | -0.055 | -0.046 | 25% |
| Sepsis | 6h | -0.064 | -0.058 | 25% |
| Sepsis | 12h | -0.206 | -0.163 | 25% |
| Sepsis | 24h | -0.221 | -0.171 | 25% |
| Sepsis | 48h | -0.496 | -0.356 | 25% |
| Respiratory | 1h | +0.077 | +0.056 | 25% |
| Respiratory | 3h | -0.024 | -0.018 | 25% |
| Respiratory | 6h | -0.078 | -0.068 | 25% |
| Respiratory | 12h | -0.101 | -0.078 | 25% |

The gate improves the candidate's behavior by refusing its highest-uncertainty
quarter of predictions. It does not create new signal: at 1h, even the gated
candidate remains worse than persistence.

## Regime and rollout checks

Median training regime usage was balanced rather than single-regime collapse:

- AKI: approximately `[0.304, 0.319, 0.345]`;
- sepsis: approximately `[0.336, 0.285, 0.395]`;
- respiratory: approximately `[0.334, 0.347, 0.333]`.

Latent regime entropy was approximately 1.04--1.08, close to the maximum
`log(3) = 1.099` for three regimes.

168h bounded rollout stability:

- AKI: 7/7 patient splits and hospital-heldout passed;
- sepsis: 7/7 patient splits and hospital-heldout passed;
- respiratory: 6/7 patient splits passed; one patient split remains unstable,
although hospital-heldout passed.

## History Ablation

An apples-to-apples AKI comparison kept the same regime mixture, dropout,
training budget and splits while changing only history length:

| History | 3h gated delta | 6h gated delta | 12h gated delta | 24h gated delta | 48h gated delta | Stability |
|---:|---:|---:|---:|---:|---:|---:|
| 5 observations | -0.040 | -0.069 | -0.133 | -0.161 | -0.307 | 7/7 |
| 9 observations | -0.039 | -0.068 | -0.132 | -0.154 | -0.313 | 7/7 |

The difference is small. Simply extending the memory window does not explain
the long-horizon improvement. The stronger contributors are the nonlinear
latent transition, horizon-aware rollout and uncertainty fallback. Additional
history remains useful as an input, but it is not a substitute for a better
state-transition model or denser intermediate observations.

## Decision

The design changes are useful: they reduce long-rollout instability and make
the model abstain when uncertain. They do not justify global promotion.

The candidate remains `candidate-only`. A target × horizon cell still needs a
locked independent confirmation with all of the following:

- 7 patient-heldout bootstrap intervals excluding zero;
- hospital-heldout improvement;
- conformal coverage inside 0.87--0.93;
- bounded rollout stability;
- no future-column leakage;
- no selection based on the final test set.

All claims in this document are factual observation claims only.
