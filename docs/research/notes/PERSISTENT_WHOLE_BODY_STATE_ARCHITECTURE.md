# Persistent Whole-Body State Architecture

Date: 2026-08-28

## Decision

The old three-latent theory of independent **system**, **organ**, and **body**
levels is retired. Functional-system labels remain useful for extraction,
auditing, and provenance, but they are not separate hidden physiological
states.

The model now has exactly two persistent state types:

1. one patient-level whole-body state;
2. one state for each anatomical organ.

These are one dynamical system, not separate predictors. The body state
summarizes their shared condition and feeds back into organ transitions. Every
target and horizon is read from the same recurrent trajectory, but a target
forecast reads its anatomical owner organ by default. A non-owner organ may
move that target only through a separately validated sparse adapter.

```text
causal measurement history
          |
          v
patient predict-update encoder
          |
          v
anchor body state + anatomical organ states
          |
          v
sparse recurrent physiology transition (1-hour internal steps)
          |
          +----> organ state trajectory
          |
          +----> body state trajectory
                        |
                        v
          body + owner-organ state change
                        |
                        v
          target/actual-time direct delta decoder
                        |
                        v
          owner-organ JEPA + validated sparse adapters
                        |
                        v
             validated cell or persistence
```

## Why the first candidate failed

The first implementation mixed two incompatible latent meanings:

- the persistent branch produced an absolute anchor/body-state latent;
- the EMA JEPA target encoder represented measured future change.

It then blended these vectors with `0.25 * sigmoid(gate)`. The learned gate
remained near `0.03`, so the new branch received only a small fraction of the
useful gradient. That created a cold-start loop: the branch was weak because
it was barely trained, and it was barely trained because it was weak.

The formal point gate confirmed the symptom. The branch made tiny consistent
improvements over the original joint JEPA for respiratory rate, but not enough
to beat persistence. No continuous cell passed the complete gate.

## Corrected latent semantics

The recurrent layer now remains in absolute patient-state space. For every
horizon it emits the evolved body state, all evolved organ states, organ
uncertainty, and elapsed time.

The first corrected version still made the recurrent state pass through a
shared future-latent readout and then a target-latent adapter. External tests
showed that this double semantic projection was unstable. It has been removed.
Each target now directly decodes body-state change, owner-organ state change,
actual target-specific elapsed time, anchor value, and observation mask into a
zero-start numerical delta around the owner-organ JEPA forecast.

There is no train-time compatibility gate that can starve the recurrent path
of gradient. The recurrent path is trained directly by shared and
target-specific JEPA future losses, and every target correction starts at
exactly zero. Activation is a separate patient-held-out decision: 20% of the
development patients cannot influence fitting, and a target residual may run
only when it significantly beats the identically specialized owner-organ JEPA
on those patients with at least 0.1% relative MAE gain. Otherwise the forecast
is exactly the owner-organ JEPA output.

Training is staged. A measurement-pure joint JEPA first learns the causal
history encoder, anchor body representation, organ tokens, and target heads.
Those compatible weights initialize the persistent candidate. The recurrent
transition and state-change readout are then trained on future supervision.
The final frozen target-head stage uses the target's owner-organ view. This is
encoder pretraining, not a parallel forecast path. A persistent future
correction must beat an identically specialized owner-organ base; beating only
an unrestricted all-organ baseline is insufficient.

The recurrent organ adjacency starts as identity. Anatomical plausibility is
not evidence of predictive transfer. A cross-organ edge must be implemented as
a sparse adapter and pass its own patient-held-out, hospital-held-out, and
no-edge ablation before it can alter a target.

## Non-negotiable boundaries

- No parallel system latent or system-level decoder.
- No independent horizon models pretending to be one trajectory.
- No direct addition of absolute-state and future-change latents.
- No future treatment, future measurement schedule, patient ID, hospital ID,
  or care-unit identity in the physiology encoder.
- No causal interpretation of input sensitivity.
- No inferred hidden state may be presented as physiological truth unless it
  improves a held-out observable outcome against baseline and placebo.
- No unrestricted all-organ input to a target future head. Shared all-organ
  state is allowed for current-state reconstruction; future cross-organ input
  must earn its own external attribution and no-edge ablation.
- No target may replace the release unless its own patient, hospital,
  forward-time, bootstrap, conformal, negative-transfer, and ablation gates
  pass.

## Promotion experiment

The corrected candidate must be compared with identical patients and training
budget against original joint JEPA, the target-specific 311-feature expert
pattern, and patient persistence.

Point accuracy is evaluated first with seven patient splits, patient-level
paired bootstrap, and a held-out hospital. A failed point gate stops the
promotion. Passing cells then proceed to interval calibration, masked recovery,
cross-organ no-route ablation, multi-step consistency, and an independent
database lockbox.

Promotion is cell-specific. A useful shared state does not authorize every
target to move.

When a target shows stable shared-state value but misses one statistical gate,
training may concentrate its value supervision and final target-head stage on
that target. This does not create an independent patient model: the head still
reads the same whole-body trajectory. Target-focused training must be declared
before evaluation and compared against an identically focused base model.

Patient split seeds and model initialization seeds are separate experimental
variables. The seven patient-held-out seeds define only patient partitions;
the candidate and neural comparators use one recorded, fixed training seed.
Conflating split and initialization randomness makes a failed split
uninterpretable. Initialization robustness, when required, is evaluated as a
separately declared deep ensemble and is never hidden inside the patient-split
gate.
