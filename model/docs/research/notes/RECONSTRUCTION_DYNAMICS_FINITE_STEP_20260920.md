# Protected Dynamics: Actual Finite-Step Replay

Fixed step0.01 applied to each of two source-fit protected directions. Each
has an unprotected BUN direction scaled to the identical L2norm; no step search,
no outcome-selected winner. Change only body_dynamics parameters of bounded7.
Original90+90source-selection patients are disjoint from gradient-fitting
patients; they previously selected the checkpoint and are not a fresh lockbox.
All33trained targets,2470/2025pairs; baseline per-patient errors reproduce1e-7.
Parameter vector restored after every sample including exceptions.

## Measured Outcomes

Protected-step BUN SmoothL1 changes:

| Direction | Selected90 | Broad83 |
|---|---:|---:|
| Probe0 |+.00001174|-.00016791|
| Probe1 |-.00003687|-.00003289|

Only probe0 broad has a nominal unadjusted interval excluding zero:
[-.00032648,-.00000323]. Other intervals crosszero; these tiny changes are not
meaningful promotion evidence. Equal-norm unprotected probe0 improves broad
more (-.00041679), while its selected change+.00006062 is uncertain.

Protected SmoothL1 mean harms: probe0selected16/33,broad14/33;
probe1selected12/33,broad16/33. These are numerical mean signs, NOT significant
regression counts. Protection does not guarantee loss non-increase on separate
patients. The finite-step experiment supports the prior first-order caution;
it does not establish training convergence, inevitable conflict, or failure of
all robust optimization approaches.

## Decision

Do not promote either one-step candidate or choose probe0because its broad
interval looked favorable. No saved runtime checkpoint; private directions
and step suffice to reproduce the diagnostic. All other parameters and active
release unchanged. No interval/forecast claims from current reconstruction.

Four direction/solver tests pass, equal-step-norm controls verified. Remaining
problem is representative loss protection across patients, not absence of an
in-sample BUN descent direction. Any sustained training trial needs fixed source
selection and all-target retention checks rather than repeated tiny-step
searches on these same selection outcomes.

Evidence: artifacts/native_reconstruction_dynamics_finite_step_20260920.json.
