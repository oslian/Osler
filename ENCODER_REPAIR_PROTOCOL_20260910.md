# Decoder / JEPA Pressure Factorial Diagnostic

Freeze this protocol before outcomes. Reuse availability encoder patients,
input semantics, exact glucose5-7h labels, seeds7/19/31, source patient fit/tune,
8epochs x16steps x32 equal-patient samples, AdamW1e-4, gradient clipping1.
All contexts train. No treatment, new predictor or new data selection added.

Eight arms: own/full input crossed with four modes:
bounded (existing tanh), unbounded (identity instead of tanh), balanced (tanh,
source-fit gradient balance), unbounded_balanced (both). The coefficient0.75
and movement gate remain identical. Identity can emit any signed change; it
does not force a larger change or oracle direction. Same SmoothL1 value loss.

Balanced arms estimate mean context-gradient norms on three32-row source-fit
patient-balanced batches at initialization. JEPA weight is value-norm/JEPA-norm
clipped to[1,20], fixed for all subsequent steps. This is a prespecified bounded
gradient-balancing diagnostic, not adaptive multi-task optimization. Own and
full use the same sampling orders. No external metrics select settings.

Epoch0 under EACH arm's activation is included in source-tune MAE selection.
Weights require activation metadata when reconstructed; identity candidates
must never load into the serving tanh model as if they were equivalent.
Release architecture defaults to tanh and receives no changed weights.

Patient-equal MAE and paired patient bootstrap: each repair vs matching bounded
control, full vs own per mode, all vs persistence. Report all three seeds and
latent rank, not just best arm. MIMIC is reused validation, not pristine external
evidence. No conformal or retention/promotion claim. Any possible benefit needs
full gates and new calibration. Risks: identity can extrapolate extreme values;
stronger JEPA pressure can harm point fitting or collapse teacher representations.
