# E38: paired PCGrad versus matched gradient averaging

## Question

E37 actual Adam origins identify cross-target cancellation, but no causal repair.
Test an actual intervention during the complete fixed bridge fit. Previous
bounded7 BUN protected-gradient probes were small, one-step and on different
parameters; they already show that local non-harm need not generalize.

Keep ONE shared state and bridge, all original backbone/readout frozen. Do not
construct a per-target body or choose edges using selection truth.

## Fixed arms

- `mean`: average two example gradients before one AdamW update.
- `pcgrad`: for different targets with negative gradient dot product, project
  each gradient against the ORIGINAL other gradient, then average; otherwise
  identical to mean. Same-target pairs remain ordinary average.

Use the established PCGrad rule (Yu et al., NeurIPS2020):
https://arxiv.org/abs/2001.06782 ; official reference
https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py .
The existing environment is PyTorch, so only the two-gradient algebra is added;
PyTorch supplies autograd/clipping/AdamW. No optimizer reimplementation/library.
This is a stochastic paired-target variant, not all53 gradients per step.

Both arms: E34 initialization, allfivefolds,2048fit examples,2passes,
seed20261004, original RNG permutation paired adjacently, AdamW1e-4/wd.01,
SmoothL1beta1, clip1. Every fit example appears once per pass;1024updates/pass.
Identical fit exposure, parameter count and optimizer updates within the arms.
E34 sequential baseline has twice as many updates: only the matched mean arm
isolates projection, not the comparison against the sequential model.
No epoch/lr/pairing selection or posthoc target routing. No inherited optimizer
state. Unsupported targets remain explicitly unsupported.

## Diagnostics and acceptance

Count conflicting pairs, changed pairs, per-example proposed-gradient harmful
dots and ACTUAL parameter-update harmful dots (including Adam momentum/decay).
Use tolerance1e-12*(1+norm product). Record magnitudes. Projection before Adam
does not promise non-harm after Adam or at finite amplitude/on unseen patients.

AllE34 three roles and20480queries, same masks/timestamps/identities/truth.
Replay each new saved model via model forward and explicit state-step helper;
original E34 saved outputs must remain unchanged. All original parameters except
bridge projection must match bitwise. Bind inputs/code/protocol/checkpoints.

Collapse overlapping folds within patient, then patient/component-equal MAE.
Selection primary pcgrad vs mean, plus fixed_immediate, immediate, fixed_delayed,
fixed_body, bound_start, unbridged, frozen and raw:9comparisons.
Use existing paired_summary5000, seed20261004+targetindex*1000+controlindex;
component seed adds100. Same minimum30patients/3components, CIupper<0 and
p_less<.05/(53*9) in both levels. Full screen requires all9. Keep raw even
if much stronger. Do not call a screen runtime promotion. Fit/same_source are
descriptive (raw unavailable), no new threshold choice from those outcomes.

Tests: conflicting/aligned/zero/identical gradients, same-target no surgery,
swap symmetry, invalid/nonfinite, unchanged nonconflicting fit, frozen backbone,
exact budget and scoring identity/group weighting. Independent scalar MAE check.
Preserve10GiB disk, private rows/checkpoints local; no row IDs committed.
MIMIC/NWICU reused development, not untouched lockboxes. No forecast, interval,
causal or runtime claim. Close this fixed branch if no complete improvement;
do not continue hyperparameter search on these outcomes.
