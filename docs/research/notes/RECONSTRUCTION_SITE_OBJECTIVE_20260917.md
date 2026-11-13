# Source Objective And Shared Gradient Diagnosis

## Question

Does the shared selection score hide target regressions, and does that establish
gradient competition as the reason for heldout reconstruction failures?

## Source-Tune Endpoint Replay

All source fit/tune shard and training dependency hashes were checked. Initial
fit-median score and both selected real/process tune scores reproduce within
1e-7. No evaluation patients, fitting, new checkpoint selection or runtime
changes were used in this replay.

The real mean score improves by 0.111413. AST/ALT account for 57.27% of that NET
improvement. However, excluding those two descriptively still gives a mean
improvement of 0.050679 across the other31 targets. Consequently the result is
not merely an AST/ALT-only average benefit.

Eight real target MAEs worsen versus their initial fit-median predictions:
calcium, CPK, INR, lipase, magnesium, phosphate, potassium and troponin I.
This confirms that a better common selection score does not enforce individual
retention. It does NOT establish which alternate epoch would improve those
targets: only the selected endpoints have been replayed by this diagnostic.

As a descriptive source-only sensitivity, divide each target MAE by its fit-only
median absolute deviation. All33 deviations are positive (minimum0.337245).
The initial/real/process scores are2.569379/2.425677/2.503489. The aggregate
ranking does not reverse. No newly normalized objective was trained or selected;
this calculation does not show normalization cannot help.

## Persistent-Layer Gradients

`scripts/audit_site_shared_gradients.py` uses eight deterministic hash-selected
FIT patients per target, independent of values/errors. It computes mean
SmoothL1 gradients for each of33targets at the selected real epoch8 checkpoint.
Scope:45,774 persistent physiology-layer parameters. Encoder and target-readout
gradients are outside this diagnostic. No weights are updated.

Under the raw target-equal mean gradient descent direction, the sampled local
loss derivative is positive for glucose and WBC, negative for the other31.
This is an infinitesimal source-sample property, not an actual AdamW update or
evidence of heldout accuracy. Gradient clipping, optimizer momentum and the
training schedule are not simulated.

For the two targets harming all three heldout references:

| Target | Shared gradient norm | Cosine with other-target mean | Local loss derivative |
|---|---:|---:|---:|
| Lipase | 0.414978 | 0.051571 | -0.006082 |
| Triglycerides | 0.142254 | 0.065450 | -0.001008 |

Their sampled source gradients are not blocked or opposed by the remaining
target mean at this endpoint. Therefore this measurement does not support a
universal gradient-conflict explanation. Eight samples at one endpoint are
insufficient to exclude earlier conflict, unsampled patients, readout effects or
encoder interference. No gradient-norm share is called a causal loss share.

## Decision

Retain separate diagnoses: common-score selection permits individual harm;
source generalization and hospital transfer remain distinct failures. Do not
present target reweighting or gradient surgery as an already established fix.
Any controlled objective repair must use source-only settings, preserve all39
catalog statuses and compare the same patients/initialization/schedule. Reused
evaluation is development evidence, not a fresh external lockbox.

This work does not promote a current reconstruction target, future cell,
interval or biological hidden-state claim. Serving weights and registry remain
unchanged. Machine evidence:

- `artifacts/native_reconstruction_site_objective_20260917.json`
- `artifacts/native_reconstruction_site_gradients_20260917.json`
