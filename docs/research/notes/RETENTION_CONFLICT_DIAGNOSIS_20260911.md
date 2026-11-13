# Failure localization and retention conflict

This diagnostic does not retrain or change runtime, weights, intervals or gates.
It measures source-fit gradients at seven previously selected checkpoints for
each of uniform and source-qualified retention. Three patient-uniform batches
per seed, identical batches between arms: 42 probes. No evaluation labels are
used in the probe. All parent artifact bindings were verified before execution.

| Shared parameters | Opposing probes | Median cosine | Retention/primary norm |
|---|---:|---:|---:|
| Uniform retention | 12/21 | -0.1878 | 0.3574 |
| Selected retention | 6/21 | 0.2070 | 0.1764 |

The groups include belief_filter, variable_mixer, module_mixer and state_encoder.
Uniform belief-filter cosine median is -0.3688; selected is 0.4680. This is
primary glucose value-plus-JEPA versus frozen-output retention, NOT the earlier
JEPA-versus-value audit. Local raw gradients do not establish Adam trajectory
causation, statistical independence, or that gradient projection will fix accuracy.
The checkpoints differ, so these are not an isolated same-weight mask intervention.

## What has actually been located

1. The earliest measured extra-context failure is already fit to unseen source
   patients: original native glucose whole/own fit MAE 32.2889/34.0378 becomes
   34.8408/33.9762 on source evaluation. This is not exclusively database shift.
2. Glucose-only fine-tuning changes the shared prediction path while 208 other
   value-head tensors remain unchanged. Actual other-target errors increase.
   Heads are trainable, but absent their own loss they receive no useful update.
   This establishes a supervision/retention problem, not that all shared state
   is useless or that release experts have regressed.
3. Preserving every initial output reduces several regressions but preserves
   inaccurate initial renal outputs. Source-qualified retention reduces that
   cost and local gradient opposition, yet leaves unprotected targets worse
   and glucose whole-versus-own unstable. Conflict alone is not the root cause.

## Diagnosis boundary

The best-supported mechanism for the *multi-target retention failure* is a
shared path being optimized for a primary target without dependable supervision
for all affected outputs. Frozen teacher imitation cannot supply missing truth.
The separate *incremental glucose generalization failure* remains incompletely
identified: measured fit gains fail on unseen source patients. Existing source,
timing, scale, dropout and retention interventions do not fully repair it.
Do not claim a single discovered cause explains all historical failures.

A discriminating follow-up would isolate shared versus target-private updates
on exactly the same source patients, preserving the same usable real auxiliary
labels. Earlier auxiliary training already failed to provide stable glucose
gain, so merely adding auxiliary losses again is not a new solution. Any new
update-allocation intervention requires a matched own-history control and actual
other-target outcomes, not only teacher agreement. Reused hospital results are
not an untouched lockbox; historical pretraining independence is uncertified.

Artifact: `artifacts/retention_gradient_conflict_20260911.json`.
Runner: `python -m scripts.audit_retention_gradient_conflict`.
Observational only; no promotion or universal biological conclusion.
