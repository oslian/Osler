# Recurrent Event-State Loop Protocol

## Question

Can the validated frozen glucose@6h JEPA source be improved by the complete
patient-scoped loop:

1. anchor-known measurement and treatment events update recurrent patient state;
2. the updated state directly predicts signed direction and residual magnitude;
3. the outcome enters error memory only after its true measurement time resolves?

## Fixed data roles

- eICU source patients: fit and tune only.
- eICU hospital-heldout: internal transport evidence.
- MIMIC-IV: previously inspected external validation evidence.
- No result in this experiment is an untouched lockbox result.

Patients are disjoint across eICU roles. Predictions are scored only after at
least one earlier outcome has resolved, while the recurrent state is replayed
from each patient's first eligible anchor.

## Inputs

Only information known at the anchor is allowed:

- current measured values, observed masks and measurement ages;
- changes from the same patient's previous anchor;
- pre-anchor insulin, dextrose, fluid, steroid and vasopressor evidence;
- residual summaries from outcomes whose measurement times have resolved.

Future treatment, future measurements and unresolved forecast errors are
forbidden.

## Comparisons

- persistence;
- frozen validated JEPA;
- unconditional resolved-error mean;
- recurrent event-state signed residual;
- the same event encoder and head with state reset at every anchor;
- capacity-matched recurrent training with causal wrong-patient event context;
- recurrent prediction with wrong-patient state/context.

All learned variants use source-only patient-disjoint fit/tune roles and the same
bounded signed-residual objective. Hyperparameters are fixed before external
evaluation.

## Promotion gate

The recurrent event-state arm must significantly beat every comparison in all
seven source-defined seeds on both eICU hospital-heldout and MIMIC, using
patient-level paired bootstrap confidence intervals. Changed, stable,
treatment-transition and changed-after-treatment-transition subsets are
reported separately but cannot override the primary gate.

Failure leaves runtime weights, registry, intervals and serving behavior
unchanged. This experiment is factual prediction only and keeps
causal_claim_allowed false.
