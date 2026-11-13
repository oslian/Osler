# Whole-Body Physiology Plan Implementation

Date: 2026-07-24

## Scope

This implementation advances the factual observation-and-forecast stack. It
does not add treatment-effect, counterfactual, clinical, or autonomous decision
authority.

## Implemented

### Canonical event ledger

`osler_jepa/event_ledger.py` defines one causal event contract for:

- measurements;
- medications;
- infusions;
- procedures.

It preserves the actual event time, normalized variable and unit, database and
source-table provenance, quality, route, dose, rate, and end time. Anchor views
use only events at or before the anchor. Future labels are rebuilt from real
post-anchor measurement events with a target-specific tolerance, actual
`delta_t`, paired observed anchor, and strict training weight. Next-measurement
time is emitted as a separate observation-process label.

`osler_jepa/event_dataset.py` turns the ledger into JEPA-ready rows containing
current values, masks, measurement ages, strict future labels, quality weights,
actual target-specific elapsed times, next-measurement times, and factual
pre-anchor treatment summaries.

### Joint JEPA representation

The current measurement-pure history path is now explicitly exposed as:

```text
neural predict-update patient state
  -> variable Transformer
  -> organ-system tokens
  -> shared whole-body latent
  -> target-specific temporal/future JEPA latents
  -> validated sparse cross-organ adapters
  -> persistence-anchored factual heads
```

`organ_latents` and `organ_presence` are auditable model outputs. They are
learned representations, not claimed biological ground truth.

### Runtime contract v2

`whole_body_state_forecast.v2` exposes, per target and horizon:

- observed value and separate nowcast field;
- factual forecast point;
- lower/upper interval only after calibration;
- nominal and actual target-specific horizon;
- measurement age;
- source model and organ latent source;
- strictly observed treatment context;
- movement permission and fallback reason;
- observation/stability/next-measurement fields when available;
- `causal_claim_allowed: false`.

Missing or unvalidated targets remain persistence or unresolved missing.

### Promotion registry v2

Runtime permission now requires all of:

- the existing seven-seed patient, patient-bootstrap, hospital, conformal,
  cross-system, and module-balanced gate;
- direction accuracy and delta correlation;
- a direct comparison with the current validated router;
- learned-latent SVD rank and held-out mode stability.

Missing evidence fails closed.

## Evaluation correction

The first learned-latent audit compared the same-index SVD vectors between fit
and held-out samples. That comparison is not rotation invariant: close singular
values allow equivalent modes to rotate within the same subspace.

The corrected gate uses principal-subspace overlap. In a bounded 20-stay,
two-module smoke:

- test effective rank ranged from 16.86 to 25.20;
- fit/test rank ratios stayed between 0.61 and 0.93;
- mean principal-subspace overlap ranged from 0.687 to 0.929;
- latent stability passed 7/7 patient splits.

Therefore the smoke latent was active and stable under the corrected metric.
The earlier same-vector metric was a false-negative evaluation for this case.

## End-to-end smoke

Configuration:

- modules: sepsis and AKI;
- horizons: 1h and 3h;
- 20 stays, 473 aligned rows;
- true JEPA world-model objective;
- learned patient belief state;
- organ tokens;
- target-specific future latents;
- detached measurement-time head;
- seven patient splits and one hospital split.

Result:

- all training and evaluation paths completed;
- validated factual target-horizon cells: 0;
- promotion status: `candidate_only`;
- v2 registry also validated 0 cells because it correctly lacked a proven
  improvement over the current validated router.

This is the intended fail-closed outcome for a smoke. The implementation is
complete, but a full-data candidate must still earn target-by-horizon movement.

## Full-data seven-seed result

Configuration:

- 917 patients and 65,077 aligned rows;
- 154 hospitals;
- 15 body-system modules and 55 canonical variables;
- six targets across 1h, 3h, 6h, 12h, 24h, and 48h;
- target-specific future-window supervision;
- measurement-process neutralization and detached measurement-time head;
- target-specific JEPA latents, organ tokens, validated edge adapters, and
  persistence anchoring.

Representation result:

- learned-latent SVD stability passed 7/7 patient splits;
- held-out effective rank ranged from 11.39 to 17.70;
- fit/test effective-rank ratio ranged from 0.886 to 1.068;
- mean principal-subspace overlap ranged from 0.859 to 0.993.

The whole-body representation is therefore active, non-collapsed, and stable
under held-out patients. The remaining failure is not latent collapse.

Forecast result:

- no target-horizon cell passed persistence bootstrap, direction accuracy, and
  delta correlation in all seven patient splits;
- `MAP@12h` beat persistence in 7/7 splits, but its direction threshold passed
  only 3/7;
- the closest longer-horizon cells were `bicarbonate@48h`,
  `heart_rate@48h`, and several MAP horizons, each passing the complete
  three-metric condition in 4-6/7 splits;
- creatinine and potassium remained unstable and correctly require fallback.

This establishes a narrower diagnosis than earlier experiments: the shared
whole-body latent is now viable, but target-horizon residual gains are not yet
stable enough to replace the validated router. Running an additional hospital
gate cannot promote a cell that already fails the mandatory 7/7 patient gate,
so no checkpoint or runtime source was promoted from this run.

Canonical full scoring artifact:

`whole_body_scoring_plan_implementation_full.json`

## Canonical event-ledger full gate (2026-07-29)

The final event-ledger experiment used 566,376 rows from 3,290 patients and
12 hospitals. Each row contained 6.59 organ modules on average. It used:

- a measurement-pure multiscale history at 48h, 24h, 12h, 8h, 6h, 4h, 3h,
  2h, 1h, and the anchor;
- 15% variable and 15% module masking;
- the true JEPA objective with an EMA target encoder;
- organ tokens, target-specific temporal/future latents, learned patient
  state, observed pre-anchor treatment context, and persistence anchoring;
- one target-head specialization epoch after the joint stage;
- patient, hospital, care-unit, and time holdouts;
- finite-sample-corrected split conformal intervals.

Two implementation faults were corrected before the final run:

1. Precomputed multiscale histories were no longer overwritten during array
   construction.
2. The EMA target encoder remains in evaluation mode while the context
   encoder trains, so target latents are not made stochastic by dropout.

The conformal implementation now uses the exact split-conformal order
statistic `ceil((n + 1) * 0.9)`. Patient-split compatibility accepts a split
whose point coverage is in the 0.87-0.93 band or whose Wilson interval contains
0.90, while pooled coverage must still be in the target band. This corrects
finite-sample evaluation without weakening the coverage target.

### Full gate result

- evaluated target-horizon cells: 229;
- validated target-horizon cells: 0;
- validated joint-runtime cells: 0;
- promotion status: `candidate_only`.

The closest cell was `AST@1h`. It passed every value, bootstrap,
module-balance, hospital, care-unit, and time gate, but failed patient
conformal stability. Its pooled coverage was 0.8967 over 3,583 rows, while one
patient split covered only 0.8546 over 509 rows. This is genuine
under-coverage, not a small-sample scoring artifact. No threshold was relaxed.

Target-head specialization was therefore not made the default. It improves
some patient and time checks but mildly regresses hospital and care-unit
generalization. Mondrian anchor-regime conformal remains optional because it
did not rescue the closest cell in the bounded comparison.

### Final SVD and scoring diagnosis

An independent seven-seed scoring audit used 127,404 rows, 922 patients, and
12 hospitals. The learned representation passed the collapse/stability audit
in 7/7 splits:

- median fit effective rank: 29.22;
- median held-out effective rank: 29.22;
- median held-out/fit rank ratio: 0.995;
- median same-index top-mode cosine: 0.946;
- median rotation-invariant top-subspace cosine: 0.979.

The model is therefore not failing because its whole-body latent collapsed.
However, the data-derived transition modes were much less stable across
patients: median same-index mode cosine was 0.128 and median top-subspace
cosine was 0.338.

Patient-equalized scoring found 18 evaluated cells that beat persistence in
7/7 splits, including all glucose horizons and most bicarbonate and
creatinine horizons. Only `bicarbonate@24h` showed a 7/7 improvement from the
full whole-body input over an own-target-only ablation. Thus the joint model
can learn useful factual trajectories, but the incremental cross-organ signal
usually does not beat the already validated per-target router robustly enough
to earn runtime control.

Canonical compact result:

`whole_body_joint_final_gate_summary_20260729.json`

## Final interpretation

The plan's architecture and safety contract are implemented end to end. The
remaining limit is not missing plumbing, latent collapse, or an overly strict
finite-sample formula. It is biological and observational heterogeneity:
cross-organ information adds small, target- and horizon-specific increments
that vary across patient, hospital, care unit, and time.

The validated router remains active. The joint JEPA remains a research
candidate and can only replace a target-horizon cell after that cell
independently passes every gate. No whole-body or cross-organ claim is granted
from representation stability alone.

## Verification

- Full repository unit suite: 304 tests passed on 2026-07-29.
- Focused event-ledger, hidden-belief, sparse-target, runtime, history,
  alignment, movement-policy, and promotion-registry suite: 66 tests passed.
- No existing validated router was replaced.
- No future treatment column is accepted as factual input.
- No numeric interval is emitted without calibration evidence.
- No hidden state is presented as measured biological truth.
