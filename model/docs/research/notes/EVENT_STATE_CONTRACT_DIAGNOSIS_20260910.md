# Event-State Failure Localization

> Follow-up: the documented synthetic defects have been repaired. The current
> probe asserts corrected invariants, not the historical failures (reproducible
> at commit 175277d). See `docs/research/findings/EVENT_STATE_REPAIR_FINDINGS_20260910.md` for the final
> seven-seed negative result and remaining raw-event limitations.

## Scope

CodeGraph was synchronized (25 changed files) and the path from `eligible_rows`
through `replay_features` to `fit_recurrent_signed_residual` was inspected.
This is diagnostic-only. No runtime weights, registry, calibration or serving
behavior changed. Historical JSON results are unchanged.

The recurrent glucose@6h experiment has patient-equal MAE 36.202 versus frozen
36.265 in eICU, and 29.347 versus frozen 28.944 in MIMIC (seven-seed means).
It significantly beats frozen in 0/7 fits in both cohorts. Similar performance
does not establish one shared root cause.

## Earliest structural mismatch

`audit_recurrent_event_state_loop.run` applies `eligible_rows` before building
the update stream. Eligibility requires a valid 5-7h future glucose label and
fresh current glucose. This is appropriate for scoring pairs, but insufficient
for building an online event stream. Intermediate observations without a valid
future label never update this GRU. Stream membership depends on future-label
availability. This is selection dependence, not proof of future-value leakage.
Update events must be separated from scored forecast anchors.

## Synthetic reproductions

The three probes ran successfully with synthetic data only:

Reproduce with `python -m scripts.probe_event_state_contract`. Its assertions
characterize the historical defects; passing does not mean the defects are fixed.

| Probe | Result | Interpretation |
|---|---|---|
| Same patient value 100 alone versus alongside two patients valued 900, scale fixed at 100 | Its centered-current feature changes from 0 to -8 | `np.nanmedian(prepared.current)` uses the evaluation cohort; independent patient inference is violated |
| Two measurements 100 then 120, both observed masks=1 and age=0 | `new_measurement_fraction` remains 0 | The flag detects missing-to-present, not repeat measurement arrival; value/delta channels still contain information |
| Insulin rate 1 then 10 with unchanged presence/count | Encoded treatment vector difference is 0 | `_treatment_matrix` discards rate-only changes; this is not proof that raw rate coverage is complete |

These are input defects/limitations, not measured explanations of clinical MAE.
They have not been shown to affect every JEPA input pipeline.

## Actual learning contract

- A separate GRU predicts residuals on top of frozen JEPA, not online updates
  of shared JEPA latent. It receives raw summaries and the frozen prediction,
  not the full computed JEPA representation.
- Feedback is `truth - frozen_prediction`, not the previous recurrent output's
  error. This is valid frozen-bias learning, not the proposed full self-correcting
  loop. A residual objective is not inherently wrong.
- Treatment channels are five action classes' presence and log record counts,
  not dose/rate/route-resolved events. Rolling-window expiry is not necessarily
  an actual treatment stop.
- Epoch 0 is after an optimizer update. The initial zero-residual checkpoint
  is excluded from selection. Early best epochs do not rule out optimization
  or limited sequence support.

## Repair order and falsification

1. Fix event semantics: source-fitted centering, measurement identity and
   availability timestamps, updates independent of future-label availability,
   and unit-normalized dose/rate changes. Verify batch/patient isolation,
   future-label deletion invariance, and repeat-event sensitivity. Requires
   rebuilding inputs and retraining; old weights cannot silently be reused.
2. Separate frozen-bias feedback from candidate-error feedback. Record each
   prediction, ingest its outcome only when available, and include zero-update
   as a source-side comparator. State must remain patient-isolated. Genuine
   feedback can amplify errors and needs its own ablation.
3. Freeze the contract; compare frozen and corrected recurrent models on the
   same patients/anchors, with stable/changing strata and signed-correction
   diagnostics. Recalibrate intervals separately. MIMIC remains validation,
   not a new untouched lockbox. If repairs do not improve MAE, the bugs were
   real but not sufficient to explain the statistical failure.

The previous statement that the complete event-driven loop was correctly
validated was too strong. The first demonstrated mismatch precedes learning,
in constructing the update stream. Fixing it does not guarantee external
stability. causal_claim_allowed: false.
