# Recurrent Event-State Loop Findings

> Interpretation correction (2026-09-10): the numbers below are unchanged, but
> the claims of a complete, correctly validated event loop and exclusion of
> undertraining are not established. This tested an eligible-anchor summary
> GRU with frozen-model error feedback. See
> `docs/research/notes/EVENT_STATE_CONTRACT_DIAGNOSIS_20260910.md` for reproduced input defects.

## Decision

candidate_only. The active release is unchanged.

This audit tested the complete causal-order loop: anchor-known measurement
and treatment events updated a patient-scoped recurrent state; that state
directly decoded a signed residual; forecast outcomes entered the input only
after their true measurement time had resolved.

| Role | persistence | frozen | resolved_error_only | anchor_reset | recurrent_event_state | shuffled_event_recurrent | wrong_patient_recurrent |
|---|---:|---:|---:|---:|---:|---:|---:|
| eicu_hospital | 40.238 | 36.265 | 35.879 | 36.514 | 36.202 | 36.399 | 36.275 |
| mimic | 30.130 | 28.944 | 29.218 | 29.688 | 29.347 | 29.449 | 29.391 |

Values are seven-seed means of patient-equal MAE on forecasts made only
after at least one previous outcome had resolved. anchor_reset has the
same event encoder and signed readout but clears recurrent state at each
anchor. shuffled_event_recurrent preserves capacity while replacing event
context with causal wrong-patient donors.

The recurrent model beat persistence in 7/7 eICU hospital comparisons but
0/7 MIMIC comparisons. It beat frozen JEPA in 0/7 comparisons in both roles,
resolved-error-only state in 0/7, anchor-reset in 1/7 eICU and 3/7 MIMIC,
and shuffled-event recurrent in 2/7 in both roles.

## Failure location

The complete loop exists and preserves causal ordering, but the learned update
does not transport. On changed eICU rows, recurrent MAE was 50.494 versus
51.298 for frozen, while stable-row MAE worsened from 14.167 to 14.913. In
MIMIC it was worse in both changed rows (45.590 versus 45.527) and stable rows
(16.566 versus 16.255).

Treatment-transition direction accuracy also did not improve: 0.653 versus
0.667 for frozen in eICU and 0.793 versus 0.798 in MIMIC. Shuffled events were
statistically competitive, so the available harmonized event representation
does not identify a transferable signed update. Source tune selection stopped
at epoch 0-3 in every seed, which argues against undertraining and is consistent
with weak, unstable incremental signal.

## Gate

Promotion required recurrent event-state to significantly beat frozen JEPA,
persistence, resolved-error-only state, anchor-reset, shuffled-event recurrent
and wrong-patient recurrent controls in all seven source-defined seeds on both
eICU hospital-heldout and MIMIC. Exact CIs and change/treatment subsets are in
the JSON artifact.

This remains factual prediction. causal_claim_allowed is false.
