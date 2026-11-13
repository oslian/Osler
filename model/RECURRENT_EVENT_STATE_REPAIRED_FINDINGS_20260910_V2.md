# Recurrent Event-State Loop Findings

## Decision

candidate_only. The active release is unchanged.

This audit used every available cached anchor for fixed patient roles,
including anchors without scored future labels. It is not a complete
raw-event ledger and available-time/storetime equivalence is unverified.
Own-error feedback only ingests detached candidate errors after resolution.

| Role | persistence | frozen | resolved_error_only | anchor_reset | recurrent_event_state | own_error_feedback | shuffled_event_recurrent | wrong_patient_recurrent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eicu_hospital | 40.238 | 36.265 | 35.879 | 36.466 | 36.443 | 36.436 | 36.347 | 36.842 |
| mimic | 30.130 | 28.944 | 29.218 | 29.706 | 30.975 | 30.930 | 29.251 | 31.247 |

Values are seven-seed means of patient-equal MAE on forecasts made only
after at least one previous outcome had resolved. anchor_reset has the
same event encoder and signed readout but clears recurrent state at each
anchor. shuffled_event_recurrent preserves capacity while replacing event
context with causal wrong-patient donors.

## Gate

The point screen required recurrent event-state to significantly beat frozen JEPA,
persistence, resolved-error-only state, anchor-reset, shuffled-event recurrent
and wrong-patient recurrent controls in all seven source-defined seeds on both
eICU hospital-heldout and MIMIC. Exact CIs and change/treatment subsets are in
the JSON artifact. Full promotion also needs interval and time/care-unit gates.

This remains factual prediction. causal_claim_allowed is false.
