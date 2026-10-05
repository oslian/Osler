# Pre-Anchor Care Context Routing Audit

## Result

The frozen whole-body JEPA and target experts were not retrained. Timestamped
POE intent, bedside assessment and observed treatment history were appended only
to the persistence utility selector. Unit-safe time-weighted, current and delivered
dose/rate features are included where both databases support the dimension. Every
event and delivered amount is truncated at or before its anchor.

| Target | eICU context - no-context MAE | MIMIC context - no-context MAE | MIMIC benefit AUC | MIMIC move | Promotion |
|---|---:|---:|---:|---:|---:|
| creatinine@6h | +0.000000 | +0.000001 | 0.531 | 0.000% | False |
| bun@6h | +0.000000 | +0.000000 | 0.520 | 0.000% | False |
| urine_output@6h | -0.637281 | -0.043698 | 0.524 | 0.017% | False |
| potassium@6h | -0.002236 | -0.000043 | 0.491 | 0.100% | False |
| bicarbonate@6h | +0.000000 | +0.000000 | 0.518 | 0.000% | False |

No target passed the full eICU patient, unseen-hospital and MIMIC-IV gates.
The target experts themselves improved eICU point MAE, but the utility selector
mostly abstained. External benefit AUC ranged from approximately 0.49 to 0.54,
which is too close to chance to identify when moving beats persistence.
Urine output and potassium had slightly lower MIMIC point MAE than the optimized
no-context selector, but neither beat every reference or its shuffled-context
placebo. This is evidence against promoting this bounded context representation, not evidence
that POE or bedside values are intrinsically uninformative.

## Boundary

This is factual context, not a treatment-effect estimate. MIMIC-IV was previously
inspected and is validation data rather than an untouched lockbox. A candidate must
beat persistence, the always-moving expert, the optimized no-context router and an
equal-width patient-shuffled context placebo. Failure leaves the release unchanged.
