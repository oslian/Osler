# Cross-Variable History Findings

Date: 2026-09-07  
Decision: `candidate_only`; raw cross-variable history did not justify encoder retraining  
Runtime changed: no

## Question

Does causal, measurement-pure history from other physiological variables add
future information beyond a patient's own target history when cohort, label,
split, estimator family, and evaluation are held fixed?

The protocol was frozen in `CROSS_VARIABLE_HISTORY_PROTOCOL_20260907.md` before
the outcome report was generated.

## Comparison

For `glucose@3h` and `map@3h`, every learned method received the same target-own
24-hour trend features. The candidate additionally received current non-target
physiology and valid normalized changes at 1, 3, 6, 12, and 24 hours. LOCF
copies were excluded as new measurements. The capacity-matched placebo used the
same feature width but substituted another patient's non-target history.

Feature widths were:

| Method | Width |
|---|---:|
| Target-own history | 7 |
| Current cross-variable state | 163 |
| Cross-variable history | 683 |
| Wrong-patient history placebo | 683 |

## Results

### eICU patient-heldout

| Target | Own-history MAE | Cross-history MAE | Candidate minus own history, 95% CI | Stable-state harm, 95% CI | Point gate |
|---|---:|---:|---:|---:|---:|
| Glucose | 28.237 | 29.097 | +0.860 `[+0.052, +1.812]` | +3.146 `[+1.743, +4.579]` | Fail |
| MAP | 9.960 | 9.972 | +0.012 `[-0.129, +0.162]` | +0.636 `[+0.188, +1.097]` | Fail |

The first performance failure therefore occurred on source patient-heldout
data. Glucose was significantly worse than target-own history. MAP was
indistinguishable in overall MAE but significantly harmed stable cases.

### eICU hospital-heldout

| Target | Own-history MAE | Current-cross MAE | Cross-history MAE | Placebo MAE | Point gate |
|---|---:|---:|---:|---:|---:|
| Glucose | 34.400 | 34.028 | 34.874 | 34.135 | Fail |
| MAP | 9.623 | 9.684 | 9.649 | 9.639 | Fail |

Neither target beat target-own history, current-only cross-variable state,
release, and placebo together. Risk and interval gates also failed for both
targets.

### NWICU diagnostic

| Target | Own-history MAE | Cross-history MAE | Release MAE | Point / risk / interval |
|---|---:|---:|---:|---:|
| Glucose | 30.747 | 30.863 | 30.194 | Fail / fail / fail |
| MAP | 8.672 | 8.764 | 8.904 | Fail / fail / fail |

NWICU was evaluated only after candidate weights were frozen. It was already an
opened dataset and was used as a transfer diagnostic, not as a new lockbox.

## Interpretation

For these two high-priority 3-hour targets, the negative result precedes the
shared encoder. Raw pre-anchor non-target history did not provide stable
incremental information once the target's own history was known. This rules out
the narrow explanation that the frozen JEPA alone discarded an otherwise
strong cross-variable history signal.

It does not show that cross-variable physiology is absent. It shows that the
available 1-24 hour tabular history, under the current observation and label
contract, cannot reliably improve these future targets. Unobserved treatment
intent, interventions after the anchor, higher-frequency events, and genuine
patient heterogeneity can still determine the future.

## Decision

- Do not retrain an end-to-end observation-aware encoder for `glucose@3h` or
  `map@3h` on the strength of this input block.
- Do not modify release weights, source selection, intervals, registry, or
  serving behavior.
- Any later encoder experiment must first identify a target/horizon where raw
  cross-variable history passes a source-only incremental-information screen
  against own history and a wrong-patient placebo.
- Keep separate research directions for high-frequency pre-event information,
  treatment intent, and longer-horizon targets; this result does not evaluate
  those contracts.

Machine-readable evidence:
`artifacts/cross_variable_history_20260907/report.json`.

