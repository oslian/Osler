# Update Invariance and Feedback Attribution Findings

## Decision

Both prespecified diagnostics completed for glucose@6h across seven source
seeds. No promotion or runtime/registry/weight/interval change. Results describe
the experimental own-error recurrent residual head, not every deployed model.
MIMIC is repeatedly inspected validation, not an untouched lockbox.

## Documentation-density stress test

Fixed weights, same actual observations and scored anchors. Extra consumption
of identical encoded events at identical timestamps creates no extra truth,
pending forecasts or error-memory updates. Effects below occur at later original
anchors. This is duplicate-ingestion sensitivity, not an estimate of the amount
of duplication in the actual data. No raw-ledger duplicate prevalence audit was
performed, so it does NOT establish the sole cause of the existing transfer gap.

Seven-seed means (glucose mg/dL; absolute shifts are row-weighted):

| Role | Copies | Mean absolute shift | Mean per-seed p95 shift | Rows shifted >1 | Seeds with significant MAE harm |
|---|---:|---:|---:|---:|---:|
| eICU hospital | 2 | 1.267 | 2.906 | 40.7% | 3/7 |
| eICU hospital | 4 | 2.233 | 5.279 | 47.6% | 3/7 |
| MIMIC | 2 | 0.864 | 2.146 | 30.8% | 5/7 |
| MIMIC | 4 | 1.520 | 3.813 | 41.3% | 5/7 |

The zero-update checkpoint at seed43 is exactly invariant; the six nonzero
checkpoints are sensitive. MAE comparisons use patient-cluster paired bootstrap,
not row independence. Counts are exploratory, without multiple-testing correction.

First localized mechanism: the GRU applies another transition on every ingestion;
event identity is not used to make repeated ingestion idempotent. A subsequent
repair should enforce event-identity deduplication before recurrent updates and
separate elapsed-time evolution from new measurement assimilation. Do not simply
discard all equal numerical values: independent repeated measurements can agree.
This test does not prove that adding real fresh observations is harmful.

## Immediate feedback attribution

Fork the same pre-anchor hidden state and error memory; one fork ingests newly
resolved candidate errors, one withholds only those errors for this prediction.
Discard the withheld fork afterwards. This isolates immediate computational
effects, NOT accumulated historical-error harm or a retrained no-feedback policy.

| Role / scope | Patients / rows | Normal MAE | Withhold new feedback MAE | Frozen MAE |
|---|---:|---:|---:|---:|
| eICU personalized | 194 / 386 | 36.433 | 36.491 | 36.265 |
| MIMIC personalized | 462 / 1300 | 30.940 | 30.963 | 28.944 |
| eICU feedback arrives after snapshot change | 74 / 111 | 42.726 | 42.831 | 43.785 |
| MIMIC feedback arrives after snapshot change | 336 / 788 | 31.178 | 31.203 | 30.207 |

Withholding has no significantly beneficial seed in either primary role.
It significantly harms 3/7 eICU and 4/7 MIMIC primary comparisons. Thus newly
arriving feedback has a small beneficial immediate contribution on average;
removing it does not resolve the much larger recurrent-versus-frozen gap.
This does not rule out errors accumulated earlier in hidden state.

There are 295 eICU and 819 MIMIC primary feedback-arrival rows. The median
forecast-origin-to-ingestion age is 15.85h and 6.17h respectively. These are not
the target's nominal horizon: feedback can wait until the next cached anchor.
The MIMIC no-detected-treatment-change subgroup has only22 patients /31 rows,
below the existing bootstrap support threshold; its CI is correctly absent.
Snapshot changes include knownness/completed-dose transitions and are not
verified treatment starts/stops. Do not interpret subgroup differences causally.

## Verification and next discriminating repair

All seven fitted selected epochs reproduce the final repaired own-feedback run.
Diagnostic replay is asserted equal to production replay before perturbation.
Tests cover that parity, delayed feedback, no duplicate outcome ingestion,
unchanged first prediction under extra post-anchor consumption, and patient
isolation. Only the preparation function was extracted for reuse in the existing
audit; serving and model code are unchanged. Full regression: 274 tests passed;
`git diff --check` passed; CodeGraph synchronized.

Next: use stable raw event identities and available times to distinguish true
remeasurement from re-documentation. Train/evaluate an event-idempotent candidate
on the same patients, retaining all real observations; compare against this
original and frozen JEPA. A synthetic stress-test fix alone is not promotion.

Artifacts: `UPDATE_INVARIANCE_PROTOCOL_20260910.md`,
`artifacts/update_invariance_20260910.json`,
`scripts/audit_update_invariance.py`, `tests/test_update_invariance.py`.
