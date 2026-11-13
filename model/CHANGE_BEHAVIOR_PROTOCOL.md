# Frozen Change-Behavior Diagnostic

## Purpose

Supplement, never replace, the existing persistence/router and external gates.
Do not train, select a model, change release intervals, or issue a promotion.
Ask separately whether the release avoids unnecessary movement in stable cases,
recognizes signed changes, and honestly describes uncertainty in each group.

## Fixed Design

- Replay the current source selector with its bound weights and calibrators.
- Use canonical measured input/history only. No future values, labels, or care-unit
  identity enter inference. Actual changes define retrospective scoring strata only.
- Use exact/strict measured future pairs with a future event time after the anchor
  and positive actual delta_t. Do not silently accept older nominal-only labels.
- Keep one hash-chosen eligible anchor per patient/cell, including all history in
  the same stay. Selection does not depend on change size. Report missing support.
- Cover the 39 continuous catalog cells where labels support them. Hypoxemia's two
  interval-event cells require their original event labels and separate probability
  scoring, not a single future oxygen measurement substituted as their outcome.
- eICU canonical grid has six horizons; the available strict NWICU input is 3h.
  MIMIC patient-exclusion caches are unavailable at their recorded location, and
  the located transition tables lack this per-row quality/timestamp contract.
  Do not describe a replay on previously inspected databases as a new lockbox.
- Thresholds are explicit research effect sizes in canonical units, not diagnostic
  criteria or clinical minimum important differences. The existing kidney audit's
  thresholds are reused. Others: glucose 20, HR/MAP 10, RR/O2sat/Hct/Na/Cl 3,
  hemoglobin 1, platelets 20. Use half/double sensitivity without picking a winner.
  Stable means |future-current| below this threshold, not healthy or safe.
- A false move is a threshold-sized prediction on a retrospectively stable case.
  Signed change recall requires both threshold-sized movement and correct direction.
  Missing predictions count as misses in recall; persistence has zero change recall.
- Report patient-equal MAE, paired difference vs persistence, direction, magnitude
  and signed bias, not a single pooled accuracy. Bootstrap whole patients with
  replacement; CI requires 30 patients. No multiplicity-adjusted efficacy claim.
- Report point/interval availability, conditional coverage, width, interval score,
  and covered fraction of all evaluable cases. Missing intervals are NOT zero width.
  Central interval score at alpha=.1 is width + 20 times distance outside bounds;
  this is not CRPS because the release does not supply a complete predictive CDF.
- Width-based narrow/wide groups use a frozen threshold (width <= 2*change threshold)
  without consulting outcomes. No outcome-based abstention is allowed.
- Retain [0.87,0.93] as a descriptive coverage band. Low subgroup support is unknown,
  not pass. These post-hoc group diagnostics do not recalibrate or revoke weights.
- Care-unit results are slices of frozen replay, NOT new hospital-heldout training.
  eICU development overlap is not excluded; this is a behavior diagnostic, not a
  confirmatory generalization result. Measured-label selection remains a limitation.

## Reproduce

```bash
python -m unittest tests.test_change_evaluation
python -m scripts.audit_change_behavior --max-patients 500 --bootstrap 1000
```

Override `--eicu-grid`, `--nwicu-3h`, `--model-dir`, and `--output` for local paths.
Only aggregate output is written, with input/model/code hashes. Patient rows are
held in memory and never committed. No causal/clinical or next-measurement claims.
