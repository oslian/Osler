# Persistent Physiology State Point-Accuracy Gate

Date: 2026-08-28

## Question

Does one persistent, recurrent patient physiology state improve factual point
forecasting over the existing target-specific strategy?

## Frozen comparison

The audit used the canonical eICU event examples and did not modify the active
release.  Every train/test comparison used the same patients, labels,
normalization, seed, and two-epoch budget.

The candidate was compared against:

1. the original joint whole-body JEPA value decoder;
2. a per-cell ridge expert fitted on the original JEPA 311-value feature
   contract; and
3. patient persistence.

The audit used seven 70/30 patient splits, patient-equalized MAE, paired
patient bootstrap, and a held-out-hospital test.  A continuous
`target@horizon` cell had to win all seven patient splits, exclude zero in all
three paired comparisons, and preserve all three directions in the held-out
hospital.  Binary event-risk cells were not mixed into this continuous-value
test.

## Result

**Rejected: 0 of 38 evaluated continuous release cells passed the complete
point-accuracy gate.**

The persistent candidate frequently beat the bounded per-cell ridge comparator,
but that comparator is not the limiting reference.  Persistence remained the
decisive gate:

- only `potassium@24h` beat persistence in all seven point estimates;
- its pooled patient delta was `-0.0169` normalized MAE, but the 95% CI was
  `[-0.0402, +0.0052]`, so the result was not significant;
- `potassium@24h` beat the original joint JEPA in only 3 of 7 splits;
- `respiratory_rate@3h` and `respiratory_rate@6h` beat the original joint JEPA
  in all seven splits with paired CIs excluding zero, but both were
  significantly worse than persistence;
- no cell had a paired patient bootstrap interval below zero against
  persistence.

The learned persistent-state gate remained near `0.03` in every split.  The
network therefore used the new recurrent state cautiously, but the supported
correction was too small and inconsistent to improve the release.

## Interpretation

The structural hypothesis is not disproved: the implementation has one
persistent body state, persistent organ states, sparse organ communication,
predict-update uncertainty, exact algebraic readout audits, and shared
multi-step rollout.  Unit tests establish those properties.

What is rejected is the stronger performance claim: under this bounded but
formal internal experiment, those properties did not make point forecasts
more accurate than the current safety anchor.  The result must not be reframed
as promotion based on a few favorable point estimates.

## Runtime decision

- `PersistentWholeBodyJEPAWorldModel` remains `candidate_only`.
- The active 27 movable cells and 14 persistence fallbacks remain unchanged.
- No checkpoint, target head, interval calibrator, or runtime registry entry is
  replaced.
- `causal_claim_allowed` remains `false`.

The machine-readable audit is
`artifacts/persistent_physiology_state_formal_point_gate_20260828.json`.

## Remaining scientific gates

Point accuracy failed before the candidate reached the other required gates:

- independent-database lockbox;
- 90% conformal coverage;
- masked-variable recovery by related organs;
- cross-organ no-route ablation;
- long-rollout physiological and conservation checks.

Running those gates cannot rescue a candidate that has not first improved
point accuracy.  They remain useful only after a materially stronger training
signal or state-transition objective exists.
