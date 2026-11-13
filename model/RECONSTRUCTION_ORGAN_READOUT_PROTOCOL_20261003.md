# E23: Does the current-value head omit useful persistent organ state?

Status: fixed before execution. Diagnostic repair candidate, not runtime promotion.

E22 located known-content readability differences after body assimilation. E21
showed that better retention need not improve masked inference. The current
NativeBodyReconstruction head reads only state.body, although replay also retains
state.organs. This experiment tests the actual masked current-value task, not
recall of observed values, future forecasting, or clinical physiology causation.

## Fixed design

- Reuse all five E21 folds and their exact hash-selected fit/selection contexts.
  Keep the original epoch10 encoder and state layer frozen. Preserve the complete
  lawful history, target/sibling block mask, event times, and integration steps.
- Replay the entire state. Require original body and native prediction parity
  before fitting any reader. Never infer observations for missing targets.
- Inventory all 53 targets; fit a target-fold only with at least 30 fit and five
  selection patients. Selection support alone, never outcomes, determines eligibility.
- Five representations, each with identical 159 process features:
  body64; body64+flattened anatomical organ states;
  body64+shuffled organ states; body64+random linear expansion of body64;
  original raw265 (already includes the same process features).
- The three augmented arms have identical widths. Random expansion is a fixed
  Gaussian matrix scaled by sqrt(64), seed 20261003+fold, and adds no information
  beyond body. It controls for extra coordinates but is not proof of identical
  effective statistical capacity. Placebo permutes entire organ tensors among
  distinct patients with the same masked target, independently within each role;
  it never draws from fit into selection, uses labels, or unmasks siblings.
  It does not condition on source component, so it does not isolate physiology
  from all source/workflow effects.
- Refit Ridge(alpha=100, cholesky) with fit-only weighted StandardScaler and the
  existing HGB100 PARAMETERS; existing bounded source weights. No tuning, early
  stopping, new seeds, new loss, or selecting the more favorable reader afterward.
- Save all fitted readers and predictions privately; reproduce them after reload.
  Fit errors are descriptive diagnostics, not performance validation.

## Assessment

For each reader compare real organ augmentation with body, shuffled organs,
body expansion, and raw: eight contrasts, fixed 53-target family. Average repeated
fold losses for each patient before patient and source-component paired bootstrap.
Screen: >=30 distinct patients, >=3 components; both 95% paired CI upper bounds
below zero and one-sided p < .05/(53*8), 5000 repetitions, fixed seeds.
Joint-reader agreement is required for a claim of reader-robust evidence.
Report every supported target and all failures; no promotion from a mean alone.

Interpretation:
1. Real organs beat body and both equal-width controls: locally useful state
   compartments were omitted by the body-only readout.
2. This holds but raw remains better: partial representational repair only.
3. No reliable benefit: close this fixed direct-organ-readout branch, not all
   organ-state learning. Reader limitations and finite support remain possible.
4. Conflicting readers or source groups: retain that uncertainty explicitly.

This reused development sample is bounded, not the full cohort or an untouched
external lockbox. No release weights, registry, intervals, or serving changes.
This test cannot establish that body updates irreversibly erase information.

