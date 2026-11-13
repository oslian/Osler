# Expanded MAP Selected-Weight Fit Gap

Replayed all14 selected checkpoints without fitting. Original old scoring
predictions reproduce within1e-5mmHg; source-tune MAEs reproduce recorded
values. Stored event transforms retain float64 precision; casting them to
float32 in the first diagnostic attempt failed the exact-transform assertion
and was corrected before scoring. This was a diagnostic error, not a training
defect. Parent bindings verified before each replay.

Full versus own, common20k patient bootstrap, seven splits:

| Scope | Significant gain | Significant harm |
|---|---:|---:|
| All fitting-patient eligible rows | 2/7 | 3/7 |
| Original source-tune rows | 0/7 | 2/7 |
| Query rows sampled by both selected fits | 1/7 | 3/7 |
| Query rows sampled by neither selected fit | 0/7 | 5/7 |

The last two groups use exact deterministic sampler replay through each arm's
selected epoch. They refer to query rows, not independent outcomes: unselected
rows can share future measurements and histories with selected rows. They are
not a held-out-patient evaluation and cannot support deployment claims.

## Interpretation

Full context does not show a stable advantage even on actually sampled query
rows at selected checkpoints. Therefore an explanation confined to external
database shift or pure overfit of a universally successful fit signal is not
supported for this MAP trial. Harm is stronger on unsampled rows within fitting
patients, consistent with limited generalization across time points as well.

This does not isolate the sole cause. The checkpoints were selected on tune
MAE, not fit MAE; own/full selected epochs and exposure differ. These are
selected-weight diagnostics, not complete optimization trajectories. More
epochs, latent changes, and no-information claims are not established remedies.
No model was modified; no intervals, external gate or promotion occurred.

Evidence: `artifacts/map_expanded_fit_gap_{seed}_20260912.json` (seven seeds),
`artifacts/map_sampled_fit_20260912.json`. Predictions remain private.
