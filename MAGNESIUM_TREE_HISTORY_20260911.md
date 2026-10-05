# Nonlinear raw history diagnostic

Same fixed-lag raw input, labels, patient roles, patient weights as ridge.
Existing HistGradientBoosting absolute-error configuration, source-tune
leaf-size choices7/15/31; own/full/support x7seeds. No external selection.

| Median MAE | Own | Full | Support | Persistence |
|---|---:|---:|---:|---:|
| Source evaluation | .219218 | .224107 | .219800 | .277679 |
| Hospital | .179137 | .185421 | .182835 | .238434 |

Full-own gain0/7both,harm1/7source2/7hospital. Full-support gain0/7harm0/7both.
Both own/full persistence7/7both. Nonlinearity improves the raw ridge baseline,
but does not establish extra-context gain. Not proof all possible nonlinear
models or raw event histories lack information. No JEPA replacement, calibration
or runtime promotion. Artifact magnesium_tree_history_20260911.json.
