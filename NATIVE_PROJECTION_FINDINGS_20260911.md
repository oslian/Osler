# Observation projection positive-control repair

All14 matched artificial-label fits completed;34bindings verified.
`artifacts/rate_anchor_jepa_synthetic_fast_20260911.json`.
Only observation projection first Linear lr raised1e-4->1e-3, including lab and
clock columns. Both real/support receive it; same capacity,32epochs, roles,
synthetic signs, loss, source selection. Not new architecture or clinical data.

| Artificial-label median MAE | Fast real | Fast support | Frozen whole |
|---|---:|---:|---:|
| Source evaluation | .424474 | .593412 | .582501 |
| Source bracket | .424091 | .590733 | .585315 |
| Hospital | .402314 | .580522 | .579797 |
| Hospital bracket | .438079 | .584622 | .582322 |

Fast real beats fast support, frozen whole, and persistence significantly7/7
in each group, no significant harm. Previous standard-lr synthetic real had
hospital median .580313 with real-support gain4/7; the new median .402314 is
descriptive across runs, not a paired old-vs-new CI. Strong within-run
signal-present/support-hidden contrasts establish uptake of the artificial
event increment under the modified optimization.

This supports observation-projection optimization as a contributor to the
prior known-signal failure; a disconnected path or inherent inability of this
architecture to learn event values is not the explanation for that task.
It does not isolate new event weights from all old observation columns, nor
establish optimal learning rate, complete convergence or real treatment utility.
Real selected epochs31,31,30,32,31,29,30 suggest additional convergence questions;
do not retrospectively extend only successful seeds.

Next required experiment: identical projection-lr change on original REAL
potassium labels, same native real/support controls, plus paired original
native predictions and original whole/own/persistence comparisons. No new
hyperparameter selection on hospital results. Synthetic success is not real
forecast promotion. All synthetic weights remain private and excluded from
runtime, intervals, clinical or causal claims. No serving change.
