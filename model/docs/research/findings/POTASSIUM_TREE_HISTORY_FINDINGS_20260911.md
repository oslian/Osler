# Nonlinear matched raw history result

Completed63 HistGradientBoosting fits. Same patient partitions, lawful historical
features and evaluation rows as linear and neural controls; tune-only leaf-count
selection. No internal random row validation, no external data read.

| Model/input | Median evaluation MAE |
|---|---:|
| Tree full | .370529 |
| Tree own | .377185 |
| Tree support | .378698 |
| Prior full JEPA | .374244 |
| Persistence | .437907 |

Full vs own significant improvement3/7,harm0/7; vs support improvement2/7,
harm0/7; vs JEPA improvement2/7,harm0/7. Both tree full and own beat persistence
7/7. These descriptive CIs are unadjusted and split populations overlap. Neither
all-seed stability nor independent external validation is established.

Unlike the linear raw-history failure, this nonlinear family recovers some
incremental full-input benefit under the matched contract. This is positive
evidence against asserting that all cross-variable information is unavailable.
It suggests useful interactions may not be efficiently captured by the tested
JEPA training/readout, but different loss/optimization/capacity mean it does not
prove a unique encoder defect. No learned edge or causal interpretation follows.

Retain as candidate-only benchmark. Do not adopt/ensemble based on evaluation
scores. A follow-up should freeze source-selected tree predictions as a
diagnostic comparator before external evaluation, or test a source-only
interaction-preserving JEPA objective with capacity and placebo controls.
No runtime weights, intervals, registry or promotion changed.

Protocol: `POTASSIUM_TREE_HISTORY_PROTOCOL_20260911.md`.
Artifact: `artifacts/potassium_tree_history_20260911.json` (complete=true).
Command: `python -m scripts.audit_potassium_raw_history --trees`.
