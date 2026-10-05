# Auxiliary Own-History Diagnostic

Before results: compare all 14 auxiliary targets plus glucose using the same
source-fit/tune subjects and measured pairs as the completed auxiliary training.
Reuse fixed checkpoints. Add Ridge(alpha=10) residual controls from target-only
current value/mask/age and target-only 10-lag value/mask/age. Use inherited
normalization, patient-equal fit weights normalized to row count, normalized
delta labels, and no hyperparameter selection. All input snapshots are
availability-safe. No other-variable, future, hospital or subject identifier
features. Patient identifiers only define splitting and weighting.

Require 30 tune patients for patient-paired bootstrap comparisons. Report all
targets/seeds, not just potassium. No external reads, no runtime changes.
This is a lower-complexity reference, not capacity-matched neural ablation:
winning cannot alone prove cross-organ information, and losing does not prove
information absence. Source-tune also selected the glucose checkpoint and is
not an independent final evaluation. No promotion from this diagnostic.
