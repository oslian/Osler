# Same Nonlinear Heads: Fit Versus Source-Selection Diagnostic

Fixed before execution: all47 targets/five original folds, frozen epoch10
encoder and previously fitted nonlinear readouts. No refitting, early stopping,
hyperparameter search, target routing, outer or calibration outcome inspection.
Prior33-target native fit-gap audits exist; this diagnostic concerns the NEW
body/raw/process HGB heads on their complete current fitting populations.

Read checksum-verified fit features and already verified selection predictions.
Verify fit/selection patient and component disjointness in each original fold,
complete target support and model PARAMETERS. Population is the original
unweighted fitting median, reproduced against the selection constant. Also
report the fitting-weighted median constant to diagnose absolute-error objective
adequacy, without substituting that baseline into any existing promotion gate.

Execution correction before any complete target result: the original median
uses float64 centers, while native cached labels first pass through float32.
The initial direct median equality assertion failed on albumin by4.3e-8.
Reproduce the original center-derived median and separately verify EVERY cached
fit label against float32(center) followed by original normalization. Do not
relax the old baseline equality or substitute a query-derived constant. Preserve
the failed first contract; corrected diagnostic writes a separate v2 directory.

Fit predictions: native, body64 HGB, raw265 HGB, process265 HGB (first106 numeric
columns zeroed), population and trained-shuffle HGB on real individual states.
For the shuffle head, additionally evaluate its actual training donor states
using the original within-component cyclic assignment and original seed+target
index. Label this separately: own-state fit error is not its training objective.
Do not compare shuffled own-state error as though those input-label pairings
were used to optimize that null.

Report patient mean, equal-component mean and bounded-weight mean absolute
error, mean signed error and central/intermediate/tail contributions using
the same fit-normalized truth bands <=1,(1,3],>3 as the previous diagnostic.
Fitting weights use the existing [.5,2] scheme. Selection weights are computed
from its own components only for descriptive reporting, not fitting or routing.
The two weighted populations differ; cross-role MAE gaps confound case mix.

The interpretable contrast is whether a head improves its comparator within
fitting data yet loses that SAME comparator within selection data. Retain every
fold and target, and distinguish fit/query benefit signs without statistical or
unique-cause claims. Fold macro means are descriptive: folds overlap, and fitting
patients are not independent observations when repeated across folds.

No training-error success can authorize promotion. Fit-to-selection reversal
supports an overfit/transport warning, not proof of memorization or the sole
root cause. Failing to beat native on fit is not proof of an optimizer bug:
native and HGB function classes and objectives differ. Preserve all original
gates/runtime/weights/intervals and keep patient data local.
