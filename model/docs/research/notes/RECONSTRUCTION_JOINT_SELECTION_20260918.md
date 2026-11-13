# Joint-source Selection Trajectory

All epoch scores reproduce from recorded per-group/per-target encoder-selection
MAEs. Epoch6 is correctly selected by the fixed equal-group/equal-target mean.
No evaluation labels read, no weights changed, no new selection performed.

Selected-group macro improves1.966970(initial median)->1.843224;
broad improves1.931923->1.828244. However,9/33selected-group targets and8/33broad
targets are worse than their epoch0 source-fit median on encoder-selection data.
Selected failures:calcium,CPK,fibrinogen,glucose,lipase,magnesium,MAP,RR,troponin.
Broad failures:CPK,HR,magnesium,RR,temperature,triglycerides,troponin,WBC.
Thus failure is already visible in source model selection, not exclusively
external distribution shift. Improvement sums5.5087/4.2990 dominate harm sums
1.4251/.8777 under the aggregate criterion.

This demonstrates that aggregate selection permits per-target harm. It does not
prove conflicting gradients, insufficient capacity or that changing the objective
will fix generalization. Source medians differ by group distribution; the epoch0
predictor is the same equal-group prior for both. Per-target best epochs in the
report are descriptive, optimistically selected on reused data, not deployable.

Both-group checkpoint selection alone cannot guarantee every target's benefit.
Any alternative retention-aware training/selection must be specified on source
data and tested independently, preserving one shared state and alltarget scope.
Do not graft oracle per-target epoch choices or change current gates.
Runtime unchanged. Evidence:
artifacts/native_reconstruction_joint_selection_tradeoff_20260918.json.
