# Joint Source Training Protocol

Question: can one shared body model retain the selected population while learning
the broad population, without an inference-time retrospective cohort selector?

Use unchanged sample shards from both matched sources:1072fit/180encoder patients,
original33trained targets; retain all39catalog statuses. No evaluation, confirmation
or reserved patient enters fitting/selection. Source-group tags control sampler and
reporting only, never encoder input or serving decisions. Do not combine histories
across patients or merge conflicting cached observations.

Fixed eight epochs, seed19, same architecture/optimizer/mask recipe. Each target
gets536draws per source group per epoch, covering all pairs before cycling;
35376updates/epoch. Real/process/shuffled-label arms share exposure budgets.
Shuffled labels stay within source group and target, preserving population priors.
Readout starts at equal-group source median under absolute loss; no labels outside
fit. Select checkpoint by equally weighted group and target encoder-selection MAE,
epoch0 eligible. Always retain per-group per-target results, not just aggregate.

This doubles people and compute relative to the single-source experiment: cannot
attribute improvement uniquely to mixture versus compute. Evaluate frozen selected
checkpoints on BOTH original178evaluation patients and expanded3433confirmation
patients with the same controls and corrected tests. Report harms independently;
do not hide a minority under overall gains. Reused groups are not new lockboxes.
No runtime promotion, relaxed gates or fabricated intervals. This protocol is a
preparation artifact; no joint-source training has completed yet.
