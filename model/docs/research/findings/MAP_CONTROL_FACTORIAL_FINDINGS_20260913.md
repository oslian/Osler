# MAP Training-Matched History/Query Factorial

All 14 mixed-arm fits completed, across seven fixed patient partitions.
31 direct artifact dependencies were hash-verified. Initialization, source
transforms, patient draws, optimizer, loss and 32-epoch tune selection match
the frozen own/control references. This is not inference-only masking.

Control means non-MAP numbers removed, with measurement support/age/presence
and native update cadence retained. It is not a full physiological input.

| History / query | Median new-patient MAE | Median hospital MAE |
|---|---:|---:|
| Own / own | 8.84140 | 10.21067 |
| Control / control | 9.23789 | 10.61140 |
| Control / own | 9.18538 | 10.58767 |
| Own / control | 8.92984 | 10.30887 |

Control-history/own-query versus own/own: significant harm 7/7 in both roles;
median paired deltas +0.37723 and +0.34456. Versus control/control, no significant
gain or harm in either role. Removing query support from training does not
repair the history-associated gap.

Own-history/control-query versus own/own: no significant gains or harms in
either role, median paired deltas +0.05329 and +0.08211. This is not equivalence
or noninferiority proof. Versus control/control it improves 7/7 new-patient and
6/7 hospital splits, with median paired deltas -0.33223 and -0.24538.

## What This Localizes

In this fixed support-control experiment, excess error follows the expanded
history representation/update path, even without additional numerical values.
It does not require expanded current-query support. The earlier fixed-weight
query-removal harm was input mismatch/reliance evidence, not proof that query
support itself caused the training gap.

History masks, ages, event flags, update counts and their interaction with the
recurrent filter are still jointly changed. This does not uniquely identify
forgetting, update frequency or a particular feature as the root cause.
Nor does it show that discarding other organs is the solution to a shared
body-state model. A repair must preserve useful cross-organ observations while
preventing unrelated observations from degrading target history representation.

This is a partial diagnostic repair relative to the control, not an improvement
over the own baseline, not a calibrated/runtime promotion, and not a complete
anatomical JEPA validation. Reused hospital data and overlapping seed cohorts
are not independent external replications. No release behavior changed.

Evidence: `artifacts/map_control_factorial_{seed}_20260912.json` for seeds
7,19,31,43,59,71,83 (protocol began September12, completed September13).
