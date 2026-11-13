# MAP Neural Versus Tree Task Scope

Checked the actual training runner and private eligible query tables, not only
the findings titles. Expanded native neural training reads the original
`queries.parquet` and `map_all_query_labels.npz`: 33,477 eligible insulin-record
triggered anchors. The recent native-MAP tree task uses 205,723 eligible
measurement-availability anchors. Exact same (stay, query time) eligible pairs
overlap at only 3,905 anchors, 1.898% of the new task.

Therefore the recent missing-value, event-weight, derived-gap and support-tree
tests have NOT trained JEPA on the new MAP-native task. They are valid raw
readout diagnostics but cannot substitute for that neural experiment. They
also do not test the complete persistent anatomical body-state architecture.

The earlier native neural runner uses explicit source transforms, event groups,
EMA delta targets and own/full arms, but a different query distribution. Its
negative result remains scoped to that task. `configure` deliberately uses
eval/dropout-off for that historical matched policy; do not label this an
accidental frozen-gradient bug or silently change it during a time-task test.

The actionable missing bridge is a matched native-event history cache at the
new queries plus a source-only neural comparison, retaining patient partitions,
the prior training policy and actual availability. New queries cannot simply
index old ragged histories: their cutoffs differ. Existing sharded snapshot
inputs are not native-event histories. This is not a new promotion or a claim
that retraining will succeed. No active model changed.

Evidence checked: `scripts/train_map_expanded_native.py`,
`scripts/build_map_all_native_tokens.py`, `scripts/build_map_native_anchor_inputs.py`,
and the corresponding private query/label arrays. Counts computed by an exact
one-to-one join; no patient data included here.
