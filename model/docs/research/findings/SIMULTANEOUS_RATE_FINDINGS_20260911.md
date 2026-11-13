# Simultaneous snapshot ambiguity

Code inspection and synthetic regression reproduce arbitrary selection of the
last source_id at equal event timestamp with unequal rates. IDs do not establish
clinical precedence. Snapshot now emits unknown for nonfinite/conflicting latest
same-dimension rates. Identical duplicates retain the rate, including true zero.
No fallback to an older rate after conflicting latest observations.

Read-only native-rate audit over5526new-query paired windows found exactly1
latest-time tie with conflicting values,1patient. Therefore this defect cannot
be presumed the main cause of widespread incremental-context failure. New
helper affects candidate extraction; no release weights or serving registry
changed. Existing immutable reports describe old snapshots. Refits must use a
fresh artifact and cannot assert exact old-input reproduction on affected rows.

Tests cover source-ID permutation, conflict abstention and duplicate true zero.
No accuracy retraining or promotion from this correctness repair.
