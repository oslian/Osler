# Snapshot correctness repair does not improve accuracy stably

Twenty-one matched refits complete, same native-rate queries and fixed settings.
Frozen old snapshots reconstructed solely for control and all old real MAEs
reproduced; own refits reproduce exactly. Repaired simultaneous-rate snapshot
is used for new real/support fits. No retrospective ambiguity selection policy.

New vs old significant gain0/7 and harm0/7 all source/hospital groups. Median
hospital all .393562->.394694; bracket-after .406434->.407396. Some seeds are
exactly unchanged. One ambiguous input can affect fitted tree splits, but its
repair does not explain the main incremental-context gap. Keep the semantic
repair, do not claim predictive improvement or revert for flattering MAE.

No runtime, interval or promotion change. Artifact
`artifacts/rate_snapshot_repair_20260911.json`; private new weights separate.
