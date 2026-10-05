# Seed19 Replication: Corrected First Epoch

Current report: `artifacts/native_reconstruction_replication_v2_19_epoch1_20260914.json`.
Use `scripts/train_reconstruction_replication_v2.py` for continuation, NOT the
initial driver. The initial seed19 process-control branch trained on numbers
but evaluated neutralized numbers and is invalid. See the immutable repair
contract `REPLICATION_PROCESS_WIRING_REPAIR_20260914.md`.

V2 reran all three arms from initialization. All fit/tune calls use the same
input-view helper; five partition/wiring tests pass. Real and placebo logs
exactly reproduce the unaffected original arms. Source dependency hashes verify.
The corrected process training loss differs from real, as expected.

Seed19 recomputes patient roles from its partition (942fit318tune309evaluation,
194hospital). It does not reuse old shard-role assignments or seed7 trained
weights. Thirty-nine source-eligible targets; 26,264fitpairs consumed once or
more in each arm's36,738updates.

| Arm | Initial source tune | Corrected epoch1 tune |
|---|---:|---:|
| Real | 1.914917 | 1.845440 |
| Process | 1.914917 | 1.899023 |
| Placebo | 1.914917 | 1.914907 |

Normalized source-selection errors only. These numbers are not comparable to
seed7 absolute scores because patients changed, and do not establish replication.
Seven epochs remain for every arm; optimizer/RNG checkpoints are saved.
No heldout accuracy inspection, future forecasting, interval or runtime changes.
The new-driver wiring defect does not invalidate the prior seed7 process run,
which neutralized the sample tensors before both training and selection.
