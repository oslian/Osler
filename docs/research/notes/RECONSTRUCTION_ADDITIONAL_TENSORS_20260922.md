# Additional-Domain Training Tensor Preflight

Reused the pre-existing fixed 38-person parity pilot, without selecting on
outcomes. Its 847 accepted anchors belong to 37 people; the person with no
accepted anchors was not replaced. Reused the original reconstruction builder
and checkpoint module mapping, but no learned weights or old normalization.

All 847 requested examples were rebuilt. Raw labels match the full scaling
centers within float32 conversion tolerance. Complete target and sibling
history masks and independent event counts match earlier acceptance. All
4,235 fold/example transforms are finite and preserve every non-value tensor
channel exactly. No observed input lacks fit-label normalization support in
this pilot. This does not establish full-population input support.

Raw tensor payload is 12,774,960 bytes, median history 10 events and maximum
226 events. Multiplying the pilot mean by the full sample count gives 25.29GiB
before serialization/index overhead. This is illustrative, NOT a representative
storage estimate; the pilot deliberately samples components equally. Do not
blindly create five copies of the full history cache. A raw shared cache with
fit-specific transforms is the existing viable direction; capacity must be
checked before materialization and long histories remain untruncated.

An empty accepted-site query exposed a Parquet empty-set type error. Empty
sites are explicitly skipped without resampling. The full pilot then finished
successfully. No model training, accuracy claim, runtime change or promotion.
Next: complete input-support audit and full history preparation with bounded
storage, followed by the fixed matched physiology/process training protocol.

Report: `artifacts/native_reconstruction_additional_tensor_preflight_20260922.json`.

## Lossless Cache Follow-up

Added an indexed SQLite patient/target cache using numpy compressed arrays
with pickle disabled. Metadata bindings reject incompatible resumes; per-shard
transactions do not mark incomplete writes complete. Duplicate patient/target
rows and payload corruption fail closed. Read-only access is enforced by SQLite.
Two unit tests cover exact dtype/value round trips, zero preservation,
rollback on close, duplicate rejection, read-only writes and corruption.

The same fixed pilot completed again: 847 examples, 4,235 fold transforms,
and exact round trips of tokens, float64 event times, labels and anchor times.
Full database size including index overhead is 2,150,400 bytes versus
12,774,960 raw tensor bytes. No histories were truncated or downcast. This is
a pilot storage result, not a representative full-population compression rate.
The read-only cache can be shared across all folds; their scalers remain
separate. Full materialization and training remain pending.

Updated code is bound by the new report; the earlier preflight report preserves
its historical script hash rather than being rewritten.
Report: `artifacts/native_reconstruction_additional_tensor_cache_20260922.json`.
