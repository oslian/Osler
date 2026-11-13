# New MAP Native Training Batch

Implemented `RaggedNativeBatch`, which retains every event and query context
but constructs dense padded histories only for requested training batches.
It does not allocate a padded history for all event rows. Two unit tests check
exact equality to the existing dense assembler (including reordered global
query IDs), and reject incomplete coverage, duplicated queries and bad pointers.
Combined with the existing scheduler suite, six tests pass.

Loaded all 60 verified shards, 266,872 queries and 625,722 event-token rows plus
the padding row. Maximum history length is 388; none is truncated. A real JEPA
forward/backward/AdamW/EMA step used four eligible source-fit queries under seed
7. Treatment feature normalization used only source-fit history events. No
evaluation outcome was used. The historical deterministic dropout policy and
native observed-step filter were retained.

- Loss: 0.3202529848, finite.
- History projection gradient norm: 1.1959406137.
- History projection weight update norm: 0.0112649072.

This is an integration smoke test, not a trained comparison or accuracy result.
It does not apply/validate all source-support and numerical transformations of
the eventual matched trial, and the four examples need not be distinct patients.
It demonstrates a functioning update path, not that useful physiology is learnt.
No checkpoint was saved and runtime weights, registries and calibration remain
unchanged. Full own/full matched training remains required.

Evidence: `artifacts/map_native_training_batch_20260912.json`.
