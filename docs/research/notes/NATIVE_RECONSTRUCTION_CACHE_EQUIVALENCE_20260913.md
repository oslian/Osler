# Query-Safe Group Encoding Cache

Builder now memoizes encoding by the complete canonical source-row group tuple
within one invocation. Query-asof conflict/revision scheduling is unchanged.
Different source groups at the same timestamp do not share cache entries.
Masked example tensors are copied; no mutation of cached unmasked group tokens.

15 tests pass. Benchmark reconstructs the exact previous builder source and
checks its SHA256(e0849cbe4e59f8a84be290fd99477d8db4df980df744d9c4ed06bcb14ad408e3)
before using it as reference. All example metadata, values, masked input arrays
and float64 times have identical streaming hashes on the first64 real source
anchors. Full1,877anchors reproduce all prior target counts(6,249 examples).
Full cached build takes35.07seconds in this run. Only64anchors have full-byte
reference comparison; full-count equality alone is not full-byte equivalence.

This removes redundant per-query encoding before full-cohort construction;
it is not a new physiology result, efficacy improvement or promotion. Gradient
balance across a representative multi-target batch remains unmeasured; the
prior four-example gradient smoke does not settle that question.
No weights, calibrators or registry changed. Historical bindings not rewritten.

Evidence: artifacts/native_reconstruction_cache_equivalence_20260913.json.
