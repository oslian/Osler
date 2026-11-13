# Observation projection optimization control

Same native synthetic +/-0.5 task, seeds, source partitions, tune-only selection,
32epochs and EMA/value objective as NATIVE_SYNTHETIC_PROTOCOL_20260911.md.
Only change first observation Linear weight/bias AdamW lr1e-4 to1e-3. Both real
and support arms receive this rate. Other parameters retain1e-4. This includes
old lab/clock columns, not just newly initialized event columns. Same model
capacity, no decoder or state bypass. No assertion that10x is optimal.

Primary real-versus-support patient bootstrap on artificial labels; frozen
whole and persistence secondary. Previous synthetic aggregate is descriptive
unless an explicitly paired comparison is subsequently run. Do not interpret
synthetic improvement as biological efficacy. Private fast_* artifacts only,
no runtime, calibrator, promotion, or causal changes. No new external lockbox.
