# New rate record does not locate the change instant

Verified timing ambiguity: consecutive unequal records at10 and30 with draw20
only bracket an observed difference across the draw. Treating30 as actual change
time is unsupported. Added explicit observed-pair bracket classification and
boundary/unchanged/duplicate-order tests. This does not reconstruct delivery.

Among554 previously post-draw-recorded change windows,439 only straddle the draw,
115 have at least one unequal observed pair entirely at/after draw (98patients).
Hospital43windows/40patients become33straddle-only/31patients and10entirely-after
windows/9patients. Patient counts need not be additive across window groups.

Even entirely-after observed pairs do not establish exact clinical change time,
continuous infusion, or real-time record availability. Existing score comparisons
remain numerically valid for their record-based group, but are not an experiment
on known post-draw treatment intervention. Strict group currently lacks patient
support for the existing bootstrap minimum; do not declare model efficacy failure.

No fitting, forecast repair or runtime changes. This localizes a test-contract
limitation, not the unique cause of whole-body model failures. Future event update
work must preserve interval uncertainty rather than inventing a change instant.
Artifact: `artifacts/rate_measurement_brackets_20260911.json`.
