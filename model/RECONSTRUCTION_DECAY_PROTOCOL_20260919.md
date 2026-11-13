# Decay-Only Accumulation Trial

Freeze the accumulated sqrtLR experiment's data,33heads, shared model, seed19,
patient queues,528-example blocks,536steps/eight epochs, Adam betas and clipping.
Change only weight_decay from.01 to matched_decay(1e-4,.01,528,sqrt(528)*1e-4),
approximately.229721969. This matches decay-only exposure for always-active
parameters; does not match skipped-update tensors or sequential Adam moments.
Retain all epoch checkpoints and select by the same90+90sourcepatient macroMAE,
including epoch0. Only one new arm; no data-dependent rate search.

Freeze selection before evaluating the same previously inspected cohorts.
Primary comparison is the original accumulated sqrtLR candidate; also report
sequentialexpanded, old and population. No per-target retrospective selection.
Process/placebo accumulation controls and new external lockbox remain untested,
so no runtime promotion from this isolated real-arm experiment. 33trained and
6untrained target statuses preserved; no futureforecast or interval claim.
