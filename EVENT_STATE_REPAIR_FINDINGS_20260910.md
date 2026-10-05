# Event-State Repair Findings

## Decision

Implementation defects were repaired; transferable self-learning was NOT
established. Candidate-only. Runtime weights, registry, intervals and serving
behavior are unchanged. No causal claims or new untouched-lockbox claim.

## Implemented and tested

- Source-frozen centering instead of evaluation-cohort centering.
- All available cached anchors update state, independent of future-label
  eligibility. Only valid fresh 5-7h pairs contribute training loss/scoring.
- Timestamp-derived repeat-measurement detection. Age is no longer clipped
  at six hours; float64 age arithmetic avoids epoch-hour rounding artifacts.
  Unknown/invalid measurement age is not silently treated as measured.
- Explicit, dimension-separated last-known rate and completed-dose features,
  with known masks. Zero rate is distinct from unknown. eICU rate provenance
  distinguishes drugrate from infusionrate; only unambiguous drugrate units
  are used. Bag content is not treated as delivered dose.
- Detached candidate-output error feedback is ingested only after its outcome
  time. Separate frozen-error feedback remains an experimental comparator.
- The untrained zero-residual checkpoint participates in source tune selection.
- Candidate point-screen success cannot set runtime promotion: interval and
  time/care-unit gates remain mandatory.

`tests/test_event_state_contract.py` and `scripts/probe_event_state_contract.py`
cover independent patient inference, repeated versus carried measurements,
stale-age precision, label-free update nodes, unknown units/zero rates,
future-rate exclusion, unfinished-dose exclusion, delayed own-error feedback,
and exact zero-update behavior. These tests do not establish clinical benefit.
The final complete regression suite passed 271 tests.

## Final statistical comparison

Seven source-defined seeds; patient-equal glucose MAE (mg/dL), lower is better.
Both final runs share the same implementation and residual scale. The scored
patient subsets and frozen/persistence predictions are unchanged between runs.

| Method | eICU hospital | MIMIC |
|---|---:|---:|
| Persistence | 40.238 | 30.130 |
| Frozen JEPA | 36.265 | 28.944 |
| Historical Sept 9 recurrent candidate | 36.202 | 29.347 |
| Repaired, scored-anchor updates only | 36.395 | 30.239 |
| Repaired, all cached-anchor updates | 36.440 | 30.985 |
| Own-error feedback, scored-anchor updates | 36.383 | 30.190 |
| Own-error feedback, all cached-anchor updates | 36.433 | 30.940 |

Historical versus repaired is a bundle comparison, not attribution of an
individual repair: treatment dimensionality and source-scale support changed.
The two final repaired runs do isolate cached-stream inclusion at fixed scale
and training recipe. Their mean difference is descriptive, not a separate
paired CI claim. Both repaired and own-error candidates beat frozen in 0/7
patient-bootstrap comparisons in BOTH domains. Full-stream recurrent beats
persistence 7/7 in eICU but 0/7 in MIMIC. No new cell is promoted.

Full-stream support:

| Role | Patients | Update anchors | Scored anchors |
|---|---:|---:|---:|
| eICU development | 398 | 1,732 | 846 |
| eICU hospital | 372 | 1,513 | 797 |
| MIMIC | 996 | 7,208 | 2,349 |

Each seed fits 258 and tunes on 140 disjoint development patients. Primary
evaluation further requires a prior resolved outcome. Update-anchor counts
are not independent patients or newly acquired labels.

## Failure localization

Surface: the new correction does not reliably improve the frozen forecast.

Direct mechanism supported by ablation: adding more cached state updates and
switching to candidate-error feedback do not rescue transport. Full-stream
MIMIC MAE is worse than scored-only; own-error feedback changes the mean only
slightly. Reset and shuffled-event controls remain statistically competitive.
One seed selects the exact zero checkpoint in each final run, demonstrating
that a source-side non-update choice is possible but insufficient to prevent
all external harm.

Remaining interpretation: source-tuned signed corrections are unstable outside
the fitting population. Small support, recurrent update frequency/measurement
process differences, treatment capture and model optimization are possible
contributors, not individually proven causes. These results reject the claim
that the reproduced software defects alone explain the failure, or that merely
feeding back one's own errors guarantees useful learning.

## Explicit remaining gaps

- This uses all anchors in an existing cache, NOT every raw bedside event.
  The cache itself is a selected measurement-anchor cohort. A full raw-event
  ledger with independently determined update times is not yet implemented.
- Available/storetime equivalence is not verified for every measurement and
  administration. Last-known snapshots are not confirmed active treatments.
- Exact event-level model feedback between cached anchors is not tested.
- No new conformal calibration, time/care-unit promotion gate or unseen
  database evaluation was done. Do not reuse existing intervals for this head.

The next legitimate diagnostic requires a frozen, event-availability-verified
stream and matched update-frequency support, not a relaxed promotion threshold
or another claim that all remaining problems are fixed.

## Artifacts and reproducibility

- `EVENT_STATE_REPAIR_PROTOCOL_20260910.md`
- `artifacts/recurrent_event_state_repaired_full_20260910_final.json`
- `artifacts/recurrent_event_state_repaired_scored_20260910_final.json`
- Historical: `artifacts/recurrent_event_state_glucose_6h_20260909_v1.json`

Final reports bind source/module/input hashes. Rerun
`python -m scripts.audit_recurrent_event_state_loop` with fresh output paths;
add `--scored-only` for the update-stream ablation. Existing outputs are never
overwritten. Intermediate repair reports (un-suffixed and v2) are retained as
debugging provenance only: they preceded the final unclipped float64 age fix
and must not be presented as the current implementation's validation.
