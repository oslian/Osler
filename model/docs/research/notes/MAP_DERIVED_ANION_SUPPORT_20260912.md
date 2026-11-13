# Explicit Derived Gap Support

Created `derived_anion_gap_without_k_same_time`, separate from reported gap.
Formula Na-Cl-HCO3 requires all three exact-time, accepted-unit, unambiguous
components to have arrived. Atomic same-availability updates preserve chronology.
Incomplete or conflicting versions emit masked invalid evidence that blocks
carrying an older derived value; later valid revisions may restore it. Valid
zero is retained. Component source IDs are private provenance, not model inputs.

Built 48,372 event versions, 27,410 valid. Seed7 eligible MAP query support,
with maximum six-hour measurement age:

| Role | Observed queries | Supported patients | Patient-equal observed fraction |
|---|---:|---:|---:|
| Fit | 37,386 | 830 | 40.02% |
| Tune | 12,602 | 304 | 40.12% |
| New source | 12,442 | 276 | 37.24% |
| Held-out hospitals | 4,123 | 147 | 38.66% |

This prepares an explicitly defined input; it is NOT training or accuracy
evidence. Similar support proportions do not establish equal distributions or
physiological equivalence. All components come from the existing qualified lab
cache, not a newly reconstructed complete invalid-result stream. Same timestamp
is not a specimen ID. No inferred laboratory convention is assigned to reported
gap and no reported observation is overwritten.

Two regression tests cover staggered availability, conflicts, restored versions,
valid zero, incomplete new panels, unit rejection and mismatched draw times.
No runtime, calibration or promotion change. A matched forecast trial remains
necessary. Evidence: `artifacts/map_derived_anion_support_20260912.json`.
