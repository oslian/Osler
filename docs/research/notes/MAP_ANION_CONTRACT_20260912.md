# MAP Anion-Gap Contract Audit

All 5,262 saved anion-gap events exactly match native lab IDs, numerical values,
measurement times and conservative available times. Native name is `anion gap`
for all. Accepted units are mmol/L or mEq/L case variants. No transcription or
timestamp discrepancy was found in this saved subset.

Exact measurement-time five-analyte panels were checked without as-of carrying.
Conflicting same-time values are excluded. Compare reported gap with Na-Cl-HCO3
and Na+K-Cl-HCO3, using an absolute difference <=1 as a descriptive rounding
tolerance, not a clinical threshold or automatic recoding rule.

| Seed7 role | Complete panels | Matches without K | Matches with K | Neither |
|---|---:|---:|---:|---:|
| Fit | 1,798 | 1,663 | 118 (6.56%) | 17 |
| Tune | 528 | 500 | 24 | 4 |
| New source | 342 | 331 | 7 | 4 |
| Held-out hospitals | 848 | 619 | 228 (26.89%) | 1 |

This is evidence consistent with mixed calculation conventions within a single
canonical name, and different mixture proportions by role. It is not proof of
each hospital's documented formula: same timestamp is not a specimen ID, the
panel is selected by complete observations, and laboratory methods/rounding may
differ. Counts are panels, not equal-weight patients. Most reported gaps match
the no-K formula in both roles; not a universal four-unit offset.

The full tree uses this variable, but no matched accuracy intervention has yet
quantified whether this convention difference explains its failure. Do not
subtract K from every gap, overwrite raw observations, or fit a correction using
held-out outcomes. A legitimate follow-up needs explicit convention provenance
or a separately named, same-specimen/time derived feature built identically in
source and evaluation with source-only fitting. Original observed gap must stay
distinct from derived gap, and co-measurement support must be audited.

No model, cache or runtime modification; no promotion. Exact-panel/conflict tests
pass. Evidence: `artifacts/map_anion_contract_20260912.json`.
