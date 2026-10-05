# Original-Event Support Ceiling

Report: `artifacts/native_reconstruction_support_ceiling_20260913.json`.
This input-only audit uses all patients in the existing seed-7 MAP-derived
cohort: 923 fit, 335 tune, 311 evaluation and 194 hospital-heldout. Role sets
are disjoint and encounter-to-patient mapping is checked. No outcome-based
selection or model training occurs.

For each canonical variable, count unique patients having finite, source-qualified
measurement events with availability minus measurement time between zero and
one hour. This is an UPPER BOUND, not completed reconstruction eligibility:
source-version conflicts and independent target/sibling-masked history still
need to be checked by the actual reconstruction builder.

## Result

36 of 53 variable slots have at least 30 potential patients in every role.
Examples (fit / tune / evaluation / hospital):

| Variable | Fit | Tune | Evaluation | Hospital |
|---|---:|---:|---:|---:|
| Creatinine | 916 | 334 | 309 | 194 |
| BUN | 915 | 334 | 309 | 194 |
| Glucose | 923 | 335 | 311 | 194 |
| Hemoglobin | 919 | 333 | 310 | 193 |
| Albumin | 660 | 242 | 237 | 172 |
| Troponin I | 417 | 160 | 141 | 42 |

Seventeen slots fall below 30 in at least one role: CK-MB, cortisol, CRP,
hsCRP, ESR, ferritin, fibrinogen, free T4, ionized calcium, LDH, myoglobin,
pH, prealbumin, serum osmolality, TSH, uric acid and urine output.
Fibrinogen has 354 fit patients but only 26 in the heldout hospitals, for
example. Source sample size alone does not establish evaluability.

Zero support here means absent from this qualified source/cohort/time contract,
not absent from every local database or impossible to measure/predict.
These are unique variables, not target-horizon forecast cells. The 36 counts
are not validated capabilities and not an external-data generalization result.

## Decision

The 16-person pilot limit is not the available cohort limit for dense targets.
Build larger samples through the exact as-of conflict-aware reconstruction
path, then perform a per-target support preflight before training. Keep sparse
slots explicitly unevaluable where support is insufficient. Do not lower the
30-patient reporting rule or infer adequate power merely from crossing it.

No release weights, registry, intervals or serving changed; no promotion.
