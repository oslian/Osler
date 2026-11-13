# Additional Domains: Selected-Cohort Support Limit

## Why This Was Checked Before Training

The old selected reconstruction cohort is not simply a diagnosis cohort or
all insulin-exposed patients. Its lineage is:

1. `build_insulin_potassium_pairs.py`: stays with compatible insulin rate
   records; potassium availability-anchored current/future pairs.
2. `build_rate_anchored_potassium.py`: restrict to old paired potassium stays;
   require a same-label rate change within 360 minutes, remove conflicting
   offsets and simultaneous multi-label changes; retain inherited patient roles.
3. `audit_map_native_anchor_support.py`: MAP arrivals within that restricted
   stay universe, not a new unrestricted population.
4. `build_native_reconstruction_expanded.py`: sample fresh current targets
   from those patients, with target/sibling-masked histories.

Current reconstruction does not consume future labels, but the ancestral
population selection depended on potassium future-pair availability. This is
a cohort selection/transport issue, not proof that future values enter the
current encoder. Do not identify this retrospective group with diabetes,
renal disease, or an anchor-known physiological subtype.

## Full Additional Pool Inventory

All 73,944 unassigned candidate patients were checked with the unchanged
unit-safe insulin rate parser. No model predictions or errors selected anyone.

| Support | People | Events |
|---|---:|---:|
| Raw labels containing insulin |5,394|127,992|
| `Insulin (ml/hr)` |5,379|127,565|
| `Insulin` without rate unit |282|394|
| `Insulin (units/hr)` |15|32|
| `Insulin (mcg/min)` |1|1|

Label groups overlap in people. Raw documentation spans 14 connected hospital
components; the units/hr group spans five. Only **five people** have a compatible
same-label rate change after the historical conflict/simultaneous-label rules.
These are upper-bound whole-stay retrospective counts; requiring the full
historical potassium/MAP/role criteria can only reduce them.

For creatinine/BUN/glucose/albumin/total protein, no component has 30 supported
patients under either accepted-rate or accepted-rate-change grouping. Thus
the pooled 47-variable support does NOT repair the old selected cohort's
multi-domain calibration support. The candidate hospitals have many labs and
vitals but a different usable infusion-record contract.

## What This Does And Does Not Explain

This identifies a concrete failure of the proposed data expansion assumption:
more disconnected hospitals do not necessarily supply more examples of the
same previously selected population. Documentation/units affect eligibility.
It does not establish that untreated physiology differs or that this is the
single root cause of the model's prior error. Absence of a qualified rate is
not absence of therapy. Flow in ml/hr cannot be relabeled units/hr without
verified drug concentration and dimensional semantics.

The retrospective flags are stored privately with provenance and hashes;
`inference_feature_allowed: false`. Rate record availability is not verified,
and records may occur after a proposed reconstruction anchor. They must NOT
be used as inference features, routing decisions or diabetes labels.

## Decision And Next Work

Do not train a supposed enlarged selected-cohort model on this pool or change
unit rules to make its support count pass. Preserve existing held-out domain
exclusions. The new pool can still support a broad current-reconstruction
experiment after target-masked-history acceptance and prospective domain roles.
Any new clinical subgroup must be defined using verified pre-anchor evidence,
explicitly named as a different cohort, and evaluated independently. Matching
the old selected cohort remains unfulfilled; do not silently redefine it.

The originally planned complete masked-history acceptance audit has NOT yet
run. This cohort preflight changes the next training design before expensive
sample building, rather than claiming additional observations already solve
generalization.

Four new rate/support tests pass: six-hour boundary, equal re-documentation,
conflicting values, simultaneous multi-label changes, and unique-person counts.
Both full inventory scripts completed. No weights, runtime, interval, gate,
promotion or causal claim changed.

Reports:
- `artifacts/native_reconstruction_additional_insulin_support_20260921.json`
- `artifacts/native_reconstruction_additional_insulin_units_20260921.json`
