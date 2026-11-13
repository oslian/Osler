# Potassium treatment-input coverage

## V2: solution flow is not drug delivery

The follow-up `artifacts/potassium_treatment_support_20260911_v2.json` preserves
all original drug-rate counts but separately retains explicitly documented
solution volume flow. `event_treatment.rate_channels` uses the existing unit
parser; it does not infer concentration, delivery duration, or availability.

| Action | Events with solution mL/h | Queries with solution mL/h in prior6h |
|---|---:|---:|
| Bicarbonate | 307 | 36 |
| Potassium repletion | 65 | 20 |
| Magnesium repletion | 331 | 13 |
| Insulin | 2016 | 464 |
| Diuretics | 454 | 21 |

Thus the previous zero recognized potassium/bicarbonate rate was not failure
to parse a numeric value: it was a drug-dimension whitelist excluding volume.
That exclusion remains correct for delivered drug amount. The new separate
channel repairs discarded descriptive flow information without calling mL/h
mEq/h or mg/h. Solution and drug windows can overlap and must not be summed.
Zero is retained. Unknown, conflicting, or inherited volume/drug units fail
closed. These features are not yet trained, calibrated, or deployed, and no
forecast accuracy benefit is claimed. Independent availability remains absent.

V1 remains a record of drug-compatible support; V2 is the current expanded
coverage report. Existing rate_value and runtime predictions are unchanged.

Aggregate: `artifacts/potassium_treatment_support_20260911.json`.
Runner: `scripts/audit_potassium_treatment_support.py`.

The recent potassium raw-history and encoder contrasts contain lab histories,
not complete treatment histories. They must not be described as testing all
available patient-state inputs or ruling out treatment information.

Read the local infusion table for the exact6174 queries/1309 patients. There
are78568 infusion rows in the cohort. Classification reuses the existing
electrolyte/endocrine contracts plus AKI terms; rate parsing reuses the existing
unit-safe parser. No outcome associations, fitting or external data were used.

| Action | Classified events | Queries with event in prior6h | Queries with recognized rate |
|---|---:|---:|---:|
| Insulin | 9306 | 1081 | 497 |
| Diuretics | 1291 | 78 | 39 |
| Bicarbonate | 307 | 36 | 0 |
| Potassium repletion | 65 | 20 | 0 |
| Magnesium repletion | 334 | 13 | 0 |
| Renal replacement | 0 | 0 | 0 |

Rows are source-ID/action deduplicated. Actions may overlap. Event windows are
strictly(anchor-6h,anchor]; no future event times are included. These are
event-time support counts, not verified bedside availability. Zero rates remain
observations; unknown units remain unknown. No rows or patient IDs are exported.

## Contract limitations found

1. `load_eicu_infusion_rates` uses AKI action classification. That classifier
   intentionally omits insulin, potassium, bicarbonate and magnesium. Reusing
   it as a complete potassium treatment loader would omit these actions; this
   does not establish a defect in its original AKI scope.
2. The newer `event_treatment` snapshot helper is scoped to five glucose-related
   actions and explicitly documents that timestamps without storetime are
   retrospective evidence. It is not a complete potassium availability ledger.
3. Local infusion headers contain infusionoffset but no independent entry or
   store time. Medication has drugorderoffset, drugstartoffset and drugstopoffset;
   order evidence must not be relabeled as administration. Treatment has only
   treatmentoffset for timing. An absence of independent availability timestamps
   prevents a strict as-known claim, not all retrospective research.
4. Zero rate support means this parser/contract could not establish a compatible
   rate. It does not mean no treatment occurred or no dose information exists
   elsewhere. Medication, procedure and fluid-balance coverage were not scanned
   in this bounded infusion audit. Dialysis is not expected to be fully captured
   by infusion labels. Label matching remains imperfect.

## Decision

Do not append these records to availability-safe inputs while pretending the
event timestamp proves knowledge time. Keep evidence-time and known-time claims
separate. A next treatment comparison needs a target-complete action contract,
explicit availability policy, matched own/full controls, and unchanged patients
and labels; retrospective timing sensitivity must be reported separately.

This locates an untested input gap, not the sole root cause or an accuracy fix.
No weights, runtime, registry, intervals or promotion changed. Existing negative
lab-history findings remain valid for their actual narrower input contract.
