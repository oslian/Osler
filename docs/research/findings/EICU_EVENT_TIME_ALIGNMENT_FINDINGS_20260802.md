# eICU Event-Time Alignment Findings

## Scope

This audit checked whether canonical eICU measurements and observed treatment
events actually occupied the same synthetic timeline before evaluating
treatment-aware factual forecasting. It remains research-only and does not
support causal, treatment-effect, clinical, or decision claims.

## Root Cause

The canonical measurement adapter converted eICU minute offsets relative to
`1970-01-01`. Medication, infusion, and procedure adapters used
`2100-01-01`. Their relative offsets were individually correct, but their
absolute synthetic timestamps differed by 130 years.

The event ledger therefore contained treatment events, while the as-of example
builder found none inside any measurement history window. Previous canonical
eICU examples contained zero `hist_*` treatment columns. The
`include_treatment_context` path was syntactically enabled but semantically
empty.

## Repair

All eICU offset-based events now use one shared synthetic origin,
`2100-01-01`. Extraction also fails closed when median timestamps from event
kinds differ by more than one year. This prevents another apparently valid
ledger from silently combining incompatible time domains.

Urine output remains interval-normalized and explicitly represented in
`mL/hour`.

## Rebuilt Data

The aligned ledger preserves the original event counts:

| Event kind | Count |
|---|---:|
| Measurement | 5,342,256 |
| Medication | 25,355 |
| Infusion | 20,606 |
| Procedure | 13,951 |
| Total | 5,402,168 |

It covers 2,520 encounters and 1,841 patients. The rebuilt examples contain
309,258 rows from 2,388 encounters and 1,802 patients, now with 96 `hist_*`
treatment-context columns instead of zero.

Representative non-zero support is:

| Pre-anchor context | Rows |
|---|---:|
| Fluids | 177,420 |
| Insulin | 94,656 |
| Potassium repletion | 68,922 |
| Vasopressor | 44,586 |
| Transfusion evidence | 27,348 |
| Bicarbonate | 11,724 |

The independent evaluation artifact contains 144,672 rows, 1,079 stays, and
957 patients, with zero patient overlap with the 845 development patients.

## Seven-Seed Result

The corrected treatment-aware core-12 sparse-distributional JEPA was trained
on 1,000 stays over seven patient-disjoint seeds. Exactly one target-horizon
cell passed every internal gate:

```text
urine_output@1h
```

It beat the validated teacher, persistence, and capacity-matched no-expert
ablation in 7/7 splits; patient-equal conformal calibration and SVD stability
also passed 7/7. Median MAE delta versus the teacher was
`-0.4443 mL/hour`. Direction accuracy ranged from 0.645 to 0.701 and delta
correlation from 0.264 to 0.342.

The same fixed cell then passed held-out hospital, care-unit, and forward-time
accuracy, no-expert, module-balanced, patient-bootstrap, and conformal gates.

Other targets did not pass the complete contract. Treatment context therefore
adds real information but is not a universal forecast upgrade. In particular,
hemoglobin remained limited by uncertainty calibration, and the former
measurement-only `urine_output@3h` cell did not remain fully validated inside
this different treatment-aware candidate.

## Runtime Decision

The new factual research source is identified by:

```text
candidate_id:
  teacher_anchored_aligned_treatment_sparse_distributional_joint_jepa
input_contract_id:
  joint_asof_whole_body_state_aligned_treatment_interval_urine.v1
validated cell:
  urine_output@1h
unit:
  mL/hour
```

Runtime requires an exact contract and unit match. Legacy measurement-only,
raw-volume, or misaligned-treatment input falls back to persistence with an
explicit rejection reason.

The earlier measurement-only `urine_output@3h` registry remains valid for its
own input contract. It is separate evidence, not an interchangeable model or
permission to use the treatment-aware candidate at 3h.

## Meaning

The main finding is not that treatment context fixes all remaining targets.
It is that a 130-year synthetic time-origin mismatch had silently erased all
treatment history from canonical eICU examples. Correcting that defect exposed
a reproducible short-horizon urine signal and shifted the best validated urine
horizon from 3h in the measurement-only model to 1h in the treatment-aware
model.

This is consistent with short-term fluid and diuretic context, but it remains
factual association only. `causal_claim_allowed` remains `false`.

## Artifacts

- `osler_jepa/event_adapters.py`
- `aligned_treatment_interval_urine_validated_registry_20260802.json`
- `whole_body_runtime_candidate_catalog_v2.json`
- aggregate event/example/gate reports under the external artifacts directory

Verification on 2026-08-02: the complete `medical_jepa` unit suite passed
350/350 tests. A runtime smoke test allowed the validated 1h cell only under
the aligned-treatment contract and rejected the legacy contract with
`input_contract_mismatch`.
