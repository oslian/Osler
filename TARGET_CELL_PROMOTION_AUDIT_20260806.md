# Target Cell Promotion Audit (2026-08-06)

## Decision

Two additional target-horizon cells passed the full factual promotion gate:
`chloride@12h` and `hematocrit@12h`. The active runtime catalog now contains
48 unique cells across 16 targets.

The gate was unchanged:

- 7/7 patient-held-out splits;
- 3/3 external splits (hospital, care unit, and forward time);
- patient-level paired bootstrap superiority over persistence and the matched
  simpler comparator;
- 90% conformal coverage between 0.87 and 0.93;
- factual research only, with no causal or clinical promotion.

## Candidates Tested

| Cell | Patient passes | External passes | Decision |
| --- | ---: | ---: | --- |
| `chloride@12h` | 7/7 | 3/3 | Promote |
| `hematocrit@12h` | 7/7 | 3/3 | Promote |
| `wbc@6h` | 4/7 | 3/3 | Reject |
| `wbc@12h` | 5/7 | 3/3 | Reject |
| `wbc@24h` | 5/7 | 3/3 | Reject |
| `anion_gap@3h` | 6/7 | 3/3 | Reject |
| `creatinine@48h` hurdle source | 4/7 | 3/3 | Reject source; cell already validated by `shared_module_nested_stack` |

All cells had previously shown point-prediction value. A plain patient-cluster
calibration did not resolve the final coverage instability. A fixed hybrid
policy then separated two kinds of uncertainty without changing the point
predictor or relaxing the gate:

- patient splits use an observable regime-Mondrian interval over measurement
  age, predicted stability, and pre-anchor treatment context;
- hospital, care-unit, and forward-time splits retain the globally normalized
  interval that already generalized across domains.

This policy promoted chloride and hematocrit. It did not rescue WBC or anion
gap, so those cells remain fail-closed.

## Promotion Interpretation

The successful result identifies two distinct generalization problems. Patient
heterogeneity benefits from physiological regime conditioning, while external
domain shifts require a broader global calibration. Applying either method to
every split was inferior to assigning each method to the split type it was
designed to handle.

## Interpretation

The remaining failures are not caused by one missing global calibration trick.
WBC and anion-gap residuals remain unstable even after observable regime
conditioning. Further attempts should change target-specific future
supervision or physiological inputs rather than tune the same intervals again.

The production decision is fail-closed: cells absent from the validated
catalog remain on persistence or missing output.

## Artifacts

- `artifacts/reports/wbc_patient_cluster_hurdle_6_24h_20260806.json`
- `artifacts/reports/anion_gap_patient_cluster_hurdle_3h_20260806.json`
- `artifacts/reports/hematocrit_patient_cluster_hurdle_12h_20260806.json`
- `artifacts/reports/creatinine_patient_cluster_hurdle_48h_20260806.json`
- `artifacts/reports/chloride_patient_cluster_hurdle_12h_20260806.json`
- `artifacts/reports/chloride_hybrid_regime_hurdle_12h_20260806.json`
- `artifacts/reports/hematocrit_hybrid_regime_hurdle_12h_20260806.json`
- `artifacts/reports/wbc_hybrid_regime_hurdle_6_24h_20260806.json`
- `artifacts/reports/anion_gap_hybrid_regime_hurdle_3h_20260806.json`
