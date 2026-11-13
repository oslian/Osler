# Three Additional Runtime-Gated JEPA Cells

Date: 2026-08-03

## Result

Compared with `whole_body_runtime_candidate_catalog_v2.json`, the validated
system-level coverage now contains exactly three additional target-horizon
cells:

```text
glucose@1h
respiratory_rate@3h
respiratory_rate@24h
```

The union grows from 30 to 33 cells without removing an existing cell.
`glucose@3h` and `glucose@12h` were already available through the shared
router; newly materialized JEPA sources for those cells are alternate sources,
not additional system coverage.

## Glucose at 1h

`glucose@1h` uses the administration-aligned eICU contract
`joint_asof_whole_body_state_aligned_treatment_interval_urine.v1`. The final
hierarchical candidate passed all seven patient splits and the held-out
hospital, care-unit, forward-time, module-balanced, bootstrap, and conformal
requirements. Treatment inputs are strictly pre-anchor observations.

## Respiratory rate at 3h and 24h

Both respiratory cells use the same administration-aligned eICU contract and
require `breaths/minute` at runtime.

| Cell | Teacher | Persistence | No-expert | Patient conformal |
|---|---:|---:|---:|---:|
| `respiratory_rate@3h` | 7/7 | 7/7 | 7/7 | 7/7 |
| `respiratory_rate@24h` | 7/7 | 7/7 | 7/7 | 7/7 |

Each cell also passed held-out hospital, care-unit, forward-time,
module-balanced, paired-bootstrap, no-expert, and conformal requirements.
`respiratory_rate@24h` originally missed only care-unit conformal coverage.
A patient-grouped crossfit adaptive conformal estimator corrected that domain
calibration without using forecast test labels or relaxing the required
0.87-0.93 coverage band.

## Rejected alternatives

The following remain candidate-only:

- `creatinine@1h` and `creatinine@3h`: point gates passed, but care-unit
  patient-equalized conformal coverage remained just above 0.93;
- `hematocrit@3h`: reversed against persistence in the forward-time split;
- `calcium@6h`: the target-only confirmation fell to 6/7 persistence and 5/7
  conformal splits;
- `platelets@12h`: prior forward-time evidence reversed, so it was not used to
  fill the third cell.

No threshold was relaxed and no failed target was counted.

## Runtime boundary

The materialized registries are:

- `aligned_treatment_glucose_hierarchical_validated_registry_20260803.json`;
- `respiratory3_validated_registry_20260803.json`;
- `respiratory24_validated_registry_20260803.json`.

They are included in `whole_body_runtime_candidate_catalog_v4.json`. Missing
or mismatched contracts, units, checkpoint status, targets, or horizons always
fall back to persistence. These remain factual research forecasts only:

```text
causal_claim_allowed: false
clinical_promotion_allowed: false
```

Verification: the complete `medical_jepa` unit suite passed 357/357 tests.
