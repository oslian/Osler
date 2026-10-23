# Adjacent Three-Level Cell-First Rescue Findings (2026-08-10)

## Question

Can different bounded methods promote additional target-horizon cells under
the complete factual gate, without changing the required
`system <-> organ <-> body` architecture or relaxing fallback policy?

## Gate

Success required the same cell-level evidence used by the strict full-cohort
audit: seven patient splits; hospital, care-unit, and forward-time holdouts;
patient bootstrap; persistence and validated-teacher comparisons;
module-balanced evaluation; matched coupling ablation; calibrated conformal
coverage; direction/delta scoring; and SVD/collapse checks. Discovery-only
split subsets could reject a method but could never promote it.

## Methods And Results

| Method | Intended cell | Result |
| --- | --- | --- |
| Normalized / patient-cluster conformal variants | `o2sat@3h`, `urine_output@12h` | Rejected; calibration formula alone did not repair the external-domain miss. |
| Stronger hospital/care-unit invariance | `glucose@1h` | **Passed the complete gate.** |
| Change-weighted organ-stage loss | `ast@1h` | Rejected; point gain shrank and delta correlation remained near zero. |
| Lower residual shrinkage | `respiratory_rate@1h` | Rejected; coupling strengthened slightly but expert-anchor evidence regressed. |
| Within-stay recency weighting | `alkaline_phos@3h`, `@6h` | Rejected; forward-time point and module gates still failed. |
| Respiratory system bundle plus domain invariance | `o2sat@3h`, `respiratory_rate@1h` | Rejected; hospital behavior improved but care-unit point/module behavior regressed. |

Failed experimental objectives were removed from the production training CLI.
Their reports remain research evidence; they are not runtime candidates.

## Validated Cell

`glucose@1h` passed all required patient and external gates when the
physiology encoder used domain-invariance weight `0.10` over both
`hospitalid` and `careunit`. Across the seven patient splits, direction
accuracy was approximately `0.617-0.637` and delta correlation approximately
`0.289-0.327`. The selected path is body-mediated:

```text
system -> organ -> body -> organ -> system
```

Direct organ coupling remains disabled for this cell. The result is factual,
research-only, and keeps `causal_claim_allowed: false`.

## Runtime Integration

- Registry: `adjacent_three_level_glucose_domain_invariant_validated_registry_20260810.json`
- Catalog: `whole_body_runtime_candidate_catalog_v12.json`
- Exact input contract: `eicu_canonical_exact_future_full_cohort.v1`
- Canonical unit: `mg/dL`

Catalog-wide unique coverage remains 49 cells across 16 targets because
`glucose@1h` was already validated under a different input contract. The new
cell proves that the strict system-organ-body JEPA can serve glucose under the
full-cohort eICU contract; it does not create duplicate coverage credit.
