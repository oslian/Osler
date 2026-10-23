# Adjacent Three-Level Full-Cohort Validation (2026-08-10)

## Question

Can the strict anatomical hierarchy

```text
functional system <-> anatomical organ <-> whole body
```

turn the earlier 1,000-stay patient-level signals into externally stable,
target-gated factual forecast cells?

## Method

- eICU canonical event examples: 561,024 rows, 3,287 patients, 12 hospitals;
- 53 targets and 14 anatomical organ/structure nodes;
- 1h, 3h, 6h, 12h, 24h, and 48h horizons;
- seven patient-heldout seeds;
- system-first, organ-second, body-third training;
- hospital and care-unit provenance removed adversarially from the physiology
  latent;
- patient-grouped cross-validated adaptive conformal calibration;
- hospital-, care-unit-, and forward-time-heldout external gates;
- matched system-only path ablation, module balance, patient bootstrap,
  direction, delta-correlation, and latent SVD gates.

The run required two computational fixes that do not change the model or
evaluation contract: latent caches are stored as CPU float16 and restored to
float32 per batch, while the collapse gate uses deterministic patient-equalized
randomized SVD on at most 4,096 rows per split.

## Results

Compared with the earlier 1,000-stay audit:

| Stage | Earlier | Full cohort |
|---|---:|---:|
| Evaluable target-horizon cells | 193 | 208 |
| 7/7 teacher + persistence cells | not separately reported | 66 |
| 7/7 point + organ/body path cells | 38 | 44 |
| Full seven-seed patient candidates | 5 | 10 |
| External factual-gate cells | 0 | 4 |
| Final scoring + runtime-registry cells | 0 | 3 |

The four cells that passed every external split were:

```text
ast@1h
urine_output@1h
urine_output@3h
urine_output@6h
```

`ast@1h` was rejected by the final scoring gate. Its direction accuracy was
above 0.60, but delta correlation remained below 0.05 in every patient split.
It therefore remains fallback-only.

The three validated target-gated research cells are:

```text
urine_output@1h
urine_output@3h
urine_output@6h
```

Their direction accuracy was approximately 0.67-0.73 and their delta
correlation approximately 0.64-0.73 across all seven patient splits. All seven
latent SVD checks passed, with effective ranks above 8 and top-subspace
overlap above 0.94.

Every validated cell selected the body-mediated path. Direct organ coupling
is disabled for these cells because it did not independently earn runtime
authorization.

## Runtime Boundary

The materialized registry is
`adjacent_three_level_full_cohort_validated_registry_20260810.json`.
It requires the exact input contract
`eicu_canonical_exact_future_full_cohort.v1` and urine output in `mL/hour`.

The system remains:

- factual research-only;
- target and horizon gated;
- persistence-backed for every absent cell;
- non-causal and non-clinical.

## Conclusion

Increasing cohort size and using patient-grouped adaptive calibration produced
a real but bounded gain. The strict three-level whole-body model moved from
zero externally authorized cells to three. It did not turn all 38 earlier
signals into runtime cells. Most remaining cells fail because their path gain,
cross-domain point performance, or uncertainty calibration is not stable in
every required split.

