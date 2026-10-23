# Renal Shared Patient-State Coupling Findings

## Question

Can the validated renal target-horizon cells share one patient's renal state,
instead of learning twelve unrelated patient adapters, while preserving 90%
coverage and making prediction intervals narrower?

## Architecture tested

```text
same patient's causal renal history + renal belief features
                         |
                  shared kidney state
                         |
       +-----------------+-----------------+
       |                 |                 |
 creatinine heads      BUN heads      urine/MAP heads
       |                 |                 |
 independent validated expert anchor + calibrated shared residual
```

- All twelve cells use the same patient's history rows and one shared kidney
  latent. Multitask training therefore lets creatinine, BUN, urine output and
  MAP constrain the shared renal representation.
- Each target-horizon keeps its own output and scale head. One cell's point
  prediction is not recursively fed into another cell, avoiding cascade error.
- The existing independently validated patient model remains the anchor.
  Calibration patients select a bounded gain for the shared residual; gain zero
  means no coupling.
- A matched independently trained patient-state model is the no-coupling
  ablation. A coupled cell must beat this model, not only the population model.

## Full result

The audit used seven patient-heldout seeds plus hospital-, care-unit- and
late-time-heldout gates. The interval columns are mean 90% **full widths**;
creatinine and BUN use mg/dL, urine output uses the source dataset's mL window,
and MAP uses mmHg.

| Cell | Patient gates P/C/V/N | External gates P/C/V/N | Independent width | Coupled width | Reduction | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| creatinine@3h | 7/7, 7/7, 7/7, 5/7 | 1/3, 2/3, 2/3, 2/3 | 1.400 | 1.286 | 0.114 (8.1%) | interval unstable |
| creatinine@12h | 4/7, 5/7, 7/7, 7/7 | 3/3, 3/3, 3/3, 3/3 | 2.004 | 1.816 | 0.188 (9.4%) | patient point unstable |
| creatinine@24h | 3/7, 7/7, 7/7, 7/7 | 1/3, 1/3, 3/3, 3/3 | 2.175 | 2.010 | 0.165 (7.6%) | point unstable |
| BUN@3h | 4/7, 5/7, 7/7, 7/7 | 2/3, 2/3, 3/3, 3/3 | 18.664 | 16.219 | 2.445 (13.1%) | point unstable |
| **BUN@6h** | **7/7, 7/7, 7/7, 7/7** | **1/3, 1/3, 3/3, 3/3** | **20.636** | **18.799** | **1.837 (8.9%)** | patient pass; external fail |
| BUN@12h | 4/7, 5/7, 7/7, 7/7 | 2/3, 3/3, 3/3, 3/3 | 28.717 | 26.475 | 2.243 (7.8%) | patient point unstable |
| BUN@24h | 4/7, 6/7, 7/7, 7/7 | 1/3, 1/3, 3/3, 3/3 | 34.408 | 32.869 | 1.539 (4.5%) | point unstable |
| BUN@48h | 7/7, 7/7, 7/7, 6/7 | 1/3, 2/3, 3/3, 3/3 | 47.389 | 45.667 | 1.722 (3.6%) | one patient interval miss |
| urine output@6h | 3/7, 5/7, 7/7, 4/7 | 0/3, 1/3, 3/3, 2/3 | 458.691 | 457.107 | 1.584 (0.3%) | unstable |
| urine output@12h | 5/7, 5/7, 7/7, 3/7 | 0/3, 1/3, 3/3, 1/3 | 511.993 | 514.666 | -2.673 (-0.5%) | wider; reject |
| MAP@3h | 5/7, 5/7, 7/7, 3/7 | 0/3, 2/3, 3/3, 1/3 | 41.454 | 41.374 | 0.080 (0.2%) | unstable |
| MAP@6h | 3/7, 5/7, 7/7, 6/7 | 0/3, 1/3, 3/3, 3/3 | 43.270 | 43.112 | 0.158 (0.4%) | unstable |

P/C/V/N means point prediction, coupling ablation, valid 90% coverage, and
significantly narrower interval than the independent patient model.

## Decision

No cell passes every patient and external gate, so no coupled artifact is
promoted. The twelve independently validated renal precision artifacts remain
active.

The strongest result is BUN@6h: it passes all seven patient splits and reduces
the mean full interval by 1.837 mg/dL (8.9%), but it does not generalize across
all external domains. This local success proves that the cells can learn useful
shared patient signal. The remaining failure is domain-dependent coupling gain,
not absence of renal coupling or failed uncertainty calibration.

## Boundary

This is factual prediction only. It does not estimate treatment effects and
keeps `causal_claim_allowed: false`.
