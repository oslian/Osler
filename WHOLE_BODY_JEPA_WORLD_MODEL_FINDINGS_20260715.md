# True JEPA World Model Smoke Findings

Date: 2026-07-15

## What changed

The whole-body candidate now has a real latent-prediction path:

```text
anchor history/current -> context encoder -> context latent
context latent + horizon -> latent predictor -> predicted future latent
future state -> EMA target encoder -> target future latent
JEPA loss -> predicted latent versus stop-gradient target latent
```

The existing numerical heads remain factual decoders for evaluation. They are
not the JEPA objective. The model is still observation-only and has no causal
or treatment-effect claim.

## Smoke result

The same strict gate was run on 9,123 aligned rows, five modules, seven
patient-heldout seeds, and one hospital-heldout split.

| Target | Patient value pass | Patient cross-system pass | Median delta vs persistence |
|---|---:|---:|---:|
| Glucose@6h | 2/7 | 4/7 | +0.290 |
| Potassium@6h | 3/7 | 5/7 | +0.002 |
| Bicarbonate@12h | 4/7 | 6/7 | -0.006 |
| MAP@1h | 2/7 | 7/7 | +0.303 |

Overall status: `candidate_only`; validated target-horizon cells: `0`.

## Interpretation

The implementation is now genuinely JEPA-like rather than merely a direct
forecast model with a Transformer backbone. The smoke also shows that a valid
JEPA latent objective does not automatically improve factual numerical
forecasting. MAP cross-system representation is promising, but value accuracy
for glucose and potassium did not improve enough for promotion.

## Decision

- Keep the true JEPA world model as a research candidate.
- Keep the existing target-gated factual decoder and persistence fallback.
- Do not replace the current validated observation router yet.
- Future promotion still requires patient value, hospital value, both
  conformal gates, and cross-system ablation for each target-horizon cell.
