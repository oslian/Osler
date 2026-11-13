# Pure JEPA History/Target-Head Findings

Date: 2026-07-13

## Decision

The default DKA candidate now excludes hand-built belief features. It uses:

1. measurement-pure v5 state inputs;
2. strictly pre-anchor physiologic history features;
3. observed action context and timing;
4. target-specific residual heads with cross-fitted patient splits.

The earlier belief-adapter result is retained only as a historical ablation and
can be reproduced with `--include-beliefs`. It is not part of this pure-JEPA
candidate.

## Full MIMIC-IV result

The pure candidate used 8,050 usable rows from 989 stays. On active-DKA rows:

| Target | Persistence MAE | Pure JEPA adapter MAE | Result |
| --- | ---: | ---: | --- |
| Glucose | 143.374 | 75.598 | improved |
| pH | 0.0688 | 0.0592 | improved |
| HCO3 | 3.570 | 2.148 | improved |
| Anion gap | 5.025 | 2.835 | improved |
| Potassium | 0.595 | 0.455 | improved |
| MAP | 11.523 | 10.093 | improved |
| Sodium | 2.920 | 2.500 | improved |
| Osmolality | 8.403 | 5.685 | improved |
| Creatinine | 0.267 | 0.267 | fallback |
| Urine output | 83.936 | 74.736 | improved |

## Seven-seed patient-held-out gate

Seeds: `7, 11, 19, 23, 37, 53, 71`.

A target passes only when the pure candidate is significantly better than
persistence in every seed and beats the frozen v5 prediction in every seed.

| Target | Pure JEPA mean MAE | Persistence mean MAE | v5 mean MAE | Gate |
| --- | ---: | ---: | ---: | --- |
| Glucose | 66.534 | 103.452 | 119.006 | PASS |
| pH | 0.0555 | 0.0613 | 0.0677 | PASS |
| HCO3 | 2.043 | 2.814 | 3.164 | PASS |
| Anion gap | 2.661 | 3.808 | 4.003 | PASS |
| Potassium | 0.436 | 0.525 | 0.887 | PASS |
| MAP | 10.283 | 11.497 | 11.525 | PASS |
| Sodium | 2.418 | 2.764 | 7.281 | PASS |
| Osmolality | 5.309 | 6.919 | 19.580 | PASS |
| Creatinine | 0.249 | 0.238 | 0.504 | REJECT / persistence |
| Urine output | 70.682 | 76.541 | 105.901 | PASS |

Result: **9/10 targets pass the patient-held-out gate without hand-built
belief features**.

## Interpretation boundary

This supports the narrower statement that the measurement-pure JEPA/history
representation plus target-specific heads can outperform the frozen v5 model
and persistence on most evaluated DKA targets. It does not prove that the raw
v5 JEPA checkpoint alone beats the hand-built belief layer, because this is a
new factual adapter trained on real MIMIC-IV rows.

The transition artifact has no hospital or care-unit identifier, so hospital
held-out validation was not claimed. No causal treatment effect is inferred,
and no checkpoint is promoted from this result.

The hand-built belief modules remain in the repository for other validated
system-specific paths and historical comparison. They are not inputs to the
default pure-JEPA DKA candidate.
