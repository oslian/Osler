# JEPA Belief Adapter Candidate Findings

Date: 2026-07-13

## Candidate

The candidate keeps `dka_symbolic_jepa_v5.pt` frozen and adds a factual
real-data adapter with four inputs:

1. measurement-pure v5 rollout features;
2. strictly pre-anchor physiologic history: previous level, age, delta, slope,
   and observation count;
3. existing online-compatible belief features;
4. observed action dose, exposure, recency, and timing.

Each target receives its own residual ridge head. Outer splits are grouped by
stay, and model fitting is limited to the discovery stays. No checkpoint was
changed or promoted.

## Full MIMIC-IV candidate result

The initial cross-fitted candidate used 8,050 usable rows from 989 stays. The
active-DKA MAE comparison was:

| Target | Persistence | Measurement-pure v5 | Safe target adapter |
| --- | ---: | ---: | ---: |
| Glucose | 143.374 | 153.898 | 75.275 |
| pH | 0.0688 | 0.0709 | 0.0592 |
| Bicarbonate | 3.570 | 3.361 | 2.163 |
| Anion gap | 5.024 | 4.120 | 2.770 |
| Potassium | 0.595 | 0.855 | 0.456 |
| MAP | 11.523 | 11.666 | 10.090 |
| Sodium | 2.920 | 6.975 | 2.497 |
| Osmolality | 8.403 | 20.539 | 5.730 |
| Creatinine | 0.267 | 0.647 | 0.267 |
| Urine output | 83.936 | 126.620 | 74.155 |

## Seven-seed patient-held-out audit

The fixed residual adapter was then evaluated over seeds
`7, 11, 19, 23, 37, 53, 71`. A target passes this candidate gate only when it
beats persistence with a patient-level bootstrap upper confidence bound below
zero in all seven seeds and also beats the original v5 in all seven seeds.

| Target | Candidate mean MAE | Persistence mean MAE | v5 mean MAE | Gate |
| --- | ---: | ---: | ---: | --- |
| Glucose | 67.786 | 103.452 | 119.006 | PASS |
| pH | 0.0561 | 0.0613 | 0.0677 | PASS |
| Bicarbonate | 2.047 | 2.814 | 3.164 | PASS |
| Anion gap | 2.639 | 3.808 | 4.003 | PASS |
| Potassium | 0.433 | 0.525 | 0.887 | PASS |
| MAP | 10.123 | 11.497 | 11.525 | PASS |
| Sodium | 2.448 | 2.764 | 7.281 | PASS |
| Osmolality | 5.410 | 6.919 | 19.580 | PASS |
| Creatinine | 0.260 | 0.238 | 0.504 | REJECT / persistence |
| Urine output | 69.582 | 76.541 | 105.901 | PASS |

The transition artifact contains no hospital or care-unit identifier, so a
hospital-held-out test was not claimed or run.

## Interpretation

This is strong evidence that the original JEPA failure was partly an input and
representation problem. The hand-built belief layer was not winning because
the data had no signal. It was exposing patient-specific trajectory signal in
a target-specific, noise-controlled form that v5 did not receive.

The result does not prove causal treatment effects and does not make the v5
checkpoint itself better. The improvement is in a cross-fitted observational
adapter. The next engineering step is to transfer this representation into a
new action-conditioned JEPA training contract, then require the same target
gate plus an independent hospital or database holdout before promotion.
