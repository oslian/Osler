# State-Conditioned Kidney Forecast Audit

## Result

The frozen shared whole-body JEPA was not retrained. Each target uses the same
legal pre-anchor representation. A classifier estimates `P(dynamic)` from shared
body context and belief state; stable predictions stay at persistence, while the
dynamic expert predicts signed delta quantiles. A plain equal-input head and a
capacity-matched shuffled-state model are mandatory controls.

| Target | Cutoff | eICU patient | eICU hospital | MIMIC-IV | Promotion |
|---|---:|---:|---:|---:|---:|
| creatinine@6h | 0.80 | False | False | False | False |
| bun@6h | 0.80 | False | False | False | False |
| urine_output@6h | 0.50 | False | False | False | False |
| potassium@6h | 0.80 | False | False | False | False |
| bicarbonate@6h | 0.70 | False | False | False | False |

No target is promoted. The serving registry and its validated persistence
fallbacks are unchanged.

## MIMIC-IV transfer details

| Target | Dynamic AUC | Candidate MAE | Persistence MAE | MAE delta | Candidate CRPS | Persistence CRPS | 90% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| creatinine | 0.777 | 0.1855 | 0.1856 | -0.0001 | 0.1597 | 0.1684 | 0.964 |
| BUN | 0.746 | 3.0279 | 3.0293 | -0.0014 | 2.4740 | 2.4930 | 0.937 |
| urine output | 0.715 | 103.28 | 111.92 | -8.64 | 82.07 | 90.55 | 0.950 |
| potassium | 0.540 | 0.4271 | 0.4289 | -0.0018 | 0.3028 | 0.3207 | 0.934 |
| bicarbonate | 0.539 | 1.8975 | 1.9118 | -0.0143 | 1.4803 | 1.4143 | 0.962 |

Point estimates improved for all five targets after replacing the first
outcome-conditioned magnitude head with an all-outcome signed-delta head.
Only urine output had a material external point improvement. Creatinine, BUN
and potassium improvements were too small or unstable, while bicarbonate's
point estimate improved but its distributional CRPS worsened.

The probabilistic signal is stronger than the point-routing signal:
creatinine, urine output and potassium had significant MIMIC CRPS gains;
BUN's CRPS confidence interval crossed zero. All MIMIC intervals over-covered,
showing eICU-to-MIMIC distribution shift rather than under-coverage. They are
therefore not eligible runtime intervals even though source-only normalized
conformal scaling corrected eICU coverage.

## Stable versus dynamic diagnosis

- Creatinine, BUN and urine-output dynamic classifiers transferred with AUC
  0.72-0.78. Potassium and bicarbonate were near chance in MIMIC.
- Dynamic patients usually benefited, but false dynamic classifications moved
  stable patients away from a very strong persistence prediction.
- The gated head did not consistently beat the plain equal-input signed-delta
  head on eICU patient and held-out-hospital tests. That control failure is the
  main reason the apparent transfer gains cannot be promoted.
- Persistence is therefore a valid state expert, but the current shared state
  cannot yet choose it with enough specificity across databases.

This is not a flat negative result: the shared state carries reproducible
dynamic-state and distributional information for kidney targets. It is a
negative promotion result because that information is not yet precise enough
to improve point forecasts, controls and calibrated intervals simultaneously.

## Contract repairs completed before the audit

- `derived_interval_rate` is preserved as its own canonical quality class.
- missing `end_time` represented by pandas `NaT` becomes `None`.
- unknown future-label timing policies fail closed; all current 53 release
  variables now have explicit target-specific policies.
- stale continuous target anchors abstain instead of silently persisting.
- continuous readout abstention is exposed in runtime output.

## Boundaries

This is a factual development/transfer audit. MIMIC-IV has been inspected before
and is not described as an untouched lockbox. No external labels selected a model
or cutoff. A target can be promoted only if point MAE beats persistence, the plain
equal-input head and shuffled-state placebo across all patient/domain/time gates;
CRPS beats the empirical persistence distribution; and 90% interval coverage is
within [0.87, 0.93]. Failure leaves the current release and persistence fallback unchanged.
Measurement-time prediction is a separate task and was not modeled here. No causal
or clinical decision claim is allowed.
