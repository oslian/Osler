# BUN@3h Website External Evaluation

## Question

How accurate is the exact `patient_state_precision_bun3_v1` artifact used by
the BUN website when it is applied to local patient data that did not train the
artifact?

## Data And Boundary

- Training database: eICU.
- External test database: MIMIC-IV v3.1 whole ICU.
- Test population: 16,643 patients, 19,482 ICU stays, 45,465 eligible BUN
  windows.
- Each prediction used six causal rows from the same stay.
- The nominal horizon was 3 hours. The future BUN measurement tolerance was
  plus or minus 1.5 hours.
- Anchor-after measurements and future treatments were excluded from inputs.
- The report contains no patient identifiers or row-level predictions.
- This remains factual research-only forecasting. It is not a clinical or
  causal result.

## Main Result

The treatment-context version is the closest match to the website's complete
optional input contract.

| Metric | Result |
|---|---:|
| Row MAE | 2.649 mg/dL |
| Patient-equalized MAE | 2.491 mg/dL |
| RMSE | 4.320 mg/dL |
| Median absolute error | 1.705 mg/dL |
| Bias | -0.352 mg/dL |
| Within plus/minus 2 mg/dL | 56.3% |
| Within plus/minus 5 mg/dL | 87.2% |
| Within plus/minus 10 mg/dL | 96.8% |
| Direction accuracy when absolute change >= 1 mg/dL | 65.6% |
| Delta correlation | 0.388 |
| Empirical coverage of nominal 90% interval | 93.7% |
| Mean full interval width | 13.124 mg/dL |
| Median full interval width | 10.897 mg/dL |

The negative bias means the model under-predicted future BUN by about 0.35
mg/dL on average.

## Baseline Comparison

Patient-equalized MAE:

| Source | MAE | Difference from website model |
|---|---:|---:|
| Website personalized artifact | 2.491 mg/dL | reference |
| Population ridge | 2.975 mg/dL | website better by 0.484 mg/dL |
| Persistence | 2.466 mg/dL | website worse by 0.026 mg/dL |

The website model significantly beat its population model. The paired
patient-bootstrap 95% CI for personalized-minus-population MAE was
`[-0.505, -0.462]` mg/dL.

It did **not** beat persistence on this external database. The paired 95% CI
for personalized-minus-persistence MAE was `[+0.001, +0.049]` mg/dL. The
difference is small but points in the wrong direction, so external promotion
is not allowed.

The empirical interval coverage was 93.7%, above the frozen 0.87-0.93 gate.
The interval is therefore slightly too wide after cross-database transfer and
is not externally precision-promoted for MIMIC-IV.

## Treatment Context Ablation

| Metric | Missing treatment context | Observed treatment context |
|---|---:|---:|
| Row MAE | 2.843 | 2.649 |
| Patient-equalized MAE | 2.658 | 2.491 |
| RMSE | 4.580 | 4.320 |
| Bias | -0.851 | -0.352 |
| Within plus/minus 5 mg/dL | 84.8% | 87.2% |
| Direction accuracy | 59.9% | 65.6% |
| Delta correlation | 0.279 | 0.388 |
| 90% interval coverage | 92.3% | 93.7% |

Observed pre-anchor treatment context materially improved point accuracy and
directional signal. It did not fully close the persistence gap.

## Internal Validation Context

The canonical eICU seven-seed patient-heldout audit had a median BUN@3h MAE of
3.825 mg/dL for the patient-state candidate, versus 4.215 mg/dL for the
population anchor. All seven patient gates and all three hospital/care-unit/
late-stage gates passed. This is valid internal-domain evidence, while the
MIMIC result above is the stronger cross-database check.

## Literal Website Endpoint Smoke Case

One de-identified MIMIC trajectory was reconstructed as a 23-row CSV and sent
through the actual `/api/predict-csv` endpoint. The measured future BUN was
15.0 mg/dL; the website predicted 10.349 mg/dL, for an absolute error of 4.651
mg/dL. Its interval was 5.592-15.106 mg/dL and contained the measured value.

This single case verifies the upload-to-result path only. It is not used as an
accuracy estimate; the 45,465-window aggregate evaluation above is the
accuracy result.

## Interpretation

There is no single percentage called accuracy for a continuous lab value.
The most useful summary is:

> On independent MIMIC-IV patients, 87.2% of website predictions were within
> 5 mg/dL of the measured future BUN, with MAE 2.65 mg/dL. However, the model
> remained 0.026 mg/dL worse than persistence after patient equalization, and
> its nominal 90% interval was slightly over-wide.

The result supports keeping the website research-only and retaining
persistence as the external fail-closed baseline. The next accuracy work
should target cross-database recalibration and domain-invariant feature
normalization, not a looser promotion threshold.

## Reproduction

The aggregate evaluator is
`bun_forecast_demo/evaluate_local_patients.py`. It loads the same serialized
artifact as the website and emits aggregate-only JSON.

```bash
python -m bun_forecast_demo.evaluate_local_patients \
  --cohort /path/to/mimiciv_observation_transitions_3h.parquet \
  --output /tmp/bun_forecast_demo_mimiciv_external_evaluation.json \
  --target-tolerance-hours 1.5
```
