# Persistence Routing Findings

## Question

Can the frozen whole-body state decide, for an individual kidney-related
forecast, when a target expert should move away from persistence?

The point expert and all pre-anchor features were held fixed. eICU was used for
development and MIMIC-IV for previously inspected external replication. The
validated runtime was not changed.

## Methods Tested

1. **Dynamic-state classifier**: predict whether the future absolute change
   exceeds a target-specific clinical magnitude.
2. **Realized-utility classifier**: on disjoint patients, predict whether the
   expert's absolute error is lower than persistence.
3. **Distribution-risk router**: predict q05/q50/q95 of future delta and compare
   the expert's expected absolute loss with zero-delta persistence. Both raw and
   dimensionless relative risk margins were tested.

Every candidate was compared with persistence, an equal-input plain target
head, and a capacity-matched shuffled routing control. It also had to pass
patient, care-unit, late-stay, CRPS and 90% coverage gates.

## Result

No individual-level hard router passed the complete gate for creatinine, BUN,
urine output, potassium or bicarbonate at 6 hours.

The realized-utility classifier could not identify who benefits from the
expert in MIMIC-IV: benefit AUC was approximately 0.48-0.54. Its conservative
bound therefore sent almost every case to persistence.

The distribution-risk router improved CRPS significantly for all five targets.
With the raw risk margin it also produced a lower pooled MIMIC-IV point MAE than
persistence for all five targets, but most point gains were too small or
unstable to pass the patient/domain/time comparisons. Relative risk scaling did
not repair transfer and made creatinine and BUN point MAE worse than
persistence.

| Target at 6h | Raw-risk point MAE | Persistence MAE | Move fraction | CRPS better | 90% interval transfer |
|---|---:|---:|---:|---|---|
| Creatinine | 0.1849 | 0.1856 | 0.0030 | Yes | Overcoverage |
| BUN | 3.0275 | 3.0293 | 0.0007 | Yes | Overcoverage |
| Urine output | 102.58 | 111.92 | 0.7737 | Yes | Overcoverage |
| Potassium | 0.4250 | 0.4289 | 0.0053 | Yes | Care-unit undercoverage |
| Bicarbonate | 1.9075 | 1.9118 | 0.0040 | Yes | Overcoverage |

The plain equal-input head remains informative: it is materially better than
persistence for urine output and potassium in MIMIC-IV and for several targets
in eICU. The unresolved part is not simply forecasting capacity. It is the
sample-level decision of when that expert will beat persistence across a new
database.

## Diagnosis

The hard routing label is dominated by future innovations that are not fully
identified by the current anchor state: acute treatment changes, new events and
measurement-process differences. Absolute risk scores also change scale across
databases. A flexible classifier cannot recover information that is absent at
the anchor, while conservative selection collapses into near-universal
abstention.

The conditional distributions contain useful information, demonstrated by the
five CRPS improvements, but their eICU-calibrated widths do not transfer to
MIMIC-IV. This is a separate calibration/domain-shift problem.

## Decision

- Keep validated target-level expert selection and persistence fallback.
- For patient-level uncertainty, retain the distributional candidate and report
  calibrated uncertainty or abstention instead of claiming a reliable hard
  winner classifier.
- Do not promote either new router or mutate the release.
- A future hard router requires additional pre-anchor signals that precede the
  innovation being forecast, especially complete dose/rate treatment context,
  treatment intent, bedside assessment and appropriately aligned high-frequency
  events.
- Recalibration on a target database requires a genuinely separate database for
  final evaluation. MIMIC-IV and NWICU have already been inspected and cannot be
  relabeled as untouched lockboxes.

All results are factual observation/forecast evidence. They do not identify
treatment effects and do not support clinical decisions.
