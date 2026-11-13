# Waveform Promotion Methods Findings (2026-08-27)

## Question

Can additional modeling methods promote the 15-minute incident-hypoxemia
Waveform JEPA beyond the table baseline and capacity-matched waveform placebo?

## Fixed boundary

- factual event-risk prediction only;
- patient-disjoint 7-seed evaluation;
- care-unit and forward-time holdouts where applicable;
- candidate must beat table physiology, the relevant summary/masked comparator,
  and a patient-shuffled equal-capacity waveform placebo;
- probability calibration must remain acceptable;
- VitalDB was not reused for method selection after the first external audit.

No candidate was promoted.

## Methods tested

| Method | Complete patient-split gate | Main result |
|---|---:|---|
| patient-weighted temperature / Platt / beta / isotonic calibration | 0/7 for every method | probability remapping did not stabilize the raw-waveform increment |
| future-SpO2 multi-task residual JEPA, weights 0.1 / 0.3 / 1.0 | 0/7 for every weight | direct future physiology supervision did not improve event generalization |
| 10/60/300/900-second morphology-aware raw tensor | 0/7 | one split passed all comparisons; care-unit and forward-time gates failed |
| training-only PCA/SVD waveform modes + regularized logistic head | 0/7 | point estimates were often close to table, but the effect was too small and unstable |
| corrected incident-event labels | 0/7 | label repair was necessary but did not independently create a promotable model |
| 5/10/15/30-minute ordered event heads | 0/7 | additional time-to-event supervision improved some placebo comparisons only |
| three-member deep ensemble with equal-capacity placebo ensemble | 0/7 | averaging initialization variance did not produce a stable waveform increment |

## Important data defect found

The original 15-minute training file contained 241 positive events. Rebuilding the
same 1,578 anchors with the corrected event implementation produced 300 events.
The 59 changed rows were false negatives caused by the old adjacent-sample event
logic. Anchor identity and ordering were identical, so this was a label defect,
not a row-alignment defect.

All subsequent label-sensitive experiments used the corrected definition:
two low SpO2 observations separated by 30 to 300 seconds within the event horizon.

## Interpretation

The existing raw waveform contains a weak signal: several methods improve point
estimates on selected splits, and the frozen model had non-random discrimination
on VitalDB. However, the increment is not stable across MIMIC patients, care units,
time, and a shuffled-waveform placebo. The failure is no longer explained by one
of the common repairable causes tested here:

- not only calibration;
- not only network capacity or initialization;
- not only aggressive temporal compression;
- not only binary-horizon supervision;
- not only the old event-label bug.

With 161 MIMIC waveform patients, forcing promotion would amount to selecting a
favorable split. The event head therefore remains research-only and fail-closed.

## What would change the conclusion

The next valid evidence must come from a larger, independent waveform cohort with
the preprocessing and event definition frozen before evaluation. Additional
anchor-safe bedside assessment and treatment-intent observations may be included,
but post-anchor treatments and future clinical decisions remain forbidden.

## Artifacts

- `artifacts/raw_waveform_probability_calibration_audit.json`
- `artifacts/multitask_waveform_jepa_audit.json`
- `artifacts/raw_waveform_event_jepa_15m_morphology_audit.json`
- `artifacts/low_rank_waveform_event_audit.json`
- `artifacts/raw_waveform_event_jepa_15m_corrected_audit.json`
- `artifacts/waveform_event_multihorizon_support.json`
- `artifacts/multihorizon_waveform_event_audit.json`
- `artifacts/raw_waveform_deep_ensemble_audit.json`

All artifacts are aggregate-only. Row-level waveform caches remain outside the
repository.
