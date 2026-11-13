# MAP Sampling-Gap Error Stratification

Fixed predictions on MAP-native anchors. Groups depend only on legal preanchor
raw draws versus lag snapshots, never future MAP: raw has >=2 distinct recent
draw times while snapshots have <2. Test MAP itself and any of HR/O2/RR/temp/
glucose. This is a coarse repeated-draw-support indicator, NOT a complete
event-retention or excursion-loss metric. Patients can appear in both groups.

Full-own patient-paired comparisons (seven splits):

| Group | Source gain/harm | Hospital gain/harm | Hospital patients |
|---|---:|---:|---:|
| Other-variable gap | 0/0 | 0/1 | 140 |
| No other-variable gap | 0/1 | 0/1 | 184 |
| MAP gap | 0/0 | 0/0 | 118 |
| No MAP gap | 0/1 | 0/1 | 185 |

Median hospital paired full-own MAE delta is +.02561 with other-variable gap
and +.13182 without; +.06674 with MAP gap and +.11346 without. No consistent
concentration of incremental failure in this coarse sampling-gap group.
This weakens the claim that missing second-draw support alone explains the
full-own gap. It does not establish equivalence, prove raw events useless,
or replace a train-matched event representation experiment.

No fitted routing, threshold tuning, retraining, calibration or promotion.
Evidence: `artifacts/map_native_sampling_gap_20260912.json`. Runtime unchanged.
