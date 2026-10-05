# Nowcast-to-Forecast Cascade Findings

Date: 2026-08-28

## Question

Can an externally gated cross-target current-state estimate be fed into the
whole-body JEPA to improve a future forecast when the target is missing at the
anchor?

## Design

- Targets: bicarbonate and heart rate, the two validated cross-target nowcasts.
- Horizon: 6 hours, the common raw future label available in both eICU and
  MIMIC-IV.
- Two scopes:
  - `artificial_mask`: hide a fresh observed anchor target;
  - `natural_missing`: use windows where the anchor target is genuinely absent
    and a future target is observed.
- Comparators:
  - JEPA with its missing target input set to the training median;
  - nowcast carried forward unchanged;
  - nowcast fed into the JEPA;
  - an eICU-selected, MIMIC-frozen residual shrinkage between the last two.
- Promotion requires patient-bootstrap and domain gates in both databases and
  in both missingness scopes.

## Result

No target passed the cascade gate.

### Bicarbonate

In natural-missing windows:

| Database | Missing-input JEPA MAE | Nowcast persistence MAE | Nowcast-fed JEPA MAE |
|---|---:|---:|---:|
| eICU | 4.729 | 3.985 | 4.148 |
| MIMIC-IV | 3.514 | 3.156 | 3.232 |

The nowcast is useful, but the JEPA residual consistently makes it worse. The
eICU-selected residual weight is therefore `0.0`. Current completion should be
kept, but it must not be fed into the JEPA.

### Heart Rate

In natural-missing windows:

| Database | Missing-input JEPA MAE | Nowcast persistence MAE | Nowcast-fed JEPA MAE |
|---|---:|---:|---:|
| eICU | 17.123 | 15.506 | 15.522 |
| MIMIC-IV | 11.048 | 12.604 | 12.709 |

eICU selected a small residual weight of `0.30`, but it did not improve eICU
heldout patients significantly and regressed on MIMIC natural-missing windows.
The missingness regime is not transportable across databases.

## Interpretation

Cross-target nowcasting and future forecasting are different distributions.
The current estimate captures much of the available cross-variable signal. A
future JEPA trained mainly on observed anchors interprets the imputed anchor as
if it were measured and over-corrects. Natural missingness also identifies a
different care process and patient population than artificial masking.

## Runtime Decision

- Keep bicarbonate and heart-rate current-state nowcasts.
- Do not feed either nowcast into the future JEPA.
- Do not create a future point solely because a current nowcast exists.
- A future attempt must train with explicit imputed-anchor augmentation and a
  missingness-regime/domain-invariance gate, then pass a new external lockbox.

This remains factual research only. `causal_claim_allowed` is false.
