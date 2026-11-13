# Native MAP Matched Trial

Actual execution: 2026-09-12. The training artifact retains its historical
20260911 filename. This is a diagnostic native-history JEPA trial, not the
complete persistent anatomical-organ architecture or a runtime promotion.

## Contract

Same original queries and patient partitions; 3,925 MAP measured pairs from
1,303 patients. Current age at most 6h, future measurement at 6h +/- 1h.
Twenty-one fits: seven seeds, own MAP history, full supported context, and
support control. The control removes non-MAP numeric observations but retains
their masks/ages/cadence AND insulin event features. It is not a purely
workflow-only or fully shuffled placebo. Five added vital coordinates use
source-only rescaling; MAP itself retains its target normalization.

Source-tune selection only, 32 epochs. No interval calibration, time gate,
MIMIC validation, or runtime changes were performed in this trial.

## Results

Patient-equal MAE, median across seeds, mmHg:

| Model | New source patients | Held-out eICU hospitals |
|---|---:|---:|
| Persistence | 10.71335 | 11.58096 |
| Own MAP | 8.94119 | 10.25143 |
| Full context | 9.34468 | 10.43047 |
| Support control | 9.29123 | 10.36498 |

Common patient bootstrap: 20,000 draws, three fixed random streams all agree.
Full versus own: 0/7 gain, 5/7 harm on new source patients; 0/7 gain/harm on
held-out hospitals. Full versus support: 0/7 gain/harm in both roles.
Own versus persistence: 7/7 source, 5/7 hospital gains. Full versus persistence:
7/7 source, 3/7 hospital gains. The historical 1,000-draw report gave 4/7 for
the last comparison; use the common-draw audit, not that boundary count.
All seven selected full models also have worse source-tune MAE than own.

The failure therefore appears before external-database transfer. Numerical
benefit over persistence is not proof of incremental cross-variable value.
The cause is not yet isolated to an encoder defect or absent physiology.

## Measurement-Method Attribution

Current MAP source was reconstructed at its legal availability time; its
value and future source identity/value were checked against saved labels.
3,803 pairs use the same invasive/non-invasive method, 113 switch methods,
and nine have multiple agreeing current methods. These are retrospective
strata, not deployable selection rules.

Same-method source patients retain full-versus-own 0/7 gains and 5/7 harms.
Hospital same-method comparisons retain 0/7 gain/harm. Switch strata have
only 8-21 source evaluation patients and four hospital patients per split,
so their confidence intervals are unsupported, not evidence of equivalence.
Method switching is not a sufficient explanation of the observed source gap.
Training influence from mixed-method rows has not been independently removed.

## Evidence

- `artifacts/native_bedside_map_training_20260911.json`
- `artifacts/native_map_common_bootstrap_20260912.json`
- `artifacts/map_measurement_method_20260912.json`
- `scripts/audit_native_map_results.py`
- `scripts/audit_map_measurement_method.py`

All parent bindings were verified before evaluation. Saved patient rows and
weights remain private. No release weights, registry, calibration or serving
behavior changed. Status remains candidate_only.
