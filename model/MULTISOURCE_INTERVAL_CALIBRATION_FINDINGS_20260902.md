# Multi-source 90% interval calibration

- Frozen point heads attempted: 37/41.
- Corrected NWICU point gate: 23 cells.
- Point and interval gates together: 16 cells.
- Point model weights and the existing production release were not changed.

## Correction to the previous 37-cell claim

The previous NWICU normalization replaced real care-unit labels with
`unknown`. A single-group bootstrap was incorrectly counted as a
care-unit pass. This audit restores the original labels, requires at
least two supported units, and weights patients equally within units.
The original 37-point-pass / 5-latent-pass claim is not a complete
external gate result. Latent-vs-measurement ablations must also be rerun
with corrected care-unit provenance before asserting unique latent gain.

## Frozen calibration protocol

1. Reproduce the original 15,000-row-cap MIMIC selection pool for each head.
2. Exclude ALL patients in that pool, including their other stays and anchors.
3. Use seven predeclared patient splits in the unused MIMIC pool. Choose
   one outcome-independent anchor per calibration patient and the exact
   ceil((n+1)*0.90) order statistic of absolute errors.
4. The deployable radius uses seed 7, fixed before evaluation. No seed,
   nominal level, radius, or model is selected using NWICU residuals.
5. Check both row and patient-equal coverage, supported-careunit pooled
   and equal coverage, and late-time coverage against [0.87, 0.93].
6. A purged MIMIC early/late stress test excludes shared patients and
   crossing future-label windows. The time axis is released timestamps,
   not a claim of prospective real-calendar generalization.
7. Test the fixed MIMIC radius on NWICU without recalibration. NWICU point
   results had already been inspected; this is not a newly untouched lockbox.

Intervals concern individual future observations, not confidence intervals
for a population mean. Nominal coverage is marginal under exchangeability;
repeated-window and cross-database coverage are empirical tests. Passing
the aggregate care-unit gate does not guarantee 90% in every small subgroup.

## All attempted cells

| Cell | Radius (+/-) | Full width | NWICU row coverage | NWICU patient coverage | Corrected point gate | Point + interval | Failures |
|---|---:|---:|---:|---:|---|---|---|
| glucose@1h | 60.0000 | 120.0000 | 0.8731 | 0.8725 | True | False | nwicu_careunit |
| map@1h | 17.9938 | 35.9876 | 0.9236 | 0.9196 | True | True | - |
| platelets@1h | unavailable | unavailable | unavailable | unavailable | False | False | insufficient_unused_mimic_calibration_patients |
| glucose@3h | 61.3587 | 122.7175 | 0.8802 | 0.8815 | True | True | - |
| heart_rate@3h | 15.5729 | 31.1459 | 0.8942 | 0.8947 | False | False | nwicu_point |
| hematocrit@3h | 4.6399 | 9.2797 | 0.9255 | 0.9197 | False | False | nwicu_point |
| hemoglobin@3h | 1.4517 | 2.9034 | 0.9283 | 0.9221 | False | False | nwicu_point |
| map@3h | 18.1169 | 36.2337 | 0.9004 | 0.8929 | False | False | nwicu_point |
| platelets@3h | 42.7915 | 85.5831 | 0.8648 | 0.8557 | False | False | nwicu_patient, nwicu_careunit, nwicu_time, nwicu_point |
| respiratory_rate@3h | 7.4973 | 14.9946 | 0.8808 | 0.8882 | True | True | - |
| creatinine@6h | 0.4000 | 0.8000 | 0.8872 | 0.8954 | True | False | mimic_time |
| glucose@6h | 71.3951 | 142.7903 | 0.8894 | 0.8904 | True | True | - |
| heart_rate@6h | 19.0000 | 38.0000 | 0.9109 | 0.9114 | True | True | - |
| hematocrit@6h | 4.4641 | 8.9283 | 0.9140 | 0.9044 | True | True | - |
| hemoglobin@6h | 1.4855 | 2.9709 | 0.9239 | 0.9167 | False | False | nwicu_point |
| map@6h | 19.6314 | 39.2627 | 0.9020 | 0.8923 | False | False | nwicu_point |
| respiratory_rate@6h | 8.0000 | 16.0000 | 0.8876 | 0.8927 | True | True | - |
| bicarbonate@12h | 4.0869 | 8.1738 | 0.9108 | 0.9076 | False | False | nwicu_point |
| chloride@12h | 5.3270 | 10.6540 | 0.9307 | 0.9258 | True | False | nwicu_patient, nwicu_careunit |
| creatinine@12h | 0.5000 | 1.0000 | 0.9110 | 0.9216 | True | False | mimic_patient_splits, mimic_careunit, mimic_time |
| glucose@12h | 73.5999 | 147.1997 | 0.8788 | 0.8810 | True | True | - |
| heart_rate@12h | 21.0000 | 42.0000 | 0.9010 | 0.9061 | True | True | - |
| hematocrit@12h | 4.9000 | 9.8000 | 0.9291 | 0.9234 | False | False | nwicu_point |
| hemoglobin@12h | 1.6583 | 3.3166 | 0.9384 | 0.9324 | False | False | nwicu_patient, nwicu_careunit, nwicu_time, nwicu_point |
| map@12h | 20.6521 | 41.3043 | 0.9022 | 0.8903 | False | False | nwicu_point |
| o2sat@12h | 4.0000 | 8.0000 | 0.8846 | 0.8895 | True | True | - |
| potassium@12h | 0.7542 | 1.5085 | 0.9241 | 0.9274 | True | True | - |
| respiratory_rate@12h | 7.8858 | 15.7715 | 0.8700 | 0.8817 | False | False | nwicu_point |
| bicarbonate@24h | 5.0000 | 10.0000 | 0.9250 | 0.9236 | True | False | nwicu_time |
| bun@24h | 13.0000 | 26.0000 | 0.8887 | 0.8986 | True | False | mimic_patient_splits, mimic_careunit, mimic_time |
| heart_rate@24h | 22.5372 | 45.0744 | 0.8956 | 0.8974 | True | True | - |
| map@24h | 21.1399 | 42.2798 | 0.8975 | 0.8880 | False | False | nwicu_point |
| potassium@24h | 0.7653 | 1.5307 | 0.9192 | 0.9234 | True | True | - |
| respiratory_rate@24h | 8.3883 | 16.7766 | 0.8813 | 0.8931 | True | True | - |
| sodium@24h | 5.5081 | 11.0162 | 0.9210 | 0.9244 | True | True | - |
| glucose@48h | 70.2783 | 140.5567 | 0.8533 | 0.8666 | True | False | nwicu_patient, nwicu_careunit, nwicu_time |
| heart_rate@48h | 24.7763 | 49.5526 | 0.8933 | 0.8990 | True | True | - |

## Usage and boundary

The new `models/multisource_interval_calibrator.json` binds only to the
frozen checkpoint, target-head bundle, and feature contract by SHA256.
The candidate registry is compiled FROM this audit; it is not a circular
dependency of the calibrator. Never apply these radii to the old release.

```python
from pathlib import Path
from whole_body_jepa.multisource_intervals import load_calibrator, prediction_interval
calibrator = load_calibrator(Path('models/multisource_interval_calibrator.json'), Path('models'))
# point must come from this bound multi-source head for this patient/horizon.
interval = prediction_interval(calibrator, 'map@1h', point)
```

Failed, unsupported, or missing cells return null bounds. Patient-level
residual caches stay under /private/tmp and are not release artifacts.
No next-measurement-time model, causal claim, or clinical promotion was added.

Method references: [Angelopoulos and Bates, split conformal tutorial](https://github.com/aangelopoulos/conformal-prediction)
and [Dunn et al., hierarchical repeated observations](https://arxiv.org/abs/1809.07441).
