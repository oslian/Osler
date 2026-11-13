# Global scale is not the native glucose repair

Protocol: `GLUCOSE_NATIVE_CALIBRATION_PROTOCOL_20260911.md`.
Aggregate: `artifacts/glucose_native_calibration_20260911.json`.
All original bindings and evaluation scores reproduce. Frozen neural weights;
14 patient-weighted affine median regressions plus 14 residual offsets fit
only on source tune patients. Both controls receive the same corrections.

| Median patient MAE mg/dL | Own | Whole | Own offset | Whole offset | Own affine | Whole affine |
|---|---:|---:|---:|---:|---:|---:|
| New source | 33.9762 | 34.8408 | 33.9934 | 34.6363 | 34.3787 | 35.6189 |
| Held-out hospital | 41.4025 | 42.4007 | 41.3074 | 42.5011 | 41.6464 | 42.5062 |

Whole-offset versus original significant gain/harm: source2/7,1/7;
hospital3/7,1/7. Whole-affine: source1/7,0/7; hospital1/7,1/7.
Neither corrected whole arm beats the equivalently corrected own arm in any
seed in either role. Affine whole harms corrected own2/7source,1/7hospital.
Overlapping splits are not independent; medians do not encode paired effects.

Source-selected whole slopes .8111-1.0269; own .8615-1.1004, all positive.
No large universal amplification is selected. Whole offsets span -2.30 to2.95.

Original median signed patient bias: source own-6.882,whole-5.674; hospital
own-12.647,whole-11.023. Whole has smaller mean bias but worse MAE. Original
row delta correlations source .7142own/.7164whole, hospital .8335/.8243.
Correlation and signed bias are descriptive, not patient-balanced accuracy or
proof of correct direction for every patient. Smaller bias can coexist with
larger absolute error, and correlations share the current-value anchor.

The tested failure is therefore not repaired by global offset/scale on legal
tuning data. Conditional heterogeneity, input semantics and finite support
remain unresolved; this does not rule out all conditional calibration. Do not
fit a hospital-specific offset after inspecting these outcomes or describe it
as external validation. Prior potassium gain-only failure remains a distinct
task, not evidence this glucose diagnostic had already been performed.

No runtime, conformal intervals, retention, external lockbox or promotion.
Inherited pretraining overlap and potassium-conditioned cohort limitations
remain. No patient rows exported. SciPy/HiGHS via existing scikit-learn used;
no new dependency or custom optimizer.
