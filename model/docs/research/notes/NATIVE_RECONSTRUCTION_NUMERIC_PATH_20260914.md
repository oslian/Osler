# Frozen Numerical-Input Path

Report: `artifacts/native_reconstruction_numeric_path_20260914.json`.
Selected expanded real checkpoint is unchanged. Every original prediction
reproduces within1e-7 before intervention. Two tests check that only the first
N numeric channels change and original tensors remain untouched.

Set all normalized numerical channels to zero while preserving observation
masks, ages, presence channels, event times, history lengths and patient roles.
Zero here means inherited normalization center, NOT missing or a physical zero.
Targets and their siblings remain fully hidden in both conditions.

Using nominal unadjusted95% paired patient-bootstrap comparisons, neutralization
harms22 targets in evaluation and17 in hospital. It improves TSH in evaluation
only; no target improves significantly in hospital.

Of the prior14 nominal both-role population gains,12 are harmed in BOTH roles:
albumin, alkaline phosphatase, ALT, AST, bicarbonate, glucose, heart rate,
hematocrit, hemoglobin, platelets, total protein, WBC.
Chloride is harmed only in hospital; PTT only in evaluation at this threshold.

| Target | Evaluation original -> neutral | Hospital original -> neutral |
|---|---:|---:|
| Albumin | 0.5494 -> 0.7785 | 0.5580 -> 0.8389 |
| Glucose | 1.3315 -> 1.3849 | 1.6921 -> 1.7450 |
| Creatinine | 1.6250 -> 1.6770 | 1.3345 -> 1.4007 |
| MAP | 0.7742 -> 0.7672 | 0.8412 -> 0.8425 |
| Magnesium | 1.0083 -> 1.1044 | 1.0422 -> 1.1347 |

All errors are normalized. Creatinine and MAP differences are not significant
in either role. Magnesium relies on numerical inputs yet still underperforms
population in hospital; input dependence is not sufficient accuracy.

This rejects the narrow hypothesis that these frozen outputs are entirely
independent of numerical inputs. It does NOT establish that all gains are
physiological, that zeroed inputs are in-distribution, or that a separately
trained support-only model would fail. No causal organ interpretation or new
promotion follows from this inference-only intervention. All CIs are nominal
and reused-data diagnostics, not multiplicity-adjusted confirmation.

The next discriminating comparison is a matched TRAINED observation-process
control under the same source selection and exposure, not deployment of zeroed
inputs or selecting only favorable targets. No weights/runtime/interval changes.
