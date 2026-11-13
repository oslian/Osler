# Patient-Equal Weight Contrast

Completed fixed source-only47target x5fold LOCO point-Huber correction.
163targetfolds have converged supported results in both weight arms. This
denominator differs from the earlier162which also required full-numeric-Huber.

Changing only correction/scaler weights to patient-equal gives:
- Component-equal evaluation: improves balanced correction109/163; improves
  unchanged81/163. Balanced correction improves unchanged55/163 on SAME163.
- Patient-weighted evaluation: improves balanced95/163; improves unchanged75/163.
- Beats BOTH unchanged and balanced at BOTH aggregation levels29/163.
- No target achieves that joint comparison in allfivefolds.
- Among358small(<30patients) component queries,163have higher MAE than balanced.

This fixed ablation supports weighting as a contributor to correction failure:
the original equal-component objective is not harmless. It does not establish
the precise pathway (sample noise versus domain-specific fitting), nor solve
the full accuracy problem. In82/163component comparisons the replacement still
does not beat unchanged. Small-domain harms prohibit blanket replacement.

Allpatients retained; same query identities/support, normalizedtotalweight1,
Huber hyperparameters and scoring. Patient/component means are descriptive,
not significance or promotion. Query records overlap across fitted folds.
No new outer inference, encoder training, future forecast or runtime change.
Two tests pass (label isolation and insufficient support). Artifact:
artifacts/native_reconstruction_patient_equal_20260925.json.

## Weight Response Sensitivity

Across163matchedroles, full-calibration Kish ESS/N versus relative component-MAE
change has Spearman rho+0.3421: more concentrated prior weights tend to associate
with larger improvement after patient-equal weighting. This is exploratory,
not causal; roles overlap, so the ordinary independent-sample p-value is NOT
reported or used. Full-role ESS also differs from eachLOCO training ESS.

Query size bins show harm rates163/358(<30),168/391(30-299),190/454(>=300).
Median relative changes are -0.00392,-0.00575,-0.00292 respectively. Thus harms
are not confined to tiny hospitals, and allthreebins have slightly improved
medians. Do not characterize the result as simply sacrificing small hospitals
to help large ones. Weighting contributes, but remaining variation crosses sizes.
Artifact: artifacts/native_reconstruction_weight_response_20260925.json.
