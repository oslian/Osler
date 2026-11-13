# Sparse Expert and Irregular Interval Findings (2026-08-01)

## Question

Can target/horizon-specific sparse experts reduce whole-body JEPA negative
transfer, and can patient-equal conformal calibration make sparse laboratory
forecasts honest enough for promotion?

## Architecture tested

The factual router remains the teacher and persistence remains the fallback.
For each target and horizon, a sparsemax gate selects among:

1. shared slow whole-body context;
2. target-specific fast future context;
3. the target's own-organ token.

The candidate is compared with the same trained adapter with sparse experts
disabled. The residual head is zero initialized, so the initial prediction is
exactly the teacher.

## Bounded result

Dataset: eICU canonical event examples, 1,000 stays, 127,404 rows, seven
patient-held-out seeds.

The sparse expert candidate improved point forecasts consistently:

| Cell | Beats teacher | Beats persistence | Beats no-expert | Median delta vs teacher | Median delta vs no-expert |
|---|---:|---:|---:|---:|---:|
| glucose@1h | 7/7 | 7/7 | 7/7 | -0.302 | -0.162 |
| glucose@3h | 7/7 | 7/7 | 7/7 | -0.363 | -0.177 |
| heart_rate@3h | 7/7 | 7/7 | 7/7 | -0.050 | -0.020 |
| heart_rate@6h | 7/7 | 7/7 | 7/7 | -0.085 | -0.036 |
| heart_rate@12h | 7/7 | 7/7 | 7/7 | -0.120 | -0.059 |

Sparsemax produced exact zero weights and target-specific source use, so this
is not ordinary dense attention. The result supports the hypothesis that some
whole-body JEPA failures were caused by negative transfer.

## Distributional correction and promotion result

Fixed, adaptive, and post-hoc normalized conformal did not pass the complete
gate. The failure was traced to patient-dependent error-scale heterogeneity.
A target-specific Student-t scale head was therefore added. Training is
strictly two-stage:

1. train the sparse-expert point predictor;
2. freeze every point-prediction parameter and train scale heads only;
3. calibrate standardized residuals with patient-cluster conformal.

This prevents uncertainty training from sacrificing point accuracy. A matched
smoke test confirmed that sparse-expert and sparse-distributional point
metrics are exactly identical.

On 1,000 stays and seven patient splits, `glucose@3h` passed all bounded gates:

- 7/7 better than the validated teacher;
- 7/7 better than persistence;
- 7/7 better than the matched no-expert ablation;
- 7/7 patient-equal 90% conformal coverage in 0.87--0.93;
- direction accuracy 0.626--0.647;
- delta correlation 0.287--0.373;
- latent SVD stability 7/7.

It then passed hospital, care-unit, and forward-time holdouts, including
teacher, persistence, module-balanced, conformal, and no-expert requirements.
The final registry status is
`validated_target_gated_joint_research_only` for `glucose@3h` only. Every
other target/horizon remains on the existing validated source or persistence.

Reports:

- `teacher_anchored_sparse_experts_1000_7seed.json`
- `teacher_anchored_sparse_experts_adaptive_1000_7seed.json`
- `teacher_anchored_sparse_experts_normalized_1000_7seed.json`
- `teacher_anchored_sparse_distributional_twostage_1000_7seed.json`
- `teacher_anchored_sparse_distributional_twostage_external_1000.json`
- `teacher_anchored_sparse_distributional_validated_registry_20260801.json`

## Sparse laboratory intervals

MIMIC-IV sparse events were rebuilt from raw measurement timestamps and saved
as a reusable persistent cache. Calibration gives every patient equal total
weight.

Validated intervals:

- INR, 6--24h;
- INR, 24--72h.

PTT and fibrinogen pass all seven patient splits but fail the care-unit
holdout through over-coverage. Cortisol passes six of seven patient splits.
These cells remain `needs_calibration_audit`.

This confirms that irregular-event forecasting is the correct task for sparse
labs, while care-unit and patient heterogeneity still limit transferable
uncertainty estimates.

## Boundary

- factual observation/forecast only;
- no causal, counterfactual, clinical, or treatment-effect claim;
- no runtime registry change;
- failed cells continue to use the validated router, persistence, or missing.
