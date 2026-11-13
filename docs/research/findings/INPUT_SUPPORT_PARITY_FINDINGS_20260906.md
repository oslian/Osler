# Input-Support Parity Repair

Completed 2026-09-06. Protocol and artifact directory retain the 20260905
identifier from the experiment start. This audit concerns the recent five
kidney-related 6h candidate readouts, not all release cells.

## Confirmed Problem

The preceding experiments did not control encoder input-support parity.
Both databases produced 311-dimensional readout features, but their contents
were generated under different upstream observation panels:

- The eICU AKI transition file provides 14 of the checkpoint's 53 physiological
  variables. The remaining 39 normalize to missing-value priors throughout
  readout fitting.
- MIMIC provides eligible observations for 49 of the 53 variables. Thus 35
  additional variable channels become active at transfer time.
- Extra inputs change the frozen encoder's shared state, organ representation,
  proposed value and target-future latent. A fixed feature-vector length does
  not imply the same input distribution for a downstream readout.
- This is scoped to these source-AKI-trained readouts. The pretrained shared
  encoder may have seen a broader panel; do not infer that extra measurements
  are intrinsically harmful or useless to the rest of the model.

There is a second confirmed design difference: the source is **AKI-selected and
onset-anchored**, whereas the MIMIC evaluation is **whole-ICU and admission-anchored**.
This experiment isolates the input-panel issue; it does not correct or isolate
the effect of the population-selection difference.

## Repair and Controls

Learn an input-support contract from eICU train-role inputs only. Before encoding
MIMIC, mask unsupported values AND ages across the complete patient history,
not just the current anchor or the final 311-vector. Keep canonical positions,
observed zeros, current targets, future labels and all patient selections intact.

Supported variables: bicarbonate, BUN, creatinine, glucose, heart rate, lactate,
MAP, O2 saturation, potassium, respiratory rate, sodium, temperature, urine output,
and WBC.

The source-equivalence assertion verifies that none of the excluded variables
has an eligible observation anywhere in the source cohort. Thus no source
encoding or calibration is changed. Source-heldout inputs are used only to
check this assertion, not to choose the mask.

No weights, thresholds, conformal scales or label windows were tuned. All 8
original point arms reproduce the preceding MIMIC audit to 1e-8 before masking.
The aligned path reuses the same expert, binary selector and amplitude heads.
Each target retains 12,000 selected patients, one anchor each; target cohorts
overlap. Release and predecessor hashes are recorded.

## MIMIC Point Results

Conditional-amplitude readout, lower MAE is better:

| Target | Original input | Source-aligned input | Persistence | MAE reduction |
|---|---:|---:|---:|---:|
| Creatinine | 0.193685 | 0.186197 | 0.185562 | 3.87% |
| BUN | 3.058475 | 2.944822 | 3.029250 | 3.72% |
| Urine output | 104.205102 | 104.167000 | 111.919492 | 0.04% |
| Potassium | 0.383655 | 0.369282 | 0.428897 | 3.75% |
| Bicarbonate | 1.879622 | 1.835171 | 1.911842 | 2.36% |

Paired bootstrap for aligned minus original MAE:

| Target | Delta [95% CI] | Patient/unit/late input-improvement gate |
|---|---|---|
| Creatinine | -0.007488 [-0.008844, -0.005988] | Fail: not all subgroups |
| BUN | -0.113653 [-0.129690, -0.096836] | Pass |
| Urine output | -0.038102 [-0.435494, +0.343863] | Fail |
| Potassium | -0.014373 [-0.016201, -0.012598] | Pass |
| Bicarbonate | -0.044450 [-0.052425, -0.037184] | Fail: not all subgroups |

The fixed expert, without the conditional-amplitude readout, also improves:
creatinine 0.202701 -> 0.188526; BUN 3.096715 -> 2.948249;
potassium 0.384478 -> 0.367297; bicarbonate 1.885986 -> 1.827261.
Urine remains approximately unchanged (102.943139 -> 103.053106).
Therefore the gains cannot be attributed solely to a better persistence selector.
This paired computational intervention establishes input-support drift as a
contributor to observed prediction error, not a biological causal claim.

## What Still Fails

- Creatinine's excess MAE versus persistence falls from +0.008123 to +0.000635,
  but its CI [-0.000764, +0.002160] crosses zero. It is not a new win.
- BUN now beats persistence in pooled and all seven patient resamples, and in
  the late-relative-time subset. Unit-balanced delta is -0.035026 with CI
  [-0.121230, +0.062453], so the full persistence gate still fails.
- Potassium beats persistence and the shuffled control, and passes MIMIC
  coverage. But the conditional readout loses to the aligned fixed expert
  (0.369282 versus 0.367297). Extra personalized adoption is not justified.
- Urine gains essentially nothing from input alignment and still loses to its
  fixed expert. Bicarbonate improves pooled MAE but remains unit-unstable.
- The earlier source-role candidate gates failed and are unchanged. A transfer
  improvement cannot retroactively erase those failures.

**New complete candidate promotions: 0/5.** This does not negate the verified
input repair; it separates a reproducible error source from full deployment
eligibility of the more complex candidate readout.

## Uncertainty Without Retuning

The same source-calibrated scales are reused. Input alignment changes predicted
distribution shapes, not the calibration objective or thresholds.

| Target | Original patient coverage | Aligned patient coverage | Aligned unit coverage | Mean width reduction | Coverage gate |
|---|---:|---:|---:|---:|---|
| Creatinine | 0.96900 | 0.94950 | 0.94109 | 18.92% | Fail |
| BUN | 0.95592 | 0.93475 | 0.94852 | 16.77% | Fail |
| Urine output | 0.94725 | 0.93592 | 0.90804 | 8.23% | Fail |
| Potassium | 0.93875 | 0.89775 | 0.87241 | 18.77% | Pass |
| Bicarbonate | 0.96142 | 0.93617 | 0.90897 | 20.16% | Fail |

These widths are not all validated narrower intervals. Only potassium meets
the existing patient/unit coverage band of .87-.93 in this transfer evaluation.
Per-unit descriptive results are preserved; marginal coverage is not a promise
of conditional coverage for every individual. CRPS beats the recalibrated
persistence distribution for four targets; bicarbonate's CI still crosses zero.

## Remaining Work, Precisely Scoped

The point failures cannot now be summarized as simply "insufficient inputs" or
"the model does not know when to use persistence". One measured contributor was
the training-to-inference input mismatch, and this candidate path now controls it.

The uncorrected population issue requires an independently designed source
cohort, not another gate threshold: build eICU whole-ICU measurement histories
with admission-relative anchors, include both stable and changing patients,
and align future-event provenance with MIMIC. An AKI-only cohort should not be
silently treated as an unbiased general-ICU training population. Do not select
MIMIC patients by their future change to manufacture a matched-looking win.

The same exact fixed-schema source/target contract must accompany any refit.
If broader inputs are to be used, expose them consistently during readout
training and validate the new model, rather than activating unseen channels
only at inference. If local outcome calibration is used, label it adaptation
and evaluate on separate patients; it is not zero-shot source-only transfer.

MIMIC has been used repeatedly and is a transfer-validation set, not an
untouched lockbox. No claim of a unique remaining root cause is supported.

## Artifacts and Verification

- `whole_body_jepa/input_support.py`: source-support derivation and full-history masking.
- `scripts/audit_input_support_parity.py`: paired, frozen-weight experiment.
- `tests/test_input_support.py`: 4 regression tests for train-only support,
  preserving physical zero/labels/identity, full-history masking and invalid contracts.
- `INPUT_SUPPORT_PARITY_PROTOCOL_20260905.md`: pre-outcome protocol.
- `artifacts/input_support_parity_20260905/input_support.json`: source-derived
  support, population scope and frozen model bindings, saved before target evaluation.
- `artifacts/input_support_parity_20260905/report.json`: aggregate outcomes only.

```sh
env OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 LOKY_MAX_CPU_COUNT=6 \
  /Users/chunyouchang/Desktop/llm_project/medical_jepa/.venv/bin/python \
  -m scripts.audit_input_support_parity \
  --output-dir artifacts/input_support_parity_reproduction
```

196 regression tests pass. Current runtime is not modified. The new mask is
scoped to the candidate audit and has not been silently enabled across the
release's independently validated cells. No raw patient-row data is exported.
