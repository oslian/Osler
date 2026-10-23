# Glucose-like Target Transfer Findings (2026-08-02)

## Question

Can the exact method that validated glucose also validate targets with similar
measurement density and short-term dynamics?

This audit is factual forecasting only. It does not support treatment-effect,
causal, or clinical claims.

## Method

The glucose method was held fixed:

- connected whole-body context model;
- target-specific sparse residual experts;
- `sparse_distributional` point and scale heads;
- target distributional patient-cluster conformal intervals;
- persistence and existing teacher-router anchors;
- no future treatment or future measurement features;
- 1 training epoch for the base, 1 adapter epoch, and 3 scale epochs.

Target selection used aggregate dynamics only: observation rate, measurement
age, future support, robust normalized change, changed fraction,
current-future correlation, actual-time jitter, skew, and label quality.

Patient separation was explicit:

- 184 discovery patients selected glucose-like targets;
- 738 disjoint patients were used for core/broad/group screening;
- 2,371 further patients were reserved for formal and external checks;
- reports contain aggregate results only, not row-level identifiers.

The 12 closest eligible targets were:

1. glucose
2. urine_output
3. calcium
4. map
5. chloride
6. respiratory_rate
7. wbc
8. total_protein
9. albumin
10. magnesium
11. hemoglobin
12. heart_rate

The broader 16-target ring added hematocrit, bicarbonate, potassium, and
alkaline_phos.

## Screening Results

### Core 12

On the 738-patient screening cohort, eight cells passed all four gates in all
three seeds:

- hemoglobin at 6h and 48h;
- MAP at 3h, 6h, and 12h;
- respiratory rate at 1h, 3h, and 24h.

### Broad 16

Adding the next four targets reduced the number of complete 3/3 cells from
eight to six. This is direct evidence of target-set negative transfer, so the
broader set was not selected for formal testing.

### Dynamics-specific groups

- Fast group (`glucose`, MAP, respiratory rate, heart rate): MAP at 6h and
  respiratory rate at 3h/6h passed 3/3.
- Laboratory group: only WBC at 6h passed 3/3 beyond glucose.
- Urine group (`glucose`, urine output): urine output did not pass. Removing
  the other organ signals made it worse, so its positive signal is genuinely
  cross-system rather than a single-target shortcut.

## Formal Patient-Heldout Result

The fixed core-12 configuration was evaluated with seven patient-disjoint
seeds on the reserved cohort. Only one new cell passed teacher, persistence,
no-expert, and conformal gates in all seven splits:

| Cell | Teacher | Persistence | No-expert | Conformal | Median MAE delta vs teacher |
|---|---:|---:|---:|---:|---:|
| urine_output@12h | 7/7 | 7/7 | 7/7 | 7/7 | -1.5733 |

Important near-misses included glucose at 6h, heart rate at 3h/6h,
hemoglobin at 12h, MAP at 6h/12h, and respiratory rate at 1h/3h/12h.

The fast and laboratory subgroup configurations were also evaluated over
seven seeds. Neither produced a complete new 7/7 cell. Their 3-seed gains were
therefore screening signals, not promotion evidence.

## External Result

`urine_output@12h` was tested across hospital, care-unit, and forward-time
splits while preserving the core-12 model.

With the first 1,000 stays:

- hospital: point prediction and conformal passed;
- forward time: point prediction and conformal passed;
- care unit: point estimate improved, but 56 subjects left the paired CI
  crossing zero; interval coverage was 0.93047, just above the 0.93 ceiling.

With the expanded 2,025-subject external run:

- care unit point prediction became significant;
- forward time remained significant;
- hospital-heldout point prediction became unstable in the harder 99-subject
  hospital group;
- care-unit coverage remained high at 0.93170.

No cell passed every external requirement. No runtime registry was changed.

## Interpretation

The exact glucose method does transfer useful signal, but not as a universal
promotion recipe.

1. Urine output at 12h is the strongest new cross-system candidate. It is not
   a single-target effect and benefits from broader organ context.
2. MAP, respiratory rate, heart rate, hemoglobin, and WBC show reproducible
   point-prediction signals, but their patient or interval stability is weaker
   than glucose.
3. Statistical similarity to glucose does not imply the same physiological
   generating process. Stable electrolytes and proteins remain close to the
   persistence optimum.
4. Smaller compatible target groups reduce negative transfer, but they do not
   solve care-setting heterogeneity or conformal calibration by themselves.
5. The remaining problem is primarily target-specific distribution shift and
   uncertainty calibration, not missing decoder capacity.

## Decision

- Keep the validated registry unchanged.
- Keep `urine_output@12h` as the highest-priority candidate-only cell.
- Keep MAP, respiratory rate, heart rate, hemoglobin, and WBC as bounded
  near-misses; do not expose them as validated runtime forecasts.
- Do not lower the 7-seed, hospital/care-unit/time, no-expert, or 0.87-0.93
  conformal requirements to manufacture a promotion.

## Artifacts

- `glucose_like_target_discovery_20260802.json`
- `glucose_like_core12_sparse_distributional_3seed.json`
- `glucose_like_broad16_sparse_distributional_3seed.json`
- `glucose_like_core12_external_patient_7seed.json`
- `glucose_like_core12_external_gate.json`
- `glucose_like_core12_external_gate_fullcohort.json`
- `glucose_like_fast4_sparse_distributional_3seed.json`
- `glucose_like_fast4_external_patient_7seed.json`
- `glucose_like_lab8_sparse_distributional_3seed.json`
- `glucose_like_lab8_external_patient_7seed.json`
- `glucose_like_urine2_sparse_distributional_3seed.json`

## Interval-Semantics Addendum (2026-08-02)

The candidate-only urine conclusion above used the legacy urine feature, which
mixed interval volume with hourly rate and admitted workflow/count records.
After rebuilding urine output as volume divided by its actual preceding
collection interval, `urine_output@3h` passed 7/7 patient splits and the full
hospital, care-unit, forward-time, no-expert, module-balanced, conformal,
direction/delta, and SVD gates.

The superseding result is documented in
`INTERVAL_NORMALIZED_URINE_OUTPUT_FINDINGS_20260802.md`. The old
`urine_output@12h` candidate must not be reinterpreted as evidence for the new
`mL/hour` target because the two input contracts are not interchangeable.
