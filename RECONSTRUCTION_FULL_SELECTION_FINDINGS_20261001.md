# Full Source Continuation Assessment

## Question and Fixed Comparison

Does complete source-selection evidence resolve the remaining continuation
failure, or did the small fixed monitor merely misrepresent source behavior?

Use every eligible query in all five source-selection folds, all 47 targets,
the frozen epoch-8/10 real and process models, unchanged population references,
fit-only scalers and masking. This is masked CURRENT-value reconstruction,
not future forecasting. Nothing is fitted, selected or promoted here. Outer
results are already reused development evidence, not an untouched lockbox.

## Completion and Integrity

All five folds completed: 1,797,884 query occurrences, 720,470 unique
patient-target pairs and 29,580 unique people. Folds overlap in query people
and fitted populations. Fold 2 reuses the certified fit-0 model but has a
different query population; fold 0 inference is certified exact outer-2 reuse.
These are not five independent experiments.

Every prediction hash, bound dependency, selection role, complete eligible
query grid, paired identity and finite prediction was checked. The original
fixed monitor is recovered from full predictions with maximum absolute metric
difference 4.44e-16. Eighteen tests pass, including incomplete grids, wrong
roles, nonfinite values, independent patient/component weighting and absent
outer support. All patient rows remain private and ignored.

## Full Source Results

Equal-target macro normalized MAE; lower is better. These means are descriptive,
not significance tests or new promotion gates.

| Fold | Patient MAE 8 -> 10 | Component MAE 8 -> 10 | Targets improving both / 47 |
| --- | --- | --- | ---: |
| 0 | 0.840887 -> 0.840640 | 0.825005 -> 0.824666 | 26 |
| 1 | 0.843941 -> 0.840651 | 0.824150 -> 0.820347 | 20 |
| 2 | 0.872003 -> 0.875138 | 0.809817 -> 0.812774 | 16 |
| 3 | 0.907635 -> 0.908530 | 0.849433 -> 0.845635 | 19 |
| 4 | 0.943787 -> 0.936414 | 0.868686 -> 0.857693 | 22 |

Continuation improves the full patient macro in 3/5 folds and the component
macro in 4/5, but does not uniformly improve targets or populations. In fold 2,
both real-model errors increase; the process control also gets worse. A better
relative margin alone must not be described as a better absolute prediction.

## What the Complete Grid Changes

- Sampled versus full source continuation direction differs in 60/235 patient
  metrics and 33/235 component metrics. Same query population, different sample
  and weighting: the sampled monitor is not a reliable substitute for full
  source assessment.
- Full source versus full outer continuation direction still differs in
  63/233 patient and 103/233 component comparisons. Two unsupported outer
  target-fold combinations remain missing, not successes or failures.
- Thus sampled-monitor distortion is real but cannot be the sole explanation
  of poor transfer. The source/outer populations also differ. These counts do
  not establish which biological or representation mechanism causes that shift.
- Of the 26 previously base-screen-failing targets, 13 have a source component
  with at least 30 patients where real MAE is higher than the process control
  at BOTH epochs. Fifteen have such a recurring mean harm versus the fixed
  population reference; the union is 17 targets. This is not merely an outer-
  only problem. Mean signs are not statistical proof of harm or independent
  replication, since epochs and folds share data and training.

The other nine targets lack that specific supported recurring-source-harm
pattern: amylase, anion_gap, cpk, esr, ferritin, lactate, magnesium, ph and
respiratory_rate. This does not prove their failure is only low power; they
can have smaller, new, unsupported or outer-only harms. Every component is
retained in assessment. The 30-person threshold is only a descriptive support
label, never a way to remove difficult groups from a gate.

## Decision and Remaining Work

The missing full-source check is now repaired. It does NOT repair all model
generalization, justify more unsupervised epoch extension, or promote a new
forecast cell. Runtime weights, registry, calibrated intervals and release
remain unchanged.

The next diagnosis must distinguish persistent source-group errors from
outer-only changes, using the now complete paired component records. Before
training another router, check whether signed bias, scale, or weak within-group
ranking accounts for the recurring source harms under the same masked inputs.
Prior generic offset, affine, source-CV, constrained-readout and gradient-
balancing negative experiments remain evidence; do not repeat them as new
solutions. Any resulting correction must be fitted and selected only within
source data and compared against its unchanged counterpart, not selected from
outer outcomes. No unique root cause has yet been proved.

## Reproduction

- `python -m scripts.infer_continuation_selection --fold N` for folds 1-4;
  fold 0 uses its existing exact-reuse certificate.
- `python -m scripts.verify_continuation_selection --fold N` for folds 0-4.
- `python -m scripts.summarize_continuation_selection` requires all five verified
  full outputs and reproduces original epoch-8 source aggregate metrics.
- `python -m unittest tests.test_continuation_selection_summary tests.test_continuation_selection_verification tests.test_continuation_selection_reuse tests.test_fixed_reconstruction_monitor`

The five verification JSONs and
`artifacts/native_reconstruction_full_continuation_selection_20260930.json`
contain aggregate results and hash bindings only. The 20260930 artifact suffix
identifies this frozen run; completion occurred on 2026-10-01.
