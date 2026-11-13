# Completed History-Safe Site Training And Evaluation

All three matched arms finished the fixed eight epochs without inspecting
evaluation during selection.33targets jointly trained on536patients from11sites;
179patients/11other sites selected checkpoints. Evaluation used178patients from
11further sites,4637patient-target pairs. All patient-linked site histories are
partition-disjoint under the frozen current-cohort contract. Historical warm-start
pretraining independence remains uncertified.

Each epoch consumed all12993fit pairs through17688balanced updates; each arm
completed141504updates. Real selected8, process4, placebo1. Target-equal tune
MAE: initial1.920125, real1.808712, process1.881069, placebo1.918042. Real still
selecting the last epoch does not establish convergence.

## Heldout Results

Nominal paired patient95% intervals favor real over population, trained process
and placebo simultaneously for12targets:

albumin, ALT, AST, bicarbonate, BUN, calcium, chloride, creatinine, platelets,
PTT, total protein, troponin I.

Population gains19targets, placebo gains20, incremental process gains12.
Lipase and triglycerides show nominal harm versus allthree references. They
remain explicit failures; no changes to thresholds or outcome-based exclusions.

| Target | Real minus process MAE | Nominal95% interval |
|---|---:|---|
| Creatinine | -0.137193 | [-0.222439,-0.052770] |
| Bilirubin | -0.066266 | [-0.174905,0.039262] |
| O2 saturation | +0.000650 | [-0.021941,0.023453] |
| Glucose | -0.054508 | [-0.113879,0.003397] |
| Triglycerides | +0.355813 | [0.104074,0.625015] |
| Troponin I | -0.297115 | [-0.564968,-0.034310] |

Values are normalized current-reconstruction errors, not physical units or future
forecast errors. Troponin I has58patients and real normalized MAE38.1261 versus
population38.5730: the small relative gain is not proof of adequate absolute
accuracy, robust tail behavior, or clinical utility.

All39catalog statuses remain visible. Amylase, anion gap, direct bilirubin, BNP,
LDH and TSH lack fit/tune support in the frozen common partition and are not
trained or counted as passes.

## Interpretation

The corrected fine-tuning/evaluation split still reveals incremental numerical
reconstruction information in the same shared-body architecture. Therefore the
earlier site-overlap issue does not explain away all shared-state signal.
It also does not solve all targets: weak increments and real harm remain.
Changed patient composition and target support prevent claiming that12here
versus prior counts is a paired accuracy repair benefit.

These are single-development-split, unadjusted comparisons on repeatedly used
data. They do not fulfill seven-seed, multiplicity, external-lockbox, future
trajectory, interval or runtime promotion requirements. Tune is not independent
validation. No serving source should be replaced from this report.

Next diagnostics should separate target-specific fitting/selection behavior
for the remaining harms from source-site support shifts, while keeping all
targets in scope. Do not keep changing heads or choose epochs on this evaluation.

## Artifacts And Checks

- `artifacts/native_reconstruction_site_training_epoch8_20260917.json`
- `artifacts/native_reconstruction_site_evaluation_20260917.json`
- Private weights/optimizer/RNG and predictions: durable Git-ignored storage.
- Eight focused site-partition, process-wiring and complete-sampling tests pass.
- All input/model bindings checked; selected epoch and evaluation counts verified.
- No runtime weights, registry, calibrators or causal claims changed.
