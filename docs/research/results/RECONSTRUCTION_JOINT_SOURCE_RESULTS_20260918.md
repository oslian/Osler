# Joint-source Reconstruction Results

Three eight-epoch arms complete,283008updates each; real/process/placebo selected
6/1/0. Same model/state architecture,1072fit/180encoder patients, equal group-target
sampling. Five median/sampler tests pass. Cohort tags do not enter inference.
Fixed checkpoints evaluated on83874broad and4637selected-cohort pairs separately.

| Cohort | Old model | Broad-only model | Joint-source model |
|---|---:|---:|---:|
| Broad3433patients | 1.36561947 | 1.30857997 | 1.30536257 |
| Selected178patients | 2.93423214 | 3.00741985 | 2.91225798 |

Target-equal normalized MAE, not clinical units. Joint improves old pooled MAE
on30/33broad and19/33selected targets. Thus average old-cohort harm is repaired
without losing average broad benefit, but not every target is retained.
Selected glucose1.65141->1.75468,bicarbonate.97608->1.00597 andO2sat.70980->.75684
remain numeric harms. Selected BUN.91290->.87594 andcreatinine1.45341->1.41103
improve; broad BUN.97015->.91300 andcreatinine1.52435->1.46430 improve.
Joint broad creatinine is slightly worse than broad-only1.45428.

No target passes the full reported source-supported corrected patient/site gate
versus old/broad-only/population/process/placebo. The broad-only comparison is an
additional diagnostic reference, not a replacement for old control requirements.
Keep individual errors and support failures visible. No universal repair or
promotion. All39statuses preserved;6untrained remain explicit.

Interpretation: representing both populations in one model mitigates the measured
distribution tradeoff. Not proof of a unique cause: combined training doubles
patient count and compute, and selection differs. One fit, reused confirmation,
no external/interval/future validation. Checkpoint0placebo equals population,
which is legitimate source-only selection, not an artificially failed control.
Runtime/weights/intervals serving unchanged. No raw data enters Git.

Evidence: artifacts/native_reconstruction_joint_source_training_20260917.json;
artifacts/native_reconstruction_joint_source_evaluation_20260918.json.
