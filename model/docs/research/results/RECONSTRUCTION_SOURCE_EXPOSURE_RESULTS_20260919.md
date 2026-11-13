# Matched Source Exposure Results

All six fits completed eight epochs, 283008 optimizer updates each. Small and
expanded conditions used the same persistent without-replacement queue algorithm,
shared-state architecture, initialization seed19, loss, optimizer and fixed
90+90 encoder-selection patients. Broad training pool changed536->10201;
selected-source training536 remained fixed. Original33 heads trained; six other
catalog targets remain explicitly untrained, not failed models.

Selected epochs real/process/placebo: small8/2/0, expanded7/6/8. Expanded real
selection MAE1.810509 versus small1.834877. This is selection, not held-out proof.
All checkpoints preserved privately, queues/RNG saved; nine sampler/store tests
pass. No training interruption or outcome-based protocol change.

## Frozen Evaluation

Same83874pairs/3433patients broad and4637pairs/178patients selected as earlier
audits. These are repeatedly inspected eICU cohorts, not untouched external data.

| Cohort | Old | Small queue model | Expanded queue model |
|---|---:|---:|---:|
| Broad | 1.36561947 | 1.31825056 | 1.26976144 |
| Selected | 2.93423214 | 2.89888820 | 2.91114355 |

Target-equal normalized MAE, not clinical units or percentage accuracy. Broad
expanded improves small by3.68%, old by7.02%; selected worsens small by0.42% but
improves old by0.79%. Numeric improvement counts: expanded vs small26/33 broad,
16/33 selected; expanded vs old32/33 broad,22/33 selected. Counts do not imply
significance. Selected glucose1.77463->1.87019 worsens; BUN.89226->.86377 and
lipase1.58099->1.37548 improve. Broad troponin6.42944->5.89401 improves yet still
loses expanded population5.85396. Broad BUN.89293->.89343 barely worsens.

## Distinct Gates

The original core comparison is against old model, population, process-only and
shuffled-label controls, with existing source support, patient/site tests and
fixed39-target correction. Expanded **ALT, AST and BUN pass this core gate on
broad**, zero on selected. The added incremental comparison against the newly
trained small-queue model yields **zero targets passing all comparisons in either
cohort**. Do not collapse these into 'no core evidence', or omit the incremental
failure to claim expansion definitively promoted a target.

Neither is a serving promotion: no new independent database, repeated seeds,
future forecasting or compatible interval validation was completed. Runtime,
release weights, registry and intervals unchanged.

## Interpretation And Remaining Work

Expanded independent-patient exposure helps broad reconstruction averages and
produces three core passes, but does not solve retention/generalization for every
target. This refutes a blanket claim that increasing real source diversity cannot
help. It does not establish diversity as the sole cause: source priors, sampled
patients and selected epochs change together; dense targets still see at most
4288 of the10201 available broad patients at this fixed budget. Full-pool exposure
and the six untrained heads remain pending. Do not repeat this paired fixed-budget
experiment as untested; next localize target-level retention and prior-versus-
conditional gains before choosing another training change. No evaluation-driven
per-target checkpoint or retrospectively defined patient routing was introduced.

Evidence:
- artifacts/native_reconstruction_source_exposure_pair_20260918.json
- artifacts/native_reconstruction_source_exposure_evaluation_20260918.json
- artifacts/native_reconstruction_source_exposure_summary_20260919.json

Patient rows/checkpoints remain under ignored data/research_recovery only.
