# Source Coverage Continuation Results

All three arms continued from finalepoch8 through19, without resetting weights,
AdamW, queues or RNG. Total672144updates perarm;389136additional. The original
eight-epoch logs remain exact. Finalepoch19 exposure equals every available
broad-fit patient-target count for all33trained targets. Seven focused queue/
budget/schedule tests pass. Six untrained heads remain pending.

Source selection (unchanged90+90patients) chooses real16,process6,placebo18.
Real source MAE1.810509->1.795680. Because real selects16, it has8576draws per
group-target, not final10184. Full final coverage does NOT imply the selected
model saw every dense-target patient. Original process checkpoint remains best.

## Unchanged Reused-Cohort Evaluation

Same83874broad/4637selected pairs, truth/hash/uniqueness/coverage checks enforced.
Candidate weights frozen by source selection; no evaluation-driven replacement.

| Cohort | Previous expanded | Continued selected model | Relative change | Numerically better targets |
|---|---:|---:|---:|---:|
| Broad | 1.269761444 | 1.265321774 | -0.350% | 23/33 |
| Selected | 2.911143552 | 2.893126729 | -0.619% | 25/33 |

Target-equal normalized MAE, not accuracy percentages or clinical units.
Glucose improves both (.997046->.979312 broad;1.870194->1.786266 selected).
BUN improves both, tiny on broad. Creatinine worsens both (1.435759->1.445204;
1.420725->1.447244). Triglycerides improve broad but worsen selected.

Core comparisons against old model, population, matchedprocess and matchedplacebo
pass AST,bilirubin,calcium on broad andzero on selected. With added small-model
and previous-expanded incremental comparisons, zero targets pass everything
in either cohort. Earlier corepassesALT/AST/BUN do not all persist. These are
development reconstruction gates, not full runtime promotion or futureforecast.

## Interpretation

Additional uninterrupted optimization plus patient exposure yields small average
benefit in both cohorts and more numerical improvements, but not stable universal
retention. It is neither evidence that extra training is useless nor proof that
insufficient exposure was the sole cause. More unique patients, extra updates
and repeated sparse-patient draws change together. We did not alter architecture,
loss weighting, selection rule or allowed information to obtain this result.

Next diagnosis should retain the complete trajectory and distinguish changed
selection support from continued optimization. Expandedsource selection replay
for newepochs and source-fit retention can be compared before any new intervention.
Do not retrospectively route targets to their evaluation-best epochs. A complete
39target training protocol is still separate pending work, not abandoned scope.

Evidence:
- artifacts/native_reconstruction_coverage_continuation_20260919.json
- artifacts/native_reconstruction_coverage_evaluation_20260920.json

Evaluation source bindings verified, raw rows/checkpoints private. No new lockbox,
conformal calibration, futureforecast, release weights, registry or runtime
promotion. Both evaluation cohorts were previously inspected.
