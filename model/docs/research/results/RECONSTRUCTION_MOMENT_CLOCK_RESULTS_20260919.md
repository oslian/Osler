# Moment-Clock Result

The fixed eight-epoch experiment and both reused-cohort evaluations completed.
Only AdamW betas changed from (.9, .999) to (.9**528, .999**528); source,
initialization, example order, learning rate, decay and accumulation remained
as specified in RECONSTRUCTION_MOMENT_CLOCK_PROTOCOL_20260919.md.
Source selection chose epoch5, score1.8161814284, versus original accumulation
epoch7, score1.8178456141. Slightly better selection did not transfer to the
aggregate evaluation score. Different selected epochs remain part of the
predeclared training/selection intervention, not an isolated endpoint update.

| Reused cohort | Original accumulation | Moment-clock candidate | Relative error change |
|---|---:|---:|---:|
| Broad | 1.279048825 | 1.285662869 | +0.517% |
| Selected | 2.893759874 | 2.927830058 | +1.177% |

Scores are target-equal normalized current-reconstruction MAE, not clinical
units, accuracy percentages, or future forecasting. Each cohort has13/33
numerically improved targets; neither has a target passing every reported
corrected patient/site comparison. The other six targets remain untrained.

Glucose improves slightly in both cohorts (.980696->.978784 broad;
1.823221->1.819659 selected). BUN and creatinine improve selected but worsen
broad. ALT worsens both; AST and troponin improve broad but worsen selected.
These numerical changes are not individual significance or promotion claims.

## Decision

This moment-clock adjustment is not an aggregate repair. Together with the
negative decay-only trial, it rules out these specific clock-matching recipes
as demonstrated solutions; it does not prove all optimizer designs ineffective
or establish a unique cause of the remaining target/cohort tradeoffs. Matching
coefficient half-lives does not make averaged-gradient Adam equivalent to
sequential Adam. Both moment coefficients changed together.

The same83874 broad and4637 selected patient-target pairs were checked against
their frozen references. Hash bindings, truth equality, uniqueness and complete
pair coverage were enforced by the evaluator. The cohorts have been repeatedly
inspected. Matched new process/placebo controls, independent external evaluation,
future forecasting and interval calibration were not completed in this trial.
No release weights, registry, intervals, serving behavior or promotion changed.

Evidence:
- artifacts/native_reconstruction_moment_clock_source_20260919.json
- artifacts/native_reconstruction_moment_clock_evaluation_20260919.json
- scripts/train_moment_clock_source.py
- scripts/evaluate_moment_clock_source.py

Patient rows and checkpoints remain in ignored local data storage.
