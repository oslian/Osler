# Frozen Numeric Ablation Protocol

Question: does the frozen cross-fit reconstruction model depend adversely on
bicarbonate values for creatinine in the group with weak bicarbonate-target
association? Potassium and hemoglobin are fixed contrast inputs, not a search
over variables. The hypothesis was generated after outer inspection, so this
is diagnostic development, not confirmatory validation or promotion.

Use all eligible creatinine outer patients in all five folds, both groups,
unchanged checkpoints/scalers/labels. Reproduce original predictions before
each intervention (absolute tolerance1e-6 in original normalized units).
Set one variable's numeric history to zero in fitting-fold standardized units
(the fitting-fold location). Preserve masks, ages, timestamps, event count,
other variables and target/sibling masking. Zero is not missing. Run three
separate interventions, never combine or choose one using outer scores.

Compare paired absolute error with unchanged real predictions and retained
process/population references. Report selected/broad means, component means,
and whether prediction changes occur without any observed intervened variable
(must not). No significance/promotion gate, no deletion of adverse patients,
no test-derived model selection. This is an out-of-distribution numerical
sensitivity test, not a clinical intervention or causal effect. Improvement
would justify a SOURCE-TRAINED follow-up, not shipping the altered predictor.
