# Nested Baseline Preparation

Source seed7, same32epoch own-history JEPA recipe and release warm-start.
Exclude residual64fit/32tune patients from both baseline fitting and selection;
holdouts unchanged. Original target scale unchanged (MAP numeric rescale identity),
non-target/event input absent. This does not use original numeric fit statistics
for active inputs; inherited release scaler/pretraining provenance caveat remains.
Baseline tune is remaining original source-tune patients. This is a nested
holdout baseline, not completed K-fold cross-fitting. Fixed budget, no tuning
on evaluation/hospital results. Save private baseline for subsequent matched
residual training. No runtime or promotion. Comparisons with original baseline
confound changed fit/selection populations and should not be called pure
cross-fitting benefits without the subsequent controlled residual experiment.
