# Source Ranking Before External Transfer

Six immutable V2 source-fit replays (19,31,43,59,71,83) were examined for all
available targets. All original fit/tune MAEs reproduce to1e-10. Query and
prediction hashes match. No training or selection. These are the earlier six
overlapping patient splits, NOT six repetitions of the new strict-site model.

| Target | Median fit Spearman | Median tune Spearman |
|---|---:|---:|
| Triglycerides |0.6618|0.0882|
| Lipase |0.6374|0.3572|
| Bilirubin |0.5803|0.4412|
| Glucose |0.3729|0.2692|
| Creatinine |0.5213|0.4271|

Triglyceride tune correlation ranges-0.019 to0.219 versus fit0.639 to0.715.
This places substantial ranking loss already at source patient generalization.
It is not merely a future external database calibration shift. It also does NOT
show a stable negative relation: reversing signs based on the strict evaluation
negative correlation would be posthoc and contradicted by most source fits.

Lipase retains positive tune correlation in all six runs (0.251-0.434), despite
subsequent strict-site MAE harm. Ranking and numerical calibration must be tested
separately; a positive correlation does not establish useful absolute accuracy.

Hospital attribution remains limited: triglyceride tune has zero or one site
with at least30patients in each split, lipase one. Supported triglyceride sites
have positive correlations0.046-0.445; there is insufficient broad site support
to assert a universal within-site reversal. Patients with multiple cohort sites
are retained in pooled summaries but assigned no arbitrary site. No site IDs or
patient identifiers are published.

## Repair Implications

Do not apply a universal offset, sign reversal or tail clipping. For ranking-
retaining targets, source-only readout calibration is a legitimate controlled
question. For triglycerides, require retained source patient/site ranking before
claiming calibration repairs the shared representation. Fit versus tune gap
supports overfitting/generalization failure as a symptom, not a unique proof
of excess capacity, insufficient data or erroneous measurement semantics.

All correlations are descriptive, tune selected these models, seeds overlap,
and no new significance or runtime promotion is asserted. Serving unchanged.

Artifact: `artifacts/native_reconstruction_source_ranking_20260917.json`.
