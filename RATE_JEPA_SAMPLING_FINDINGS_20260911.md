# Rate-query sampler diagnosis

Exact 32x16x32 source sampler replay without fitting. Across seven partitions,
training-row coverage is86.95-90.47%; distinct future-event coverage90.52-92.30%.
Whole selected epochs are24,23,29,20,4,8,26; none selects epoch32. The last
eight tune MAEs fluctuate. This differs from the earlier expanded glucose
experiment with low sampled-row coverage and late improvements.

These facts do not demonstrate convergence, sufficient representation learning,
or equal coverage for every patient. They do not support lack of broad sample
exposure as the first explanation for this particular numerical increment gap.
Do not automatically repeat the previously tested training-budget expansion.

Aggregate `artifacts/rate_jepa_sampling_20260911.json` and reproducible runner
`scripts.audit_rate_jepa_sampling`. No test outcomes used by sampler replay,
no new model, runtime, calibration, or promotion. Remaining numerical increment
instability is not explained by this diagnostic.
