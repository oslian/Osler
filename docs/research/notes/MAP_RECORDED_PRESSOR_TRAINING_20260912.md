# Recorded Pressor Increment Trial

Seven source splits, expanded MAP fit and unchanged old tune. Existing raw
full-history HGB baseline versus equal-width recorded-support versus recorded
rate+support. Same200iterations, source-only choice of7/15/31leaves,
patient-equal training weights. Baseline extra columns constant; support has
constant rate block. All seven baselines exactly reproduce expanded tree
predictions. This is a non-neural information diagnostic, not JEPA retraining.

## Results

Median patient-equal MAE, mmHg:

| Scope | Baseline | Support | Rate |
|---|---:|---:|---:|
| Old source evaluation | 9.14915 | 9.10641 | 9.13632 |
| All source evaluation | 8.89262 | 8.83405 | 8.89208 |
| Old held-out hospitals | 10.41079 | 10.47958 | 10.39982 |
| All held-out hospitals | 10.21392 | 10.21770 | 10.14075 |

Common20k patient-bootstrap: rate vs baseline0/7significant gain or harm in
every scope. Rate vs support1/7gain on all hospital rows, otherwise0gain;
old source1/7harm. Rate beats persistence7/7except old hospital5/7. These are
not independent replications and no external untouched database is used.

## Decision

No stable rate-context increment established. Exact minute-unit repair remains
correct, but current rate snapshots do not establish an accuracy repair.
The trial does not test complete change/start/stop histories or causal drug
effects. Documentation support is not a randomized placebo. Rate dimensions
absent in source fitting are masked. No conformal audit was run.

Infusionoffset lacks independent availability; entire trial is retrospective
recorded-time evidence. It cannot promote a live forecast. Source selection
never uses hospital results. No runtime weights, registry or calibrator changed.
Evidence: `artifacts/map_recorded_pressor_training_20260912.json`; all17direct
bindings verified. Private predictions remain out of repository.
