# Coverage Sampling: Source Selection

All five paired inference folds completed with frozen eight-epoch models.
Compare weighted no-replacement cycles with the original replacement sampler,
at the same update budget and input contract. This is masked current-value
reconstruction, not future forecasting or a serving-model promotion.

## Descriptive Results

Errors below are absolute errors divided by the fold's fit-only target scale,
then equally averaged over the 47 targets. Each patient-target has one query.
Component errors first average patients within each provenance component.

| Fold | Original patient error | Coverage patient error | Original component error | Coverage component error |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.845113 | 0.840887 | 0.827880 | 0.825005 |
| 1 | 0.843795 | 0.843941 | 0.825075 | 0.824150 |
| 2 | 0.879595 | 0.872003 | 0.816347 | 0.809817 |
| 3 | 0.920094 | 0.907635 | 0.858264 | 0.849433 |
| 4 | 0.941196 | 0.943787 | 0.869077 | 0.868686 |

Patient-level macro error improves in 3/5 folds, component macro error in 5/5.
Patient target mean improvements number 26/28/29/35/23 of 47; component target
improvements number 27/26/30/27/26. These are mean differences, not significant
passes. Numeric-zero process controls also change; coverage is not an isolated
proof of learning more physiological information.

## Interpretation and Limits

The fixed sampling change affects generalization, but does not uniformly repair
it. More unique training exposure is not sufficient evidence of a complete fix.
Folds share query populations and overlapping fits; do not count their patients
repeatedly as independent evidence or interpret 5/5 as independent replication.
The fitted fold2 models are certified reuse of fold0, not independent training.
Selection predictions have not been used to change settings or select targets.
Outer development evaluation, paired confidence intervals, original-screen
retention and component harms remain pending under the fixed protocol.
No runtime, calibration, model registry or promotion changed.

## Reproduction

`python -m scripts.summarize_coverage_selection`

Requires the five private prediction files and role plan; validates prediction
hashes, patient roles, unique identities and finite outputs. Public aggregate:
`artifacts/native_reconstruction_coverage_selection_summary_20260927.json`.
Patient rows remain private.
