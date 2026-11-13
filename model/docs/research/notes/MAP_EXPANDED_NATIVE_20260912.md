# Expanded Native MAP Training

## Question and Fixed Comparison

Does removing the unrelated potassium-future eligibility requirement from MAP
fitting repair native JEPA accuracy or the incremental benefit of full history?

Seven patient splits, two native JEPA arms (own MAP / full source-supported
history), 32 epochs and 16 steps per epoch. Expanded fitting uses 18,898-20,329
rows and 805-843 patients. The original restricted source-tune and evaluation
rows stay fixed. Old controls are saved predictions, not fresh control fits.
Old-history and source-transform equivalence were checked before training.
Full includes insulin context; own excludes it. This is a native-history
diagnostic, not validation of the complete persistent anatomical architecture.

## Results

Median across seven splits of patient-equal MAE, mmHg, on the same old scoring
rows. Lower is better.

| Model | Source held-out patients | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.71335 | 11.58096 |
| Old own | 8.94119 | 10.25143 |
| Expanded own | 8.97705 | 10.03774 |
| Old full | 9.34468 | 10.43047 |
| Expanded full | 9.32932 | 10.39056 |

Patient paired bootstrap, 20,000 draws per comparison, 95% CI:

| Comparison | Source gain / harm | Hospital gain / harm |
|---|---:|---:|
| Expanded own vs old own | 0/7 / 2/7 | 1/7 / 0/7 |
| Expanded full vs old full | 0/7 / 0/7 | 0/7 / 0/7 |
| Expanded full vs expanded own | 0/7 / 4/7 | 0/7 / 0/7 |
| Expanded full vs persistence | 7/7 / 0/7 | 2/7 / 0/7 |

## Decision and Limits

The cohort defect is real and repaired in these candidate caches, but expanded
fitting does not establish a stable native JEPA improvement. Additional full
context still does not beat own history. This rules out this selection repair
as a sufficient solution, not all information value or all model designs.

More repeated observations, patient composition, source normalization and
sampling exposure change together. Fixed compute is not a convergence test.
Scoring is restricted to old eligible rows; the original overall cohort is
still restricted, not all eICU. Seven splits are not independent replications.
No MIMIC, time-heldout, conformal or complete promotion gate was run here.

All seven reports are complete; their 49 direct file bindings matched at
verification. Evidence: `artifacts/map_expanded_native_{seed}_20260912.json`
for seeds 7,19,31,43,59,71,83. Predictions and checkpoints remain private.
No release weights, registry, intervals or serving behavior changed.
Status remains candidate_only; causal_claim_allowed remains false.
