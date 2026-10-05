# Fold 2 Continued Reconstruction Check

The fold-2 fit patients and scalers are identical to fold 0. The original
eight-epoch checkpoint states match, and the epoch-9/10 fold-0 continuation
preserves the same deterministic fit computation. Fold 2 is therefore an
equivalent trained fit, not another independently trained model.

Fold-2 selection patients differ from fold-0 selection patients. We ran a
separate fixed 7,167-query monitor for both arms at epochs 8, 9 and 10. The
epoch-8 scores reproduced the frozen baseline exactly.

| Fold-2 fixed monitor, normalized target macro | Epoch 8 | Epoch 10 |
|---|---:|---:|
| Real patient MAE | 0.856151 | 0.860299 |
| Real component MAE | 0.792579 | 0.795414 |
| Process patient MAE | 0.903192 | 0.905700 |
| Process component MAE | 0.840650 | 0.845292 |

Thus the real model's fixed-monitor MAE worsens on fold 2, even though several
individual targets improve. This contradicts a claim that continuation is
uniformly helpful on the fixed monitor.

Full outer inference on fold-2 test patients completed separately: 363,526
patient-target pairs from 14,845 patients. The new and original predictions
match on patient, target and fold keys exactly. Equal-target, fit-normalized
macro scores:

| Fold-2 full outer | Epoch 8 | Epoch 10 |
|---|---:|---:|
| Real patient MAE | 0.840887 | 0.840640 |
| Real component MAE | 0.825005 | 0.824666 |
| Process patient MAE | 0.892140 | 0.896812 |
| Process component MAE | 0.876682 | 0.882121 |

Real error falls by only about 0.03% patient and 0.04% component on the full
outer grid, while process error rises. Twenty-six of 47 targets have lower
real-model MAE at both levels. This descriptive full-grid result is opposite
in sign to the sampled selection monitor for the real model. The two samples
use different patients and sizes; this is evidence against treating the small
monitor as a faithful stand-in for the full outer role. It does not establish
sampling as the sole cause, and no target has passed a new promotion gate.

Evidence: `artifacts/native_reconstruction_monitored_fold2_reuse_20260929.json`,
`artifacts/native_reconstruction_monitored_continuation_summary_20260929.json`,
and private `continuation_outer_20260929/fold2.parquet` plus its hash metadata.
The outer development data have been inspected previously; they are not a new
external lockbox. No release, interval, registry or runtime behavior changed.
