# Reconstruction Failure Location Across Source And Hospital

## Completed Replay

All six V2 replications (19,31,43,59,71,83) replayed every eligible fit/tune
patient-target pair using the original selected real and process-only weights.
There were 207,300 source pair evaluations across the six overlapping splits,
NOT 207,300 independent patients. Seed7's separate runner was not included in
this replay. All39 catalog targets are retained; anion gap remains unsupported
in three V2 source splits rather than being silently replaced.

Original checkpoint/input hashes, shard identities, patient partitions and
selected epochs were checked. Durable copies match original report bindings.
Each real/process target-equal tune MAE reproduces its original best-tune score
within1e-7. Fit medians are computed from fit patients only. No optimization,
new checkpoint selection, masking change or hospital-based routing occurred.

Machine summary: `artifacts/native_reconstruction_source_fit_summary_20260917.json`.
Individual reports: `native_reconstruction_source_fit_{seed}_20260917.json`.

## Distinct Failure Patterns

The table counts **numerically lower mean MAE versus trained process-only**.
It does not count formal statistical passes. Tune selected the checkpoints and
is not an independent validation set. Evaluation is the separate source-patient
holdout; hospital patients are held out from all six fits.

| Target | Fit | Tune | Source evaluation | Hospital |
|---|---:|---:|---:|---:|
| Albumin | 6/6 | 6/6 | 6/6 | 6/6 |
| Creatinine | 6/6 | 6/6 | 6/6 | 6/6 |
| Bilirubin | 6/6 | 6/6 | 6/6 | 0/6 |
| O2 saturation | 5/6 | 2/6 | 6/6 | 0/6 |
| Glucose | 6/6 | 4/6 | 5/6 | 4/6 |
| Triglycerides | 6/6 | 0/6 | 2/6 | 1/6 |
| Troponin I | 6/6 | 2/6 | 4/6 | 4/6 |
| TSH | 6/6 | 2/6 | 3/6 | 5/6 |
| Potassium | 6/6 | 6/6 | 6/6 | 5/6 |
| MAP | 5/6 | 4/6 | 3/6 | 5/6 |

Hospital TSH has only17 patients, so its mean signs cannot support efficacy.
Creatinine has nominal incremental gains4/6 on source evaluation and3/6 on
hospital, despite favorable mean signs6/6. Albumin has6/6 nominal gains in both
heldout roles. These incremental comparisons are not the full promotion gate.

## Earliest Located Failures

1. **Source generalization failure:** triglyceride gains reverse on tune in all
   six fits. Median normalized real-minus-process MAE is-0.223522fit versus
   +0.100609tune. Troponin I and TSH also frequently reverse there. A hospital
   harmonization-only repair cannot explain this earlier failure.
2. **Hospital transfer failure:** bilirubin retains favorable means on fit,
   tune and source evaluation in all six fits, then reverses in all six hospital
   evaluations. Median incremental MAE is-0.161914source evaluation and
   +0.067849hospital. This locates a transfer boundary; it does not yet identify
   whether measurement practice, patient mix, assay contract or learned support
   interactions cause it. O2 shows hospital reversal plus tune instability.
3. **Small/heterogeneous incremental gain:** creatinine and potassium often
   retain favorable means, but paired uncertainty does not consistently exclude
   zero. This is not equivalent to inability to fit or universal mean regression.

The real model fits lower MAE than process on36-39 of38-39 eligible targets per
split. Thus an entirely disconnected numerical pathway is not a plausible common
explanation. In-sample gain is not proof of biological understanding. The shared
normalization, architecture and finite training budget may still impose common
limits, and selected eight-epoch fits do not establish convergence.

## Repair Direction, Not A Claimed Repair

Do not apply one universal fix to these different patterns. Before another
training sweep, use SOURCE-only validation to test whether a proposed change
addresses its respective failure:

- For source generalization: establish a source patient learning/regularization
  contrast with the same masks, target support and all-target retention. Require
  independent source-evaluation gain, not lower fitting error alone.
- For hospital reversal: inspect per-source-hospital incremental errors and
  observation support before designing a domain-robust objective. Do not tune
  on the already inspected hospital holdout.
- For weak positive effects: retain patient-level paired uncertainty and source
  support requirements. Do not turn favorable means into promotion by loosening
  thresholds or multiplying repeated fits as independent evidence.

This replay localizes failure stages, not a unique causal root cause. It corrects
the earlier overly broad description that every target was simply unstable.
No model repair has yet been demonstrated by this diagnostic.

## Verification

Ten focused partition, process-wiring and stability tests pass. Full source
replay provides numerical integration verification through exact tune-score
reproduction for all12 selected real/process checkpoints. Private prediction
rows remain under Git-ignored durable data storage. No release weights,
runtime registry, calibrated intervals or serving behavior changed.
