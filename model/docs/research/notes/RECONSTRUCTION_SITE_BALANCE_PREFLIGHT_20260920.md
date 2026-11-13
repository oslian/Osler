# Complete Source Site Preflight

Selected source recovery: 626 people, three with multiple hospitals. All 98
ambiguous patient-target examples have exactly one raw stay match. Values,
masked tensors, actual event times and anchor hours match exactly. Query index
is deliberately reindexed for single-query replay, not a changed anchor. Raw
event hashes match the original construction report. Initial assumed durable
raw-event path was absent; recovery used the verified original raw files and
finished successfully. No partial report was treated as completion.

Together with broad-source recovery, the private sampling manifest now maps
335,780 accepted examples: selected536fit/90encoder, broad10201fit/3323encoder.
Every patient shard hash and identity was verified; no patient crosses groups
or roles. All available targets remain in the manifest, with trained membership
explicit so six pending heads are not silently counted as trained.

## Naive Equal-Site Sampling Is Not Benign

For N patients across K represented sites, a patient in a site with n patients
gets N/(K*n) times their original per-target weight under exact site balancing.

| Group / target | Smallest site support | Largest weight multiplier |
|---|---:|---:|
| Selected / sodium, potassium, heart rate | 1 | 48.73 |
| Broad / direct bilirubin (untrained) | 3 | 80.27 |
| Broad / triglycerides | 4 | 52.39 |
| Broad / TSH (untrained) | 2 | 49.15 |
| Broad / temperature | 20 | 41.17 |
| Broad / lipase | 4 | 28.16 |

These are sampling-weight calculations, not measured training harms. However,
the earlier repeated sparse-patient exposure cannot be ignored: naive balancing
would sharply increase influence of some individual histories. Do not prescribe
equal-site replay as a demonstrated fix. A controlled source-only intervention
must bound per-patient influence or shrink site weights toward patient-uniform
sampling, preserving the old sampler as a matched control. Its policy cannot
be chosen from evaluation outcomes. Hospital identity stays outside the encoder.

No new training or accuracy claim. Six provenance helper tests pass; all live
sample mappings passed the complete preflight. No runtime/interval/promotion.

Evidence:
- artifacts/native_reconstruction_selected_sites_20260920.json
- artifacts/native_reconstruction_site_balance_preflight_20260920.json
