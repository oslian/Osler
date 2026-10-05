# MAP Summary Component Training

## Fixed Comparison

All arms keep identical full fixed-lag inputs. Append a same-width summary
block containing (1) own MAP summaries only, (2) own plus other support
summaries, or (3) all numeric and support summaries. Removed columns are
constant zero, not missing observations. Source patient support and leaf
selection, labels, partitions and tree settings are unchanged. Seven seeds,
three arms, three leaf settings. The complete-summary control reproduces
prior saved predictions exactly in all seven seeds.

## Result

Median patient-equal MAE, mmHg:

| Input | Source evaluation | Hospital |
|---|---:|---:|
| Original full lag | 9.42906 | 10.59785 |
| Add own summaries | 9.09256 | 10.19969 |
| Add other support too | 9.12918 | 10.16225 |
| Add other numbers too | 9.16463 | 10.13827 |

Common20k patient bootstrap: own-summary versus original gain3/7 in each
role, harm0. Each successive other-support and other-numeric increment has
gain0/7,harm0/7 in both roles. These are sequential conditional comparisons,
not a full factorial interaction estimate or an equivalence test.

The measured repair signal includes own-history summaries. It is not evidence
that other-variable measurements improve forecasting. This supports improving
the fidelity of own temporal representation before crediting body coupling;
it does not prove cross-organ information impossible or uniquely attribute
each summary statistic. Full lag still contains other-variable observations.

No neural encoder trained, no intervals/time/MIMIC gate, no promotion or
release modifications. Weights and patient predictions remain private.
Evidence: `artifacts/map_summary_components_20260912.json` and
`scripts/audit_map_summary_components.py`.
