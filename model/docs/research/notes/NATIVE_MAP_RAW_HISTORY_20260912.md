# MAP Raw-History Control

## Question and Fixed Comparison

Does non-MAP numeric context provide incremental predictive information under
a non-recurrent learner on the same MAP labels and patient partitions?
This extends the completed glucose raw-history diagnostic to the newly trained
MAP task; it is not a repeat claim about glucose or an architecture promotion.

Seven patient splits, same source fit/tune/evaluation and hospital roles as
the native MAP experiment. Source-only channel support and patient-equal fit
weights. HistGradientBoostingRegressor, absolute-error loss, 200 iterations,
leaf counts 7/15/31 selected only by source-tune MAE. Delta is anchored at the
same measured current MAP. Compare own MAP, all supported observations, and
non-MAP numeric-neutral support control. Unlike the native experiment, this
fixed-lag input has no separate insulin event features. Thus comparisons
within this experiment are matched; comparisons across architectures do not
isolate only the neural representation.

## Result

Median patient-equal MAE over seven splits, mmHg:

| Source | New source patients | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.71335 | 11.58096 |
| Own MAP history | 8.99978 | 10.39641 |
| Full raw history | 9.42906 | 10.59785 |
| Numeric-neutral support | 9.23228 | 10.60351 |

Common patient bootstrap, 20,000 draws, fixed seed 20260912:
full-own: source gain0/harm3, hospital gain0/harm0.
Full-support: source gain0/harm2, hospital gain0/harm0.
Full-persistence: source gain7/harm0, hospital gain4/harm0.
Own-persistence: source gain7/harm0, hospital gain5/harm0.

The same lack of incremental benefit occurs without recurrent JEPA. Therefore
recurrent interference is not the only demonstrated location of the failure.
This is not evidence that other organs have no useful information: fixed-lag
compression, limited cohort, measurement noise, and task definition remain.
No model has been selected using hospital results. No calibration or full
promotion gate was performed. Runtime remains unchanged.

## Evidence

`artifacts/native_map_raw_history_20260912.json` contains aggregate comparisons,
source selection results and input/code hashes. Private predictions are not
committed. The script now accepts an explicit target; historical glucose
report hashes identify its earlier code and are not retroactively updated.

Next diagnosis must target information representation/task support rather than
assume changing recurrent update gates alone will solve the generalization gap.
