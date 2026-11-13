# MAP Native-Missing Numerical Control

Seven matched support-only HGB fits completed. Same feature width, missing
routing, masks, ages, own MAP values, source-derived support, patient weights,
partitions and estimator settings as the native-missing full experiment.
Non-MAP observed numbers are zeroed; missing numbers remain NaN. Two focused
tests verify own values, masks/ages, legitimate zero and missingness preservation.
All parent artifact bindings and held-out MAEs reproduce.

| Median patient-equal MAP MAE | New source | Held-out hospitals |
|---|---:|---:|
| Own | 8.71540 | 10.21838 |
| Full | 8.71981 | 10.31455 |
| Support-only | 8.72592 | 10.31205 |

Full-support: 0/7 significant gains or harms in both roles. Support-own:
0/7 gains both roles, harms 0/7 source and 1/7 hospital. Common patient-paired
20,000-draw 95% intervals; overlapping splits are not independent replications.

Native missing handling does not establish incremental numerical physiology
beyond observation support. The previous partial full improvement cannot be
assigned to cross-organ values from its comparison with the original model.
Nonsignificance is not equivalence and does not prove no raw physiological
information exists. This completes the missing comparator, not a new model
repair or full anatomical JEPA experiment.

No source selection, new external lockbox, interval calibration, runtime changes
or promotion. Private weights/predictions remain outside Git. Evidence:
`artifacts/map_missing_support_20260912.json`.
