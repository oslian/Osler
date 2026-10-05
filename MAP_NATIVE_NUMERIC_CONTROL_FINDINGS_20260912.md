# Native MAP Numeric-Control Results

Seven matched 32-epoch control fits completed. All 18 direct report
dependencies hash-verified. All input invariants passed across full storage:
MAP numerical history/query, masks, ages, module presence, clock, event
observed flags and consequently activation cadence remain unchanged versus
full. Other normalized numbers are removed in both training and evaluation.

| Seven-split median patient-equal MAE | New source patients | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.44082 | 11.73923 |
| Own MAP history | 8.84140 | 10.21067 |
| Full history | 9.22298 | 10.57510 |
| Numeric control | 9.23789 | 10.61140 |

Control versus own: 0/7 gains and 7/7 significant harms in both roles.
Control versus full: 0/7 gains in both; 1/7 source harm, 0/7 hospital harms.
Control versus persistence: 7/7 gains in both. Significance uses existing
patient-level paired bootstrap intervals, not unpaired medians.

## Interpretation

Removing additional numerical content at matched update cadence is not a
repair. The own/full gap remains without the extra numbers. Thus extra
numerical values are not necessary for a gap in this comparison; blaming
numerical scale/noisy other-organ values alone is insufficient.

Support masks, age patterns, organ presence, query representation and recurrent
update cadence still differ from own. Their individual contributions are not
isolated. A retained-mask zero value is a synthetic center-valued observation,
not an unobserved variable; it is a diagnostic trained control, not deployment
input. These results do not prove cadence alone causes failure, nor that other
organ biology contains no predictive information. They do not prove statistical
equivalence of full and control.

This is a new-task matched-training control, not the previous inference-only
update-scope replay. Seven splits overlap patients and hospital data are reused.
No new lockbox, intervals, clinical capability or promotion. Release untouched.

Evidence: `artifacts/map_native_numeric_control_{seed}_20260912.json` for seeds
7, 19, 31, 43, 59, 71, 83. Private predictions and checkpoints remain local.
