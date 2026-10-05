# Expected-error routing results

Development decisions frozen on eICU only.
MIMIC is an inspected transfer-validation set, not a new lockbox.

| Target 6h | Selected rule | Persistence MAE | Always expert | Old q25 | New MAE | Move % | All-role gate |
|---|---|---:|---:|---:|---:|---:|---|
| creatinine | mean_benefit | 0.18556 | 0.20270 | 0.18557 | 0.19797 | 72.87 | False |
| bun | mean_benefit | 3.02925 | 3.09671 | 3.02925 | 3.05742 | 63.95 | False |
| urine_output | mean_benefit | 111.91949 | 102.94314 | 110.79825 | 104.69012 | 70.43 | False |
| potassium | mean_benefit | 0.42890 | 0.38448 | 0.42876 | 0.38493 | 97.80 | False |
| bicarbonate | expected_error | 1.91184 | 1.88599 | 1.91147 | 1.88293 | 78.68 | False |

## Interpretation limits

Seven seeds resample evaluation patients; this is not seven independent training runs.
The time check is late-in-stay, not a calendar-time holdout. Care-unit gate is unit-balanced,
not proof that every individual care unit passes. Per-unit coverage is reported separately.
The current release router is not among these fixed two-expert comparators; runtime promotion
also needs that comparison and a new lockbox. All artifacts remain candidate-only.
The shared encoder is frozen and its prior training exposure is not audited here.
The snapshot/bedside availability fixes apply to every arm. Historical dose fields inherited
from older transition extractors still have incomplete event-availability provenance.
Bedside values are still summarized as categories; this experiment changes routing objective,
and does not establish that available clinical information is exhausted.
