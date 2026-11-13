# Event-window input repair result

Completed21 source-only augmented tree fits on the original6174 potassium pairs.
Patient partitions, leaf counts, current values and labels unchanged. Every
old model source MAE reproduced before comparison. Summaries use only versions
known by the anchor; a test includes an excursion missed by lag endpoints and
a later revision that must not alter earlier features.

## Information loss exists

Comparing event extrema with the corresponding fixed-lag extrema finds values
outside the lag range in4069/82461 supported6h entries and23071/10859124h
entries. Entries are query-variable-windows, not independent patients/events.

Because lag values are float32 and summaries float64, a precision sensitivity
check using numpy.isclose(atol=1e-5,rtol=1e-5) reduces these to3554/82461(4.31%)
and22407/108591(20.63%). Use these conservative percentages, not the small
rounding-only differences. This check changes no features, models or scores.
The event summaries also retain older values within each window, whereas each
lag masks draws older than6h. Thus differences are not solely undersampling.

## Does recovering summaries repair prediction?

Patient-equal evaluation MAE, median across seven fixed source seeds:

| Input | Original lag model | Lag + event summaries |
|---|---:|---:|
| Own | .377185 | .373470 |
| Full | .370529 | .369971 |
| Support | .378698 | .381141 |
| Persistence | .437907 | .437907 |

Augmented versus original significant improvement: own0/7, full1/7, support0/7;
no significant harm. Augmented full beats own3/7, support5/7, persistence7/7.
Own also beats persistence7/7. These are unadjusted descriptive CIs on overlapping
source evaluation splits, not seven independent replications.

Recovering these summaries gives partial evidence beyond observation support,
but does not yield stable improvement over the original full model or own
history. Fixed-lag information loss is real; this experiment does not establish
it as the principal remaining failure cause. It also does not prove an exact
event-sequence encoder could not exploit more information than these summaries.

No external data read, external validation, calibrated interval, runtime or
promotion change. This diagnostic retains the shared-body production direction;
trees are comparators, not an independent replacement body model.

Protocol: `POTASSIUM_EVENT_WINDOWS_PROTOCOL_20260911.md`.
Artifact: `artifacts/potassium_event_windows_20260911.json` (complete=true).
Command: `python -m scripts.audit_potassium_event_windows`.
