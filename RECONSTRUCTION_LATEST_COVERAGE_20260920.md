# Latest-Variable Coverage Versus Repeat History

## Question And Fixed Comparison

Does the prior last1 ablation lose temporal memory, or merely remove the only
available observation of some independent variables? Fixed bounded-site epoch7,
original90 selected and90 broad source-selection patients,33 trained targets,
2470/2025 accepted patient-target pairs. No new patients or fitting. Six other
targets remain untrained in this candidate, not implicitly passed or discarded.

Mask target and all siblings over the entire stream first. Compare full history,
last1 event, the union of the latest event for each remaining observed variable,
and that union plus the last8 events. Keep whole simultaneous event groups,
original values/masks/ages, absolute times and anchor. Repeated equal-valued
measurements are distinct events; no value-based deduplication is introduced.

Both full and last1 results reproduce the previous report within1e-7, all33
targets and both groups. Source/checkpoint bindings checked. Seven focused tests
pass, including latest remeasurement selection, absent-variable behavior,
timestamp preservation, suffix and bootstrap tests.

## Results

Target-equal normalized current-reconstruction MAE:

| Source group | Full | Last1 | Latest per variable | Last8 plus latest |
|---|---:|---:|---:|---:|
| Selected |1.797984|1.844191|1.810225|1.805377|
| Broad |1.787841|1.812108|1.781377|1.782607|

Mean events, averaged equally across targets: selected70.60full ->6.23latest,
broad37.79full ->5.13latest. Latest-per-variable recovers much of the aggregate
last1 loss. Last1 deterioration is therefore not clean proof that repeated
long-term history is the beneficial component; observation coverage matters.

However, preserving latest coverage does not yield a universal repair:
latest-per-variable harms20/33selected and14/33broad numerically. Nominal
unadjusted95% paired bootstrap harm intervals exclude zero for8selected and
3broad targets; gain intervals exclude zero for2targets in each group.
These are descriptive post-selection results, not corrected promotion evidence.

| Target | Group | Latest minus full MAE | Descriptive95% interval |
|---|---|---:|---|
| Creatinine | Selected90 |-.003445|[-.012983,.007161]|
| Creatinine | Broad83 |-.002756|[-.010175,.004556]|
| BUN | Selected90 |+.052073|[+.000476,+.105321]|
| BUN | Broad83 |-.060684|[-.101102,-.020621]|
| Glucose | Selected90 |+.013985|[-.007270,+.034987]|
| Glucose | Broad84 |-.009361|[-.033990,+.013939]|

BUN is the clearest exploratory group-dependent reversal. Creatinine does not
show reliable benefit from this deletion. Neither supports an oracle runtime
rule derived from evaluation outcomes or a universal history cutoff.

## Interpretation And Limits

The prior test confounded missing variable coverage with loss of repeat history.
This additional control separates those factors partially, not perfectly:
retaining whole events also retains co-measured variables; deleting events
changes the initialization time, transition gaps and input distribution.
It is not a causal physiological intervention or matched retraining experiment.

The specific remaining hypothesis is heterogeneous usefulness of older repeated
observations, not a disconnected state pathway or complete lack of memory.
This has not established why that usefulness differs (case mix, acquisition
process and noisy small source groups remain possible). Any history-retention
repair requires pre-anchor selection learned only on fitting patients, with
separate source selection and frozen confirmation. Do not choose history length
by group or target from this report. A larger fixed-weight source replication
of the BUN reversal is more informative than immediately adding a learned gate.

No release weights, registry, intervals, serving behavior or promotion changed.
This is masked current reconstruction, not future forecasting. Patient rows
stay private. Evidence: artifacts/native_reconstruction_latest_coverage_20260920.json.
