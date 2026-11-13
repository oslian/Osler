# Fixed treatment-context diagnostic

Question: does omitted infusion context explain residual error on the existing
6174 potassium pairs? Source only, unchanged seven patient partitions and labels.
No external data, runtime changes, model selection or promotion.

Freeze the original event-window own/full tree capacities and hyperparameters.
Reproduce both frozen baselines before interpretation. Refit each with actual
treatment context and with capacity-matched shuffled-patient context (28 fits).
Shuffle donors inside each fit/tune/evaluation role; donor relative-time query
must not exceed recipient relative time. Preserve each recipient's lab history.

Features per potassium-related action: count in (anchor-6h, anchor], last-event
age, and latest documented rate per canonical dimension. Unknown stays NaN;
zero remains zero. Solution volume is separate, never inferred drug dose.
Snapshots are not assumed continuous or active beyond their documented event.
Event-time only: infusionoffset does not prove prospective availability.

Same patient-equal weights, 200 absolute-error histogram boosting iterations,
source-selected leaf count, min leaf30, L2=10, bins63, learning rate .05.
Compare patient-equal MAE and paired patient bootstrap95% CI (1000 draws)
against frozen baseline, placebo and persistence. All seven source splits are
required for a stable diagnostic gain; overlapping splits and unadjusted CIs
are exploratory, not confirmatory promotion. No threshold tuning on outcomes.
Only aggregate reports enter the repo; fitted objects remain private.
