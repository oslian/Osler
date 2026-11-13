# Out-of-sequence measurement arrivals

Native lab events are ordered by available time. Within each simultaneous
variable group, select the most recent measured time and reject conflicting
values, following token construction; only age0-6h included. A delayed older
arrival is measured earlier than an already available draw of the same variable.

V2 restricts both draws to each query's actual48h input window. V1 compared
against whole-stay prior history and slightly overstated affected queries;
use `artifacts/native_out_of_sequence_20260911_v2.json` for current counts.

- 2,392/5,526 query histories contain such an arrival in at least one variable.
- 2,032/5,526 contain an older glucose arrival after a newer glucose draw.
- Across the event inventory, glucose4,613/175,141eligible groups,
  bicarbonate1,196/44,516, potassium152/35,018 are delayed-older groups.

These are support counts before any source-specific channel restriction, not
patient-independent observations. Query windows overlap. No evaluation outcome
or model score is used, and no accuracy consequence has been established.

This is legitimate delayed information, NOT automatically a duplicate or bad
value. The native encoder receives available time and measurement age, so it
can in principle infer measured time. It still applies the same forward update
mechanism for fresh and delayed observations. That is a testable modeling
question, not proof of a bug or that it actually mistakes age for freshness.

Do not reorder by draw time if that would reveal results before availability,
and do not discard late records as a default repair. A distinct trial must
preserve causal arrival, acknowledge delayed-observation semantics and compare
accuracy against the current filter on the same query/label roles.

No weights, extraction semantics, runtime, intervals or promotion changed.
Current finding quantifies an unresolved input regime, not the unique cause.
