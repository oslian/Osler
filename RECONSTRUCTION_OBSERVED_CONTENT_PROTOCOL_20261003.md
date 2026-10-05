# Observed Content Before/After Persistent Aggregation

E15 already decoded observed raw values from final body. E17 instead inferred
the masked unknown target from pooled event representations. E18 identified
block contributions for raw HGB. None directly compares the same *observed*
value in its last-observed event encoding against final body encoding.

## Fixed Extraction

Use frozen epoch10 and original payload-checked masked raw/body caches for all
five source folds. Choose at most2048 fit/1024 selection patients by SHA256 salt
`observed-content-20261003:` plus patient identity. For each chosen patient,
choose one available masked-target context by the same hash of patient/target.
Do not use outcomes, prediction errors, measurement values or future data to
select. One context per patient per fold prevents many-mask replication.

Transform original histories with the original fold scaler, mask context target
and siblings exactly as before. Verify raw summaries equal original cache.
For each remaining observed variable v, identify its last valid measurement
event at/before anchor. The v-labelled truth is the actual latest numeric input
in normalized fold units, not the masked context target and not a future value.
Compute the frozen event encoder on those unique event rows. Cache each v's
corresponding64-dimensional event representation and the final original body64.
Append the same159 presence/lag/logcount features to event and body.

Do not invent observed labels for missing variables. Direct raw-value copying
is the zero-error positive reference by construction, not an eligible predictor.
Event selection knows only the observed variable and lawful event history.
Event has less subsequent context; body can contain later lawful observations.
This difference is the intended memory-path diagnostic, not equal full context.

## Fixed Readers And Scoring

For each observed variable/fold fit on observed fitting rows only: event223,
body223 and process159, each with existing HGB PARAMETERS and StandardScaler
plus Ridge(alpha100,cholesky), same bounded component weights. Fit minimum50,
query minimum30, minimum3 query source components, nonconstant fitting truth;
unsupported cells reported, never removed after viewing results. No tuning.
Fit median is an additional descriptive reference. All53 schema variables are
considered; no fixed promise all have enough support.

Six directional contrasts: event-body, event-process, body-process for each
reader family. Average repeated fold losses per patient then by component.
Paired bootstrap5000, t p < .05/(53*6), CI upper<0 at both aggregation levels;
minimum30 distinct patients/3components. Report contributing folds and all
unsupported reasons. Not all supported folds implies full five-fold support.
These are observed-content diagnostic screens, not clinical promotion.

## Interpretation And Stop

Event better with both readers supports lower *accessible* observed-content
fidelity along the aggregation path, not irreversible information loss.
One-reader advantage remains a readout/coordinate ambiguity. Body no worse
does not support this specific loss hypothesis. Both fail against process:
do not conclude there is no physiological information. Single-event encoding,
readout limits and sampling remain possible. Outcome is not missing-variable
inference, future forecasting or treatment causality.

One fixed run; all five folds and schema support accounted for before a final
conclusion. No encoder retraining, runtime updates or parameter sweeps in this
diagnostic. Private histories/IDs/features/models/predictions stay in ignored
data; publish aggregate errors and hashes only. Reused source development,
not an untouched external database. Stop on identity/time/hash mismatch.
