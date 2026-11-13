# Native new-event token cache

Materialized128,236unique simultaneous event groups into172existing lab/history
coordinates plus separate raw treatment channels.75,552groups contain eligible
new labs,54,198contain an unambiguous recorded rate,52,402are rate-only.
282groups have no usable lab/rate and must remain inactive. Counts overlap.
Same query labels and ragged schedule; cache hash-bound, row-level data private.

Each token marks new labs only, not carried-forward values. For simultaneous
same-variable labs take the most recent measured time; conflicting values at
that time remain unknown. Stale/future-measured labs are not marked observed.
True zero rate remains observed; conflicting simultaneous rates remain unknown.
Six focused contract tests pass across token, schedule and event-input modules.

The52,402native rate-only groups are not comparable to the220masked positions
on the old ten-lag grid: one counts unique native updates, the other repeated
snapshot positions. This exposes a real representation distinction, not proof
that every omitted update would improve prediction. The existing lab-only
step mask must not silently discard these new native rate groups during the
forthcoming matched native encoder trial.

No native neural training yet. Lab input width preserved, event fields separate;
old clipped clock retained to avoid an unannounced time-scale change. Infusion
record time is not verified administration availability. No runtime/promotion.
Aggregate `artifacts/native_rate_jepa_tokens_20260911.json`.
