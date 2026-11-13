# Raw Lab Event Identity Trial

Bounded glucose@6h experiment, not a whole-body raw-event rebuild. Keep the
frozen JEPA population anchor and the previous source patient/hospital roles.
New residual adapters receive only raw glucose lab history. This is a matched
new-contract comparison, not an isolated patch of the previous 308-input GRU.

Native labid/labevent_id identifies events within source/stay. Exact duplicate
versions (same identity, available timestamp and payload) are removed. Conflicting
payloads for one version fail closed. Different IDs with equal values remain.
Revised versions at later times remain distinct updates. No content-hash/value
deduplication. Times are conservative max(measured, recorded): eICU uses revised
entry offset, not an asserted original-result availability; MIMIC uses storetime.
Unknown availability or units are excluded. Values use existing glucose bounds
10-2000 mg/dL. Raw rows/caches remain outside Git.

Use all available raw glucose events, including between unscored anchors. Queries
do not create physiology events. Compare equal-architecture/capacity adapters:
- once: each raw event version updates state once; query only reads state;
- repeat_at_query: same raw events, plus re-consumption of the last event AFTER
  every query (documentation-density control).
Same source normalization, patient partition, seed, optimization and zero option.
No extra future-error task in this bounded adapter; it is measurement assimilation.

Primary evaluation retains old scored anchors only where a raw value matching
cached current was actually available within six hours and future value/time
matches a native lab event. Personalization additionally requires a prior matched
forecast outcome to have become available. Same scoring rows across all arms.
Frozen whole-body baseline inputs are inherited and not fully storetime-verified;
do not claim deployment safety or a fully clean whole-body availability contract.

Seven fixed seeds; source-only fit/tune; compare patient-equal MAE and paired
bootstrap versus repeat control, frozen JEPA and persistence. This is a point
screen, not promotion. No new conformal/care-unit/time gates or unseen lockbox.

Schema sources: https://eicu.mit.edu/eicutables/lab/ and the installed MIMIC-IV
labevents header. eICU revised offset documents when the revised value was entered;
it does not establish the time an earlier, unavailable version first appeared.
