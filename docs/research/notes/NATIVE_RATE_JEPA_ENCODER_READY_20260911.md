# Native encoder input integration

The existing NeuralBeliefFilter now has a candidate subclass that also activates
steps with an observed rate, while explicit lab masks and padding remain intact.
It is the same single recurrence, not a parallel state or new prediction head.
Ragged histories are padded without truncation; current query physiology is
copied exactly from the original cache, not from sparse last-event tokens.

Seven focused tests pass: native rate-only updates, padding invariance,
new-event/zero/stale/conflict semantics, scheduling and event projection.
Real-cache forward smoke check:5,526query mappings, maximum165history steps,
178input features, finite outputs on eight queries. This is not an accuracy
audit or a claim that eight outputs validate the entire cohort.

Native neural training remains pending. Preserve source-only event scaling,
existing patient roles, same teacher/target, and numeric-versus-support control.
Native treatment timestamps remain retrospective records. No runtime,
calibration, promotion or causal claim changed.
