# Owner Observation Mixture Scope

Correct anatomical mapping: MAP belongs to blood_vessels, not heart. Direct
owner observation pool contains MAP and lactate only. Native tokens contain
new events, not repeated carry-forward snapshots.

Source pilot counts:64fit patients have1487MAP events,58lactate-only events
across22patients and1simultaneous event.32tune patients have943MAP events and
40lactate-only events across12patients,0simultaneous. No lactate-only event
occurs after the finalMAP in any of these96queries.

Consequently routine final-MAP overwrite by later lactate cannot explain this
pilot. Earlier lactate assimilation may still affect state; this audit does not
test its accuracy effect or prove independence. Pooling semantics may matter
more for other organs with many variables; do not generalize the MAP counts.
No model changes, training, promotion or runtime/calibration updates.

Evidence: `artifacts/native_owner_event_mixture_20260913.json`.
