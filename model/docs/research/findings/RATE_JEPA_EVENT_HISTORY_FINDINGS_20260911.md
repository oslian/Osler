# Separate recorded-treatment history for shared JEPA

Built event context at every existing JEPA history snapshot, not just the
forecast query. Exact row-to-stay/time mapping verified for all46,657 snapshots;
overlapping histories agree on identity. Existing5,526 queries and labels are
unchanged. Raw patient-level context remains private.

Three separate channels: log recorded count, record age, documented insulin
rate in units/hour. Existing6h past-only context helper reused, with equal-time
conflicting rates unknown.30,993 history snapshots support a rate, including
1,939 true zero snapshots.5,525/5,526 query snapshots support a rate.
Missing rate is not zero or no treatment. No dose integration or assumed
continuation beyond records. Infusionoffset is not independent availability
evidence; this remains a retrospective record-time candidate.

This fills the temporal alignment prerequisite for treatment input to the
same history encoder. It does not yet train or validate that model, and it is
not the old tree snapshot trial. No treatment fields occupy laboratory slots.
Next paired training must distinguish full numerical treatment from identical
support-only treatment, using unchanged patient roles and source-only scaling.

Artifact `artifacts/rate_jepa_event_history_20260911.json`, private NPZ bound
to body arrays/queries/labels. No runtime, calibration, training or promotion.
