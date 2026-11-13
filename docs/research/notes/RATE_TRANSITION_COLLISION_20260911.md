# Verified rate-summary collision

Synthetic known events at minutes60,120,180,300 and query360:
rates[0,2,0,2] versus[0,0,2,2]. Existing snapshot AND range/change/span vectors
are exactly identical, despite three changes/two reversals versus one change.
Latest observed change occurred one versus three hours before query.
Thus the tested summary cannot distinguish these histories; this is a proven
representation loss, not proof the histories have different potassium outcomes.

Added rate_transition_features with observed change count, reversal count,
last-change age and signed last change. Same-label, same-unit, past-only,
conflicting-time fail-closed; duplicated records do not add changes. No dose
integration or assumption that undocumented rates continued. No changes gives
zero count but unknown last-change age, not a fabricated event at the anchor.

Regression test demonstrates old collision and new distinction, preserving
duplicate/future invariance. Not yet trained or evaluated on real outcomes.
Existing helper remains the frozen experimental comparator, not a serving
compatibility layer. Runtime, release weights and intervals unchanged.
