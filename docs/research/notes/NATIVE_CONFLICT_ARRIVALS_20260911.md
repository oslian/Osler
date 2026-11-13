# Later-arriving disagreement risk

Raw-cache audit resolves each source's known version at every arrival time.
Among2119 as-of conflicting variable/measurement-time groups,1602 have a later
arrival containing one value while the accumulated known sources disagree.
1551 of those occur within6h measurement age. Examples by variable count:
bicarbonate776,glucose343,creatinine36,potassium40,Mg8.

`encode_group` only sees one arrival-time group. Its within-group conflict
check cannot detect an earlier different-source value arriving at another time.
These counts are raw candidate-risk groups, NOT exact training-query exposure
or model error counts. Other same-arrival newer draws, windows, and masks may
exclude some. Native schedule exposure has not yet been reconstructed.

The first report can be legally used before a conflicting report exists; it
must not be retrospectively suppressed using future knowledge. A complete
online repair needs a conflict/invalidation update or as-of replay, not just
dropping all records later found conflicted. That behavior is not implemented
by the snapshot/label/summary changes alone. No native weights or runtime
changed; no prediction improvement or root-cause conclusion.

Artifact: `artifacts/native_conflict_arrivals_20260911.json`.
