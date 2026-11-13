# Derived Gap Pre-Anchor Support Strata

Frozen predictions, identical original pooled MAEs, same20k patient bootstrap.
Groups use ONLY pre-query derived masks: currently available, history-only,
and no history. Rows partition exactly; patients can appear in multiple groups.

On current-supported hospital queries (147patients), derived versus original
full mean error difference median +0.02617mmHg, versus own +0.05933, versus
no-gap +0.03668, versus derived-support +0.02213. Significant gains0/7 for all;
derived-support harms1/7. Thus the complete-case slice does not reveal the
presumed hidden benefit. Nonsignificance does not establish equivalence.

Source current support272-306patients: derived-full gain1/7, derived-own0/7,
derived-no-gap0/7, derived-support2/7. Hospital history-only140patients gives
1/7gain vs full/no-gap,0/7 vs own/support; none129patients gives0/7 vs full/own.
No group establishes stable own-history incremental value.

The pooled failure is not repaired simply by evaluating cases receiving the
derived input. No subgroup-trained routing, future-defined groups, new model,
runtime changes, interval calibration or promotion. The numerical gap-contract
follow-up remains negative within tested support. Complete grouped statistics:
`artifacts/map_derived_gap_support_strata_20260912.json`.
