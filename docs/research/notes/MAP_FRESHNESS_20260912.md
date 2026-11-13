# MAP Freshness Diagnostic

Saved native MAP predictions, no fitting or threshold tuning. Split at one
hour current measurement age, using only pre-anchor information. Same original
patient roles and full-own paired20k bootstrap. Patients may appear in both
strata at different anchors; groups are not independent patient cohorts.

3,925 pairs: median current age1h; future measurement median6h after query.
Thus current-to-future measured gap median7h, range5-12.5h. The model receives
age; these facts are task heterogeneity, not proof of an unrepresented clock.
2,782 rows age<=1h,1,143 age>1h.

Full-own source evaluation: age<=1h gain0/7,harm4/7; older gain0/7,harm3/7.
Hospital: gain0/7,harm0/7 both. Hospital patient support76/69 with overlap.
The failure is not limited to stale current observations. This does not test
retraining on fresh-only data, all time representations, or eliminate residual
noise; no data are excluded from deployment based on this result.

No runtime/calibration/promotion. Evidence:
`artifacts/map_freshness_20260912.json`, `scripts/audit_map_freshness.py`.
