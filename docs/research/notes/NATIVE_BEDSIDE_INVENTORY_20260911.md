# Nursing event expansion inventory

Same1916selected stays, numeric nursing records extracted to private Parquet:
3536392rows. Separate nursingchartoffset and nursingchartentryoffset exist.
Initial DuckDB4thread1GB extraction hit memory limit before output; single
thread2GB with insertion-order preservation disabled completed. Subsequent
event sorting must use explicit timestamps/IDs, not CSV scan order.

| Label | Rows | Stays | Entry >= measurement | Median delay minutes |
|---|---:|---:|---:|---:|
| Heart rate |326587|1916|308366|8|
| O2 saturation |314116|1912|296177|8|
| Respiratory rate |313552|1913|295856|9|
| Noninvasive mean BP |211856|1892|201662|12|
| Invasive mean BP |131774|1182|123760|5|
| Temperature C |128060|1916|121725|12|

These are whole-stay numeric counts, NOT valid scored-anchor support. Numeric
parsing alone accepts values without verified units/ranges; early entry times
must not be silently changed to measurement time. C/F sibling records should
not double-update temperature. Invasive/noninvasive BP provenance must remain
distinct until conflict policy is applied. O2 L/% is ambiguous and must not be
used as a numeric oxygen dose without unit separation.

This provides an existing local source for missing bedside channels. It does
not establish that adding them improves prediction. Next build time-valid,
unit-explicit canonical bedside events, then align to existing anchors and
retrain same-task matched controls. No clinical assessments or complete
treatment intent claimed. No runtime/training/promotion changed.

Artifact: `artifacts/native_bedside_inventory_20260911.json`.
Private numeric nursing cache remains outside Git.
