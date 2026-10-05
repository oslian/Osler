# Raw magnesium unit provenance

All20906cached magnesium values match raw eICU lab values exactly. Raw system
unit is mg/dL throughout; interface unit agrees after case normalization for
20893rows. Thirteen rows instead say MEQ/L at the interface. Their recorded
values range1.5-1.9, not the extreme9.6 observation.
These conflict IDs touch1eligible future pair and1current pair (current match
uses stay/draw time/availability; current ID absent in cache). Native historical
exposure and accuracy effects not yet quantified. Do not infer zero impact.

This is real unit-metadata ambiguity, not proof numeric values are misconverted.
Do not guess conversion or overwrite hash-bound caches. It does not explain
extreme high values by simple conflicting-interface identity. Raw specimen or
chart correctness cannot be certified from matching database records alone.
No rows removed, training/runtime/promotion unchanged.
Artifacts: magnesium_raw_units_20260911.json,magnesium_unit_conflict_20260911.json.
