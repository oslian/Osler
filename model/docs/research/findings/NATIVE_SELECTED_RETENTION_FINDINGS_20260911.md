# Selected retention accuracy

Seven fixed32epoch whole-model fits completed using the frozen source-only
selection in `NATIVE_RETENTION_SELECTION_20260911.md`. Bicarbonate/K selected
all seeds, Mg six. No posthoc hospital choice. Initial predictions reproduced
before training.82bindings verified. Aggregate:
`artifacts/native_selected_retention_20260911.json`.

| Hospital MAE | Glucose-only | Uniform retention | Selected retention |
|---|---:|---:|---:|
| Glucose | 42.4007 | 42.2208 | 42.2518 |
| Bicarbonate | 2.2994 | 2.2522 | 2.2437 |
| Magnesium | 0.3168 | 0.2073 | 0.2083 |
| Potassium | 0.4765 | 0.4364 | 0.4368 |
| BUN | 5.2409 | 7.1220 | 6.4292 |
| Creatinine | 0.3092 | 0.3962 | 0.3295 |
| Calcium | 0.6294 | 0.5018 | 0.5796 |
| Phosphate | 1.5838 | 0.9821 | 1.2055 |

Selected versus uniform hospital: BUN/creatinine gain7/7, calcium harm7/7,
phosphate harm6/7. Against original glucose-only, BUN still harms6/7,
creatinine harms2/7, while Mg gains7/7 and K5/7. Primary glucose versus
original gain1/7hospital but source harm1/7; whole-vs-own source0gain/3harm,
hospital1gain/1harm. No stable primary or all-target repair.

This resolves part of indiscriminate retention's specific cost, not all negative
transfer. Unprotected targets can regress again; reference-quality selection
cannot itself provide their missing training supervision. Other output heads
are trainable in the loaded model, not globally frozen by requires_grad flags.

Bicarbonate beats persistence7/7source/hospital, but initial already performed
well; this is preservation rather than a new promoted ability. All counts are
exploratory patient paired95% CIs on overlapping seeds, not independent tests.
Actual labels were evaluated for every supported target, not just protected
ones. Data/support/pretraining/lockbox limitations remain as in the parent
native protocol. No runtime, calibration, causal or promotion changes.
