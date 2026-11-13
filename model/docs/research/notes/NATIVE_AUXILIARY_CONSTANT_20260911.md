# Constant-change control for native auxiliary model

Fixed trained predictions; compare current plus source-fit patient-equal
weighted median delta. Fit patients only determine offsets, no tune/test fitting.
Same target support and source-evaluation/hospital rows as auxiliary experiment.
This controls typical population change, not own-history predictive capacity.

| Target | Beats constant source /7 | Beats constant hospital /7 |
|---|---:|---:|
| Glucose | 7 | 7 |
| Magnesium | 7 | 7 |
| Potassium | 7 | 7 |
| BUN | 0 | 6 |
| Creatinine | 0 | 0 |
| Calcium | 7 | 0 |
| Phosphate | 3 | 0 |

Creatinine source harm1/7; no other listed significant harms. Hospital
constant MAEs:glucose74.9563,Mg.2384,K.4821,BUN4.1150,creat.2505,Ca.3825,
phosphate.9991. Model hospital MAEs are in the parent auxiliary report.

Mg/K improvement is not explained solely by this constant-delta comparator.
That still does not establish cross-organ dynamics: own current/history may
account for the signal. BUN remains source/hospital heterogeneous; calcium and
phosphate do not establish transported increment over constant drift.
Do not reinterpret a nonsignificant result as equivalence or information absence.
No new training/runtime/calibration/promotion, overlapping seeds exploratory.

Runner: `python -m scripts.audit_native_auxiliary_constant`.
Artifact: `artifacts/native_auxiliary_constant_20260911.json`.
Initial runner indexing error corrected before successful complete output;
no model or source data changed.
