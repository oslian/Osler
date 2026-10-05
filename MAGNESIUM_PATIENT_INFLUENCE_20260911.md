# Patient influence, descriptive only

Verified saved predictions and source-selected shrink factors, no exclusion or
refit. Original paired metric already averages within patient before bootstrap.
Whole improves43.7-50.7%source-evaluation patients and46.6-54.4%hospital patients.
Shrunk blend improves45.7-60.0%source and50.5-57.3%hospital patients.
Deleting a single patient can change the mean-difference sign in3/7source,
1/7hospital whole comparisons;1/7source,2/7hospital blend comparisons.

Small mean effects are fragile in some splits, while benefit/harm is broadly
mixed. This is not evidence of a universally helpful model defeated only by a
single outlier. Majority benefit is not equivalent to lower expected MAE and
does not replace the original gate. No patient was removed, no retrospective
subgroup routing was learned, no new significance threshold introduced.
No accuracy repair, runtime or promotion. Artifact magnesium_patient_influence_20260911.json.
