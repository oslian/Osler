# Frozen patient influence diagnosis

All original MAEs reproduced with saved models, no fitting. Compare numerical
treatment against own history, patient-average absolute-error differences.
Predefined entirely-after-draw group only; no selection by observed outcome.

Hospital146patients: mean gain in7/7seeds, .00450-.01857 MAE; benefited patient
fraction51.4%-59.6%; median patient delta negative7/7. Removing any single
patient cannot reverse mean direction in any seed. Source likewise no
single-patient reversal, but only47.5%-56.5% benefit and one positive median.

Thus the small hospital mean benefit is not solely one influential patient.
This does not rule out small groups/hospital influence, prove all-patient
benefit, or convert overlapping seven seeds into independent replications.
Prior significant improvement2/7 remains unchanged. Effect heterogeneity and
uncertainty remain; no new promotion, intervals or runtime change.

Artifact: `artifacts/rate_anchored_influence_20260911.json`.
