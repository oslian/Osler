# Positive control detects a strong artificial increment

Fourteen fixed tree fits using frozen expanded patient partitions. Artificial
independent Rademacher input, additive +/-0.5 to real potassium future labels.
Equal-capacity nuisance input comparator; patient-equal weights and bootstrap.
Two-column current/age baseline, not the full-history or JEPA pipeline.

Signal vs nuisance median MAE source .446539/.630912; hospital .377904/.586152.
Significant improvement7/7 each. This confirms gross ability to learn and score
a strong simple increment. It does NOT establish sensitivity to small effects,
complex heterogeneous interactions, or treatment causality. Artificial labels
must never count as clinical validation; no models deployed or promoted.

Together with real own-history successes, this argues against a wholly broken
optimizer or universally impossible evaluation gate. It does not resolve the
remaining real conditional-information problem. Do not call it full pipeline
positive validation: full history and shared JEPA were deliberately not tested.

Artifact: `artifacts/increment_sensitivity_20260911.json`.
