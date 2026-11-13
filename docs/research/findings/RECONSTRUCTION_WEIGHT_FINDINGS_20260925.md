# Correction Training Weight Concentration

The implemented correction weight is 1/(component size), then sum-normalized.
This intentionally gives every component equal total influence. It is not a
coding error. The three-components-with-30-people support check does not exclude
additional tiny components from fitting. Their patients can have large weights.

All231nonempty target-fold calibration roles checked; patient-target rows unique.
223contain a component with fewer than30people. Median Kish weight ESS/N is
0.017181. This is a weight concentration statistic, NOT evidence that98.3% of
physiological information was lost, nor a model variance/sample-power estimate.

Examples: chloride fold1 has14120people but Kish ESS57.28, maximum/minimum
patient weight ratio11908. MAPfold1 has13248people, ESS97.81, ratio5630.
The small components carry25%/37.5%of total fitting weight respectively.
These are full correction-calibration roles, not encoder training weights or
each leave-component-out refit. Do not generalize the claim to all JEPA training.

This gives a concrete testable design contributor to noisy residual fits.
It does not establish causality. The next fixed source-only contrast should
change correction training weights alone (component-balanced vs patient-equal),
retain the identical held-out component scoring and all patients, and recompute
the weighted scaler consistently. No downstream gate or outer selection change.
Patient-equal weighting may improve precision but sacrifice small-domain
performance; both scoring levels and component harms must be reported.

Three tests pass. No trained model/runtime/calibration interval changed.
Artifact: artifacts/native_reconstruction_weight_concentration_20260925.json.
