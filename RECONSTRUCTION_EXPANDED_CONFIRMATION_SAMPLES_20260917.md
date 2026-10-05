# Expanded Confirmation Reconstruction Samples

Completed all3433additional patient histories. Fixed seed20260917 sampled one
raw-fresh availability anchor per patient-target without outcome-based selection.
Existing conflict resolution,48h actual-time history, target/all-history/sibling
masking and independent-observation requirement were reused. No rejected anchor
was redrawn. Every accepted history ends no later than its anchor.

88728patient-target anchors proposed;88232accepted;496rejected without replacement.
All39catalog targets have samples.36have>=3selected-anchor hospitals each with
>=30patients; amylase/fibrinogen have2, LDH1. This support count is not promotion.

| Target | Accepted patients | Hospitals >=30 |
|---|---:|---:|
| BUN | 3318 | 6 |
| Creatinine | 3316 | 6 |
| Bicarbonate | 2947 | 6 |
| Phosphate | 1968 | 6 |
| Heart rate | 3394 | 7 |
| MAP | 3209 | 6 |
| Lipase | 525 | 6 |
| Triglycerides | 502 | 4 |
| TSH | 310 | 3 |

The build took approximately42minutes and completed uninterrupted. Nine existing
reconstruction/schedule regression tests passed. Private sample shards carry
patient IDs and selected-site provenance; aggregate report binds index/code/cache
hashes. No row-level data added to version control.

This supplies a larger source-confirmation dataset, not demonstrated accuracy
improvement. No encoder/head has been fit to these labels or scored here. Existing
nominees should remain fixed for the next confirmation comparison; do not retune
on this group and then claim independent confirmation. Original fit eligibility
still limits which of the39targets have trained candidate heads. Missing trained
heads must remain explicit, not silently invented or removed from reporting.

Report: `artifacts/native_reconstruction_expanded_confirmation_samples_20260917.json`.
No runtime, interval or promotion changes.
