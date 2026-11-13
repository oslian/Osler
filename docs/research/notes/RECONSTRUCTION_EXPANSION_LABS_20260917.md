# Expansion Laboratory Support Preflight

Scanned original eICU labs only for additional patients whose complete raw stay
histories remain in existing fit/encoder-selection/readout-confirmation roles.
Current cohort and current ancestral patients/sites excluded. Evaluation and
reserved outcomes were not returned or used for target selection.

Reused existing lab-name mapping, canonical-unit whitelist and plausible bounds.
Required finite numeric result, measured/revised offsets, and conservative
availability delay greatest(measured,revised)-measured between0and60minutes.
This reproduces the existing timestamp convention, not proof of true first
availability. Counted unique patients per hospital and distinct lab IDs.

31of39catalog targets have >=3confirmation hospitals with >=30screened patients.

| Target | New confirmation patients per site, ascending | Sites >=30 |
|---|---|---:|
| BUN | 26,243,289,445,650,697,968 | 6 |
| Bicarbonate | 26,247,292,354,446,661,975 | 6 |
| Phosphate | 5,99,161,284,287,532,602 | 6 |

These patient counts are per site, not sums of distinct people across sites.
Amylase/fibrinogen/LDH each have only2sites>=30. Heart rate/MAP/O2sat/respiratory
rate/temperature are bedside-only here and explicitly marked unmapped_lab,
NOT failed predictions or proof of absent measurements. All39 retained.

## What this establishes

Original local labs contain much more confirmation support than the current
89-patient head-selection sample. It is worth constructing additional qualified
source examples rather than interpreting the tiny site's sign as a stable effect.

## Still not established

This is a screening upper bound, not a trained cohort: raw ID/version consistency,
simultaneous conflicting values, complete target/sibling masks, post-mask history,
actual per-patient sample acceptance and prior project exposure still need checks.
No model has been refit, no performance improvement demonstrated. Existing source
roles remain fixed; bedside inventory and full event reconstruction remain pending.

Evidence: `artifacts/native_reconstruction_expansion_labs_20260917.json`.
Input hashes verified and source-role patient counts reproduce preceding inventory.
No runtime/interval/promotion changes.
