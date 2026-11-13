# BUN Signed Residuals And Cohort Contract

Audit exact sample-site metadata against cached signed full predictions.
Input hashes verified; aggregate-only output. Residual=truth-prediction.

Selected source-selection90patients come from4sites, but71(78.9%)are at449;
effective site count1.55. Remaining sites have4,2,13patients. Site449median
residual+.1571, descriptive95%CI[-.4101,.4983]. Thus neither a stable opposite
bias nor broad multi-site calibration behavior is established for this group.

Broad source-selection3086patients cover the same4sites with counts226,1102,
227,1531; effective site count2.60. Residual medians respectively-.3335,-.3209,
-.4929,-.4197; all four descriptive median intervals exclude zero negatively.
The broad overprediction is not solely the largest site's artifact. At site449,
selected and broad medians differ, but selected uncertainty is large. Do not
claim a proven biological interaction or a hospital-only explanation.

Code-contract review (already partly documented20260917):
- Historical selected cohort derives from insulin-rate-change inclusion;
  reconstruction anchors themselves are fresh raw measurement availability
  events, not necessarily insulin-change timestamps. These are distinct stages.
- Nested fitting/encoder-selection roles separate entire hospital components,
  not random patients from one homogeneous population.
- Broad source excludes ancestral/reserved patients and uses those source
  hospital roles without the insulin-rate inclusion requirement.
- Historical expanded reconstruction chooses one candidate per patient-target
  without replacement after rejection. It does not select anchors by model error.

## Consequence

Observed disagreement must not be simplified to a universal architecture bug
or two proven physiological regimes. Broad median calibration drift is supported
across sites; selected validation support is highly concentrated and uncertain.
These facts were not repaired by changing integration resolution or adding a
scalar slope. Do not use hospital identity or retrospective insulin eligibility
as a deployed correction. A prospective repair needs source-only calibration
with independently supported strata and a fixed cross-site evaluation, not
continued outcome tuning on90patients.

Audit exit0; reused median/support helper has two passing tests. No weights,
runtime, intervals, data split or promotion changed. Patient data remain private.
Evidence: artifacts/native_reconstruction_bun_site_residual_20260920.json.
Relevant code: scripts/build_native_reconstruction_expanded.py,
scripts/build_reconstruction_nested_selection.py,
scripts/build_broad_source_events.py,
scripts/audit_reconstruction_cohort_selection.py.
