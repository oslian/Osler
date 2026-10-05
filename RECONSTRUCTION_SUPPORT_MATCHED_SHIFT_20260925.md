# Support-Stratified Residual Diagnosis: Limited Overlap

All235target-folds assessed using source-only prediction tertiles crossed with
source median splits of fresh1hvariable count, median latest measurement age
and history span (up to24strata). No target labels define strata. Require each
matched stratum to have>=30people in>=3components in EACH role. Standardize
using minimum common component-weighted stratum mass; retain unmatched counts.
This uses both covariate distributions, not a deployable source-only predictor.

136target-folds have at least one supported matched stratum. ZERO retain>=80%
of patients in both roles; best minimum retention78.97%(plateletsfold4).
The80%summary is a descriptive coverage check, not a promotion threshold.
Therefore this coarse exact-matching attempt does not settle whole-population
causes. No looser split/support search or deletion to claim success.

Examples (source/outer retained fractions):
- Albuminfold2:12.96%/24.24%,1supportedstratum; mean residual shift+.06410.
- O2satfold3:19.37%/19.79%,2strata; matched+.82875vs same-retained unmatched
  +1.11713. Cannot interpret this selected subset as whole-population repair.
- MAPfold4:44.08%/45.77%,3strata; matched+9.27086vs unmatched+9.82653.
- HRfold0:25.68%/37.33%,4strata; signed shift-.89388but average absolute
  bin shift3.33352, showing cancellation can conceal heterogeneous errors.

Support matching here is not matching variable identities, numeric physiology,
treatment context or clinical diagnosis. Nonzero residuals in retained bins do
not establish a clinical cause. Low retention is a limitation of this specific
multi-axis stratification and per-domain support, not proof the raw cohort is
generally too small. Do not revise scientific conclusions from selected bins.

Three tests verify labels cannot set cutoffs, uniform residual shift is not
falsely removed, and no-overlap remains explicit. All input hashes and exact
history joins verified. No fitting, runtime, promotion or forecast changes.
Artifact: `artifacts/native_reconstruction_support_matched_shift_20260925.json`.

Next analysis must explicitly address overlap without repeatedly choosing
strata from outer errors, and should inspect variable-specific observation
identity/availability rather than assuming these aggregate metrics identify
the patient's state. A new development-only protocol is needed before a repair.
