# Kernel Scale Repair: Real Improvement, Not a Complete Model Repair

## What Changed

Weighted StandardScaler fitted on the SAME fitting patients, before the fixed
RBF128-landmark map and absolute-error SVR. Frozen encoder, target/sibling masks,
true labels, patient/component roles, pair draws, landmark selection, regularizer
and shuffled-body control unchanged. All47 targets/all5 folds/four arms completed:
940 fits,1,797,884 selection occurrences,720,470 unique patient-target pairs,
29,580 people. All940 solvers converged. No external rows or runtime changes.

This follows the located raw metric failure, not another architecture search.
The efficacy family is prespecified2*47*7, testing scaled body and scaled raw
separately against strong original HGB/native/population and matched controls.
No query-selected setting, clipping, favorable-subset release or relaxed gate.

## The Located Defect Was Partially Repaired

Median across235 target-fold medians of fit-pair squared-distance fractions:

| Contribution | Original raw kernel | Fit-scaled raw kernel |
|---|---:|---:|
| Numeric latest+mean |0.4120%|29.2293%|
| Availability lag |99.3419%|22.3075%|
| Combinations with lag median above90% |235/235|0/235|

The mean kernel effect of removing numbers at unchanged respective bandwidth,
median across target-folds, grows0.008963->0.133865. These are metric diagnostics,
not percentages of physiological information. Other blocks still encode medical
observation processes; standardizing their scale does not make them physiology.

Scaled raw improves fitting MAE over old raw kernel in230/235 combinations.
Patient/component target-macro selection MAE falls2.525%/2.919%. Under the
prespecified corrected both-level test,10targets beat old raw kernel: albumin,
BUN, calcium, heart_rate, hematocrit, INR, phosphate, platelets, PTT, total_protein.
Raw numeric signal is no longer almost excluded by hour units. Thus the defect
had predictive consequences, not merely unattractive distance statistics.

## What Still Fails

Normalized target-macro absolute error, NOT physical units:

| Model | Patient equal | Component equal |
|---|---:|---:|
| Existing raw HGB |0.825737|0.787947|
| Existing body HGB |0.857139|0.811241|
| Native neural |0.882389|0.831593|
| Old body kernel |0.884742|0.834098|
| Scaled body kernel |0.886598|0.835116|
| Old raw kernel |0.937564|0.889135|
| Scaled raw kernel |0.913889|0.863185|
| Scaled process kernel |0.937631|0.886365|
| Scaled shuffled body |0.944165|0.894135|

Complete source diagnostic screens: body0/47, raw0/47. Neither candidate has a
corrected both-level win over native, body HGB or raw HGB. Scaled body only beats
its old kernel for calcium. Scaled raw has worse patient means than raw HGB for
ALL47targets (42component means); scaled body loses body HGB for35/38means.
Therefore this is NOT simply insufficient power or a strict gate hiding an
otherwise superior model. Existing stronger comparators still predict better.

Examples of patient-equal MAE:

| Target | Old raw kernel | Scaled raw kernel | Existing raw HGB |
|---|---:|---:|---:|
| BUN |0.700312|0.663555|0.578286|
| Creatinine |0.965719|0.940074|0.820478|
| Hemoglobin |0.535437|0.488919|0.425513|
| Glucose |0.722259|0.718701|0.689013|
| MAP |0.573423|0.566067|0.550190|

Each scaled body/raw/process kernel still loses its corresponding HGB FITTING
MAE in232/235 comparisons. A substantial gap exists before transfer evaluation.
Finite map/bandwidth/regularization and representation readability remain live
explanations; no unique mechanism established. Raw scale correction does NOT
fix the body latent's failures, and does NOT prove irreversible encoder loss.

## Numerical and Support Checks

19focused tests pass. Fit-only unit-rescaling, unchanged landmark indices,
query isolation, constant-feature behavior, patient equalization and duplicate
guards tested. Original query/reference parity and allfold metrics reproduce.
Post-standardization simultaneous rotation of input and landmarks changes
features at most1.05e-9 and relative predictions7.60e-11. This is NOT invariance
under rotating ORIGINAL inputs before coordinate-dependent scaling.

During StandardScaler fitting,268/705 distinct maps emit/contain tiny negative
roundoff variances (2,060 coordinates; lowest -9.202e-34). Saved-map audit confirms
they are below constant-feature numerical bounds, every affected scale is one,
and all scales/output arrays are finite. sklearn computes sqrt before replacing
near-constant scales, causing its warning. No fitted output was changed or
warning silently treated as proof of valid data. Original `constant_features`
diagnostics count EXACT zero variances, not all near-constant coordinates; the
separate numerical audit records the latter correctly.

Another limitation surfaced: raw query standardized absolute values reach3566.7
(median per-combination maximum121.6), and21/235 combinations have query variation
in exactly constant fit columns. This is a support/scale warning, not a proved
cause of all error. No query-derived clipping or refitting was used. Body's
maximum is55.84 and has zero such exact-constant variation combinations.

## Decision

Keep all new heads diagnostic/candidate-only. No promotion, deployed weights,
registry or interval changes. No forecasts, clinical or causal conclusions.
Both source selection and old outside datasets are already inspected; this is
not new external validation. Full requested generalization remains unresolved.

The next discriminating question is why SAME-source fit remains poor despite
finite converged optimization: examine effective kernel feature spectrum,
prediction dispersion and regularized fit objective before another parameter
sweep. This diagnostic has not yet been executed. Do not repeat basic raw unit
scaling as an untested universal fix, or declare lack of information from a
weak readout's failure. Any later efficacy experiment needs fixed source-only
choices and unchanged strong comparators.

Evidence: RECONSTRUCTION_SCALED_KERNEL_PROTOCOL_20261003.md;
artifacts/native_reconstruction_scaled_kernel_{fold0..4,assessment,numerics}_20261003.json;
scripts/train_scaled_kernel.py, scripts/score_scaled_kernel.py,
scripts/audit_scaled_kernel_numerics.py. Private rows/heads remain ignored.
