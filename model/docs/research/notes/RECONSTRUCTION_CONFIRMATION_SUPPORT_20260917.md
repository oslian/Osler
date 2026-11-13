# Confirmation Support: Interpretation Correction

Replayed fixed nominees for all 28 supported targets, reproducing confirmation
patient MAE differences within1e-10. No heads, rules or patients changed.

The prior equal-site sign reversal is descriptive, NOT established cross-site
failure. BUN confirmation component sizes are1,6,9,11,14,16,32. Only one meets
the existing30-patient per-group support convention. Its largest site's delta
is-0.0546, bootstrap interval[-0.2018,0.0926]. The singleton delta+0.387 receives
the same weight as the32-patient component in the equal-site mean. Neither
equal weighting nor patient weighting resolves uncertainty about site effects.

Phosphate component sizes1,5,5,9,9,14,30 and bicarbonate1,6,9,11,11,13,32 also
have only one component with30patients. Differences in selected alpha and mean
signs are observed, but their causal attribution to hospital biology/process
versus sampling noise is not identified.

The machine report contains empirical bootstrap percentiles for transparency.
Singleton percentiles are necessarily a single value and MUST NOT be read as
zero uncertainty or statistically significant evidence. All sites below30 are
explicitly marked insufficient. These are unadjusted descriptive diagnostics.

## Consequence

Splitting179 tuning patients into90encoder/89head patients protects role
independence but leaves weak target-by-site confirmation support. The zero
confirmation outcome is not evidence that all nominees lack useful signal.
It is also not permission to lower gates, drop small hospitals, or promote
based on the already inspected evaluation. Before another cross-site repair,
establish adequate confirmation support or a prespecified patient/site-aware
uncertainty design; quantify this limitation instead of claiming a root cause.

No model accuracy improved, no runtime changed, no promotion.
Evidence: `artifacts/native_reconstruction_confirmation_support_20260917.json`.
