# Feature-Level Support Versus Total Cohort Size

Feature audit covers53variables x4windows after mandated target/sibling masks.
Of212variable-windows,88 have zero repeat-measurement fit patients,20 have1-4,
16 have5-19,88 have20+. Zero counts include intentionally hidden variables and
unavailable channels; do not label all of them data defects. Twelve windows
have repeat support in heldout roles but none in fit. These are support counts,
not effective sample sizes or evidence that any threshold ensures learnability.

Fixed20-patient rule retains446/848numeric summaries, while all support columns
remain. Source-fit range guard retained. All controls reproduce. Compared with
range-bounded only, supported MAE11.91145->11.90221evaluation and
14.09913->14.09462hospital; both differences NS. Population/placebo contrasts
also remain NS. No stable MAP recovery benefit or promotion.

Thus insufficient per-feature support explains the preceding local extreme
extrapolation vulnerability, but guarding it does not establish useful shared
MAP information. Do not continue threshold sweeps on these same heldouts.
The broader shared-state problem remains open; this is a limited linear probe.

Artifacts: native_body_feature_support_20260913.json and
native_body_supported_range_20260913.json under artifacts/. Script, source
cache and protocol bindings recorded. No neural model/runtime/interval changed.
