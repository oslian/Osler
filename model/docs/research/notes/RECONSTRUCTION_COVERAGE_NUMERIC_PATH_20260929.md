# Matched Numeric Path Diagnostic

This repeats the prior neutralization method on BOTH frozen coverage and
replacement models, on identical identity-hash-selected source queries.
47targets x5folds, at most128queries per cell,29939query pairs total. It is a
diagnostic sample, not full-cohort or outer generalization evidence. Same patients
and fitted weights recur across some folds;235cells are not independent trials.

Only the firstN normalized numeric columns are set to zero. Masks, ages, event
times and all remaining native columns are unchanged. Zero represents the
fit-normalization center, NOT physiological zero. Target/sibling masking remains
inside the model. Every unperturbed output reproduced its frozen saved prediction
exactly. No weights or thresholds were fitted.

| Result | Original replacement | Coverage |
| --- | ---: | ---: |
| Cells where neutralization increases mean error | 197/235 | 207/235 |
| Median normalized MAE increase | 0.040595 | 0.048253 |

The numeric path is active and often useful on these source queries. This weakens
the blanket hypothesis that the network ignores all values. It does not establish
more transferable physiology: neutralization can be out-of-distribution, and
larger sensitivity can also mean fragility. The paired differences have not been
given a new significance or promotion test. The prior complete outer comparison
still controls capability claims: no new joint base-plus-incremental screen pass.

Next work must distinguish helpful sensitivity from reliance on source-specific
numeric relationships, ideally with matched in-support perturbations and
source-held-component verification. Do not restore global value removal, tune
to outer failures, or mistake this for proof of causality.

Reproduce: `python -m scripts.audit_coverage_numeric_path`.
Aggregate: `artifacts/native_reconstruction_coverage_numeric_path_20260929.json`.
No patient rows exported; no serving/calibration/registry changes.
