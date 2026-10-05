# Checkpoint selection scope

All three recent native training runners select by glucose source-tune MAE
only. They do not log actual other-target source-tune losses at each epoch.
Thus retained checkpoints cannot reveal whether a different epoch would have
improved the whole-body tradeoff. This is a missing diagnostic, not proof an
unseen good checkpoint exists.

Saved initial/final outputs allow a descriptive source-tune check. Require
30paired patients in fit and tune per target, same targets within each seed;
17-20targets supported. Average patient-equal MAE divided by inherited positive
target scale, then equally average targets. This is not a clinical utility or
promotion metric; differing target sets/scales limit across-seed interpretation.

| Candidate | Median glucose MAE | Median target-equal scaled MAE | Joint worse than initial /7 |
|---|---:|---:|---:|
| Initial | 35.8737 | .9113 | 0 |
| Glucose-only | 33.7316 | .9605 | 5 |
| Uniform retention | 33.6905 | .8999 | 1 |
| Selected retention | 33.9204 | .9396 | 4 |

This confirms that better primary selection need not improve aggregate
other-output quality even on source-tune. Uniform retention's aggregate benefit
does not erase individual BUN/creatinine harm established separately. No single
aggregate should replace per-target safeguards. No claim of significance from
these descriptive medians, new holdout, or optimal checkpoint selection.

Before another allocation sweep, a matched replay should record per-target
source-tune errors each epoch and compare prespecified primary-only selection
against a primary-noninferior, multi-target criterion. Same trajectory avoids
confounding selection with altered training. If no feasible epoch exists,
selection cannot repair training; if it exists, held-out evaluation still decides.
No calibration/runtime/weight changes or promotion.

Artifact: `artifacts/native_selection_scope_20260911.json`.
Runner: `python -m scripts.audit_native_selection_scope`.
