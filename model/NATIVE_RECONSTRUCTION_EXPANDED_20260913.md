# Expanded Native Reconstruction Cache

Report: `artifacts/native_reconstruction_expanded_20260913.json`.
Builder: `scripts/build_native_reconstruction_expanded.py`.

## Completed

The complete existing seed-7 cohort was processed with the unchanged as-of
conflict-aware history scheduler and target/sibling block masking. Patient roles
are disjoint: 923 fit, 335 tune, 311 evaluation and 194 hospital-heldout.
Accepted patient-target pairs: 25,828 / 9,380 / 8,808 / 5,444 (49,460 total).

Before reconstruction, one raw-fresh availability anchor per patient/variable
was sampled using seed 20260913, without outcome values or prediction errors.
Source-version conflicts or absence of independent masked history can reject
that sample; no redraw occurs. This differs from the previous pilot's reservoir
over all already-eligible examples, so old/new performance cannot isolate only
patient count. Sampling does not depend on model performance.

All 1,763 private shards were verified against their SHA256 index. Every saved
sample has finite values/truth, strictly increasing pre-anchor times, zero
target/sibling numeric and observation-mask channels, and at least one remaining
independent observation per retained event. Pair counts exactly reproduce the
aggregate report. No patient rows or identifiers were added to Git.

## Actual Support

39 variables meet the fixed 30-patient minimum in fit AND tune; model target
selection must use those source roles, not heldout performance. Of these, 36
also meet that minimum in both evaluation roles. This is support, not power
or validated reconstruction accuracy.

| Target | Fit | Tune | Evaluation | Hospital |
|---|---:|---:|---:|---:|
| Creatinine | 916 | 333 | 309 | 193 |
| BUN | 913 | 334 | 309 | 194 |
| Glucose | 903 | 329 | 304 | 190 |
| MAP | 806 | 300 | 270 | 177 |
| Hemoglobin | 917 | 333 | 310 | 193 |
| Troponin I | 397 | 149 | 137 | 42 |

Fibrinogen, LDH and TSH are source-eligible but have only 26, 27 and 17 hospital
patients respectively. Keep their external statistical status unsupported.
Other sparse/source-missing slots stay explicit, not fabricated.

## Decision

The pilot validation-size limitation can now be addressed for many targets.
Next train the same shared-body model on the expanded source samples with
complete target/patient exposure and matched placebo controls. This cache is
current-state masked reconstruction, not future labels, not a new independent
database, and not evidence that any target has passed. It is still a MAP-derived
cohort rather than the entire eICU population.

No training weights, release registry, intervals or serving behavior changed.
