# Fold-Local Input Contract: Completed Replay

This completes preprocessing and support checks for encoder cross-fitting,
NOT encoder training or an accuracy/promotion result. All previous runtime
weights, intervals and safety decisions remain unchanged.

## Source-Only Scaling

Native physiological values use an unclipped affine transform in
`native_event_tokens.encode_group`. Consequently a new source-fold center and
IQR can be applied without pretending missing values are observed. Fit one
summary per patient/variable: prefer its actual reconstruction anchor label;
where no label exists, use that patient's median observed history row value.
These histories are stored reconstruction examples, not a deduplicated ledger;
their sampling affects that median. Do not call it an unbiased lifetime baseline.

Only the fold's fitting patients determine center, IQR and support. Selection
and outer-test values do not contribute. Observed zero remains observed zero
in physical terms; missing inputs retain zero value plus zero mask and their
missing-age sentinel. Ages, event times and other token fields are preserved.
The supported-column IQRs in this replay are all positive. Generic constant
columns use the original normalized unit; that fallback must not be claimed
independent of the inherited scale in an untested dataset.

Across 10,737 patients the unmodified strict transformation checked 246,366
examples successfully. Affine inverse errors were at most 2.94e-5 in existing
normalized units (float32), within the prespecified 2e-6 relative/absolute
tolerance. Labels invert to 1e-12 tolerance. This is numerical reversibility,
not evidence that a trained model's predictions stay unchanged after rescaling.

## Newly Identified Support Gap

Seven variables are absent from every fitting fold: CK-MB, CRP, hs-CRP,
ionized calcium, pH, prealbumin and urine output. They are also absent from
these held-out histories; do not extrapolate this to the entire raw database.

Two folds have an additional source/held-out mismatch:

- Fold 2 has no source myoglobin but 102 held-out examples contain it.
- Fold 3 has no source anion gap but 7,269 held-out examples contain it;
  another 404 examples ask for it as an unsupported reconstruction target.

This is an identified obstacle in the proposed independent encoder experiment,
not proof that previous full-source models lacked the same variables or that
it explains all previous failures. The pooled source does contain these fields.

## Repair And Complete Replay

`encoder_fold_support.supported_example` applies a vocabulary mask determined
only by source fitting support. Unsupported variables receive missing value,
mask and age entries. Events containing only unsupported observations are
removed from the predictor stream, preventing timing leakage through them.
The anatomical observation encoder rebuilds module presence from the remaining
masks. The original observed record is not overwritten. Unsupported targets
and streams with no remaining history explicitly abstain, not fabricated zero.

| Fold | All examples | Accepted | Unsupported target | Empty history |
|---|---:|---:|---:|---:|
| 0 | 73979 | 73979 | 0 | 0 |
| 1 | 58849 | 58849 | 0 | 0 |
| 2 | 53071 | 53070 | 0 | 1 |
| 3 | 36992 | 36588 | 404 | 0 |
| 4 | 31250 | 31250 | 0 | 0 |

For the 33 trained targets, 247,610 of 247,611 examples remain input-eligible.
One has no independent supported history; it must remain in coverage reporting.
The 404 unsupported-target examples are outside those 33 trained targets, but
remain in the complete data report. No patient was deleted to improve scores.

Nine tests pass across fold isolation, fit-only scaling, exact missing/zero
semantics, label inversion, unseen-value invariance and explicit abstention.
Both full-data audits completed. No model efficacy is claimed.

## What Still Must Be Done

Train matched encoders with these fold-specific inputs. Existing bounded7
weights and body caches are not independent outer-fold representations and
must not be relabeled as such. Inherited initialization still needs either a
provenance-qualified development comparison or a matched freshly initialized
arm. Fix training duration before outer scoring or use complete inner encoder
cross-fitting; the one/two-site inner splits are not strong selection evidence.
Use the same inputs for competing arms so a representation difference is not
confounded with a different observation vocabulary. The fitting summaries and
earlier cohort choices remain reused development design, not a new lockbox.

Evidence:
`artifacts/native_reconstruction_encoder_fold_scaling_20260920.json` and
`artifacts/native_reconstruction_encoder_fold_support_20260920.json`.
