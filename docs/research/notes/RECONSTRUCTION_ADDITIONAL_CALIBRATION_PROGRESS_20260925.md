# Multi-Component Calibration Progress

## Latest: All Five Folds and Assessment Complete

Fold1fresh381234pairs14761people;fold3and4fresh380978pairs14759people each.
All roles/hashes/labels/components verified before fitting.189supported/46zero
offset target-folds.235offsetrecords reproduced and weightedcalMAE invariant
verified.29patient/27component targetmeans improve;0fullincrementalscreens.
See RECONSTRUCTION_ADDITIONAL_CALIBRATION_FINDINGS_20260925.md. Earlier pending
entries below historical; no runtime change or promotion.

Frozen protocol: `RECONSTRUCTION_ADDITIONAL_CALIBRATION_PROTOCOL_20260924.md`.
Current-value masked reconstruction only. This is point-bias calibration,
NOT90%conformal interval calibration. Existing outer data is reused validation.

## Completed Predictions

- Fold2:380978pairs14759people, exact reuse of fold0outer under verified
  bitwise model states, medians and scalers. Exact CALIBRATION-role identity
  matrix matches; no other fold's incompatible predictions substituted.
- Fold0:356944pairs14735people, freshly computed with fold0's own frozen
  real/process models. Process completed normally without interruption.

Both outputs verify exact calibration patient-target identities, no duplicates,
raw labels, component provenance, positive scales, finite predictions and bound
input/model hashes. Metadata records reused versus newly computed explicitly.
Two role-validation tests cover accidental TEST inclusion, missing/duplicate
identities, wrong labels/components and invalid scales.

Folds1,3,4 inference remains pending. These counts are fold-specific prediction
counts, not evidence of independent replicas or accuracy gains. No offsets fit,
no corrected outer assessment, no promotion or runtime changes yet. After all
calibration inference, apply the same fixed component-equal weighted median
offset rule to real/process/population; do not tune offsets on outer outcomes.
