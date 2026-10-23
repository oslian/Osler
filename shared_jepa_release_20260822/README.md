# Shared Whole-Body JEPA 27-Cell Release

This research-only release contains the trained weights and external-validation
contract used by the whole-body personalization runtime on 2026-08-22.

## Contents

- `whole_body_joint_jepa.pt`: frozen shared whole-body JEPA checkpoint.
- `unified_shared_jepa_head_bundle.joblib`: eleven validated target heads using
  the same shared patient representation.
- `runtime_registry.json`: 41-cell fail-closed registry; 27 cells can move and
  14 use persistence.
- `external_validation_report.json`: eICU selection and MIMIC-IV external gate
  results for the converted target heads.
- `FEATURE_CONTRACT.json`: exact offsets and semantics for the 311-value head
  input. Serving code must use `osler_jepa.shared_release_features` rather than
  assembling this vector manually.
- `MODEL_MANIFEST.json`: file sizes, SHA256 checksums, and safety boundary.

## Runtime Contract

The `base_point` block is the frozen shared JEPA forecast for the selected
target and horizon in normalized units. It is not the current target value;
the current normalized target is the final, separate feature.

The 53-value measurement block uses the latest valid value at or before the
anchor in checkpoint variable order. Values up to six hours old are carried
forward for this release. A physical zero is an observed value. Only an absent
or invalid measurement is replaced by the training median (normalized zero).

The release performs factual observation/forecast only. It does not support
causal treatment-effect or clinical decision claims.
