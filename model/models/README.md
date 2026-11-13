# Shared Whole-Body JEPA Point / Interval / Event-Risk Release

This research-only release combines frozen point sources under the
2026-09-02 unified selector. The original 2026-08-25 registry remains an
immutable evidence/binding artifact, not the current source-selection policy.

## Contents

- `whole_body_joint_jepa.pt`: frozen shared whole-body JEPA checkpoint.
- `unified_shared_jepa_head_bundle.joblib`: eleven validated target heads using
  the same shared patient representation.
- `unified_source_selector.json`: current source policy for all 41 cells.
  It compares base JEPA, original shared heads and multi-source heads on the
  same patients, and binds each selected source to its own calibrated interval.
  New replacements require improvement over the incumbent, not just persistence.
  See `UNIFIED_SOURCE_SELECTION_FINDINGS_20260902.md` for exact results and scope.
  The interval-only rescue adds bicarbonate@24h: 28 movable cells, 13 persistence.
  Its conditional radius uses current value and absolute predicted change; its
  scale model and conformal quantile are fitted on separate development patients.
- `runtime_registry.json`: 41-cell fail-closed registry; 27 cells can move and
  14 use persistence.
- `multisource_shared_state_registry.json`: corrected candidate registry;
  37 numeric cells consume the shared state, 23 pass the patient-equalized
  NWICU point gate with real care-unit labels, and 16 also pass interval gates.
  The previous 37-point-pass / 5-latent-pass claim used lost care-unit provenance
  and is superseded. Latent ablation needs corrected revalidation.
- `multisource_shared_jepa_head_bundle.joblib`: target heads for the multi-source
  shared-state candidate. Only cells approved by the unified selector are used
  in the default runtime; the whole bundle never replaces all existing heads.
- `multisource_interval_calibrator.json`: compatible 90% prediction intervals
  fitted on unused MIMIC patients and tested without NWICU recalibration.
  Only the 16 eligible candidate cells emit bounds; others return null.
- `external_validation_report.json`: eICU selection and MIMIC-IV external gate
  results for the converted target heads.
- `cross_database_replication_registry.json`: frozen eICU-to-MIMIC inventory.
  It distinguishes 27 strict direct-transfer cells from 14 challenge cells and
  defines the no-refit policy for future lockboxes.
- `interval_calibrator.json`: release-bound MIMIC-IV patient-split conformal
  calibrators. Twenty-four continuous cells pass patient, care-unit, and
  forward-time 90% coverage gates; four use patient-conditional normalized
  conformal widths.
- `event_risk_heads.joblib`: patient-disjoint calibrated logistic event heads
  for `hypoxemia_event@1h` and `hypoxemia_event@3h`. Both pass seven patient
  splits plus grouped hospital, care-unit, and forward-time gates.
- `cross_target_nowcast.joblib`: measurement-pure same-anchor completion heads
  trained on eICU and gated on MIMIC-IV. Only bicarbonate and heart rate pass
  all seven patient splits, the patient-weighted care-unit gate, the
  capacity-matched placebo comparison, and 90% interval coverage.
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

The release performs factual observation/forecast only. Continuous intervals
come from the selected source's own calibrator. A rejected replacement retains
the original source and interval with its original evidence scope, not a new
NWICU validation claim. Binary hypoxemia cells emit
calibrated event probabilities instead of continuous intervals. It does not
support causal treatment-effect or clinical decision claims.

When a target has never been measured, the runtime may emit a separate
`current_estimate` only for an externally gated cross-target nowcast. This
estimate never overwrites an observation and is not fed into future forecasts;
future use requires a separate cascade-error audit.

For a new database, generate a scored table with frozen predictions and run
`whole-body-lockbox`. See `CROSS_DATABASE_REPLICATION.md`; a missing patient,
domain, or forward-time gate fails closed.
