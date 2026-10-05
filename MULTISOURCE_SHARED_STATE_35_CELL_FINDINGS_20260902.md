# Multi-source shared body state: corrected 35-cell target audit

## Correction

The original claim that all 37 numeric cells passed the complete NWICU external
gate was too strong. The generic normalizer replaced real NWICU care-unit labels
with `unknown`, and a single-group bootstrap was accepted. The corrected audit
restores care-unit labels and weights patients equally within units. The old
aggregate report is historical evidence only, not current promotion authority.

## Question

Can at least 35 of the frozen 41 forecast cells use one shared whole-body state
without relying on an eICU-only result?

## Frozen design

- Development data: source-balanced eICU and MIMIC-IV.
- Shared encoder: the packaged whole-body JEPA checkpoint.
- Forecast layer: target-specific heads selected without NWICU labels.
- External lockbox: NWICU.
- Comparator 1: patient persistence.
- Comparator 2: a capacity-matched measurement-only head that receives the same
  current raw measurements but not the learned JEPA state.
- All treatment inputs are restricted to observations available before the
  anchor. No causal claim is allowed.

## Result

| Meaning | Cells |
|---|---:|
| Frozen catalog | 41 |
| Numerically evaluable on NWICU | 37 |
| Corrected NWICU point gate | **23** |
| Corrected point AND compatible 90% interval gates | **16** |
| Not evaluable on NWICU | 4 |

The requested 35 externally passing cells have **not** been achieved. All 37
numeric cells consume shared features, but that is not the same as passing the
point gate or demonstrating unique latent value. The five previously reported
latent-increment cells also require care-unit-corrected ablation before that
claim can be reinstated.

The four unevaluable cells are the two hypoxemia-event cells and the two urine
output cells. The NWICU numeric lockbox used here does not provide a compatible
event/urine contract for those cells.

## Boundary

This is a candidate, not an automatic runtime replacement. The point checkpoint
and heads remain frozen. Calibration excludes every patient in each head's
original MIMIC selection/fitting pool. Only unused MIMIC patients supply
residuals. NWICU does not supply radii or select methods; its prior point results
have been inspected, so it is not a newly untouched lockbox.

See `MULTISOURCE_INTERVAL_CALIBRATION_FINDINGS_20260902.md` for all 37 radii,
coverage rates, failures, and usage. The existing 27-cell runtime is unchanged;
these 16 calibrated candidate cells must not be added to 27 as if they were
disjoint targets. The result remains factual and observational only.

## Artifacts

- `models/multisource_shared_state_registry.json`
- `models/multisource_shared_jepa_head_bundle.joblib`
- `models/multisource_interval_calibrator.json`
- `artifacts/multisource_shared_jepa_forecast/multisource_shared_jepa_forecast_audit_20260902.json`
  (historical audit with incorrect care-unit handling; superseded for promotion)
