# System Integration Audit (2026-08-06)

## Scope

This audit traced the factual prediction path from validation reports through
promotion registries, the runtime candidate catalog, the fail-closed serving
adapter, and the personalization demo. It did not alter causal or clinical
boundaries.

## Confirmed And Fixed

1. **The catalog was not executable.** Catalog v9 listed validated cell names
   but did not embed their target-horizon registries. Catalog v10 materializes
   the exact validated cells and contracts.
2. **The runtime did not resolve the catalog.** The serving adapter can now
   resolve one source by exact input contract, target, horizon, and unit.
3. **Registry contracts failed open.** Missing input-contract IDs or variable
   units could previously authorize a prediction. They now fail closed.
4. **Canonical units disagreed.** Bicarbonate, heart rate, and respiratory rate
   used incompatible aliases across extraction and promotion artifacts. Their
   registries now use the event-ledger canonical units.
5. **One cell had two sources under the same contract.** `heart_rate@3h` was
   ambiguous. The newer target-specific full-cohort source now supersedes the
   broader older source for that exact contract and cell.
6. **Prediction provenance was unchecked.** Catalog-backed serving now requires
   the submitted prediction candidate ID to match the resolved candidate.

## Current Runtime State

- Catalog: `whole_body_runtime_candidate_catalog_v10.json`
- Coverage: 48 unique target-horizon cells across 16 targets
- Newly promoted: `chloride@12h`, `hematocrit@12h`
- Safety: exact schema, contract, unit, source, and cell matching; persistence
  fallback on every mismatch
- Verification: 381/381 unit tests pass
- Claims: factual research only; causal and clinical promotion remain disabled

The parent Osler suite passes 236/236 maintained tests. Its obsolete
`test_nexus_full.py` entry was removed instead of preserving the deleted
`NexusMedical.kg` API. Root pytest no longer recurses into the independent
`medical_jepa` environment.

The parent static audit also corrected two invalid JSON states:
`auto_learning/interaction_log.json` is now a valid empty list, and the
unreferenced `chief_complaint_router/router_stats.json` file was deleted because
it contained Python source rather than JSON.

## Remaining Integration Limits

### The active Osler web application does not call JEPA

The live path is `start.sh -> app.py -> nexus_runner.py -> nexus_engine`.
Neither `app.py` nor `nexus_runner.py` imports `medical_jepa`, and the previous
world-model bridge is absent. Therefore the current website is a symbolic
diagnosis application, while JEPA remains a separate factual-research system.

This should not be patched with a fake compatibility bridge. A real bridge must
load the exact inference bundle selected by catalog v10, submit only timestamp-
legal patient history, and return the fail-closed forecast contract. That work
depends on serializing the target-specific inference bundles described below.

### Candidate inference artifacts are not serialized

The promotion registries identify the validated source but generally do not
contain a checkpoint, fitted coefficients, preprocessing state, or conformal
residual artifact. The catalog can select and authorize a source, but an
external prediction provider must still produce the candidate value.

**Required fix:** each future promotion job must save one immutable inference
bundle containing model weights or coefficients, feature schema, scaler,
target/horizon, conformal calibration, and a content hash. The registry should
reference that bundle. A cell is routing-ready today, but is not standalone
model-serving-ready until that bundle exists.

### The personalization demo is a separate legacy research demonstration

`personalization_demo/runtime.py` still calls five hand-built belief feature
builders and applies transparent hand-coded adjustments. It does not invoke
catalog v10 or the current target-specific JEPA sources. Its README is honest
about this, but it must not be presented as the current whole-body model.

**Required fix:** replace it only after inference bundles exist. Do not wrap the
belief demo in a compatibility layer or label its output as JEPA prediction.

### The unified pure-JEPA forecast contract is a different path

`whole_body_forecast_contract.py` unifies independently trained body-system
checkpoints. It intentionally does not claim a validated shared whole-body
latent and does not consume catalog v10. This is a separate research runtime,
not a drop-in provider for the target-specific joint catalog.

## Remaining Model Failures Are Not Runtime Bugs

The regime audit left `wbc@6h`, `wbc@12h`, `wbc@24h`, and `anion_gap@3h`
fail-closed. All passed the three external splits, but only 4-6 of seven patient
splits. This pattern points to patient-level physiological heterogeneity and
unstable residuals rather than hospital-domain shift or another global
calibration defect. Regime conditioning rescued chloride and hematocrit but
did not rescue these cells.

Future work on those cells should improve target-specific future supervision
or add relevant pre-anchor physiological context. Repeating the same interval
calibration or enlarging the shared latent is not supported by this audit.
