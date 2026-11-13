# Event cells with online Platt scaling: SICdb and HiRID lockbox (protocol v11)

This folder holds the frozen deployment form of the two hypoxemia event cells, its one-shot lockbox evaluation on SICdb and HiRID, the updated replication registry and the fitted weights. Files keep their original flat paths, so module imports, the protocol hashes and the ledger resolve when commands run from this folder.

## Results

| | SICdb | HiRID |
|---|---|---|
| Robust under `online_platt` (static MIMIC model + online Platt scaling) | **2/2** | **2/2** |
| Static model alone (reference) | 0/2 | 1/2 |

- `hypoxemia_event@1h` and `@3h` (SpO₂ ≤ 92% at t + h) pass in 5 of 5 MIMIC training samples on both databases. Under the v11 decision rule they are independently replicated event cells within the claim scope.
- Brier beats persistence and the OPS-recalibrated MIMIC prevalence in every sample. The upper bound of the patient-level paired bootstrap is −0.005 to −0.023.
- Pooled calibration across all samples:

  | Metric | Range |
  |---|---|
  | CITL | −0.005 to 0.011 |
  | Slope | 0.988–1.002 |
  | O/E | 0.997–1.008 |

- The registry `adaptive_replication_registry_20261029.json` (schema v2) now holds two sources:
  - v10 continuous cells: 35 independently replicated;
  - v11 event cells: 2 independently replicated.

  That makes 37 in total. The v10 entries are unchanged.

Detail is in:
- `ADAPTIVE_EVENT_LOCKBOX_RESULTS_V11_20261029.md`;
- `site_adaptive_event_lockbox_{SICdb,HiRID}_v11.json`.

## Contents

**Protocol**
- `adaptive_event_lockbox_protocol_v11_20261029.json` (sha256 `4ff30975…2ac0`) and `ADAPTIVE_EVENT_LOCKBOX_PROTOCOL_V11.md`.
- The protocol pins:
  - 52 code and configuration files, all included here. Their hashes verify, including the v10 protocol that v11 names as its previous protocol.
  - 2 MIMIC-IV training parquets, which are not included. As a result, `verify_protocol`'s training-data check cannot pass inside this folder.

**Lockbox records**
- v11 ledger receipts are in `causal_executions/4ff30975…/`.
- Each database has:
  - an addendum receipt, whose extraction-file digests are byte-identical to the v10 addendum;
  - a scoring receipt, which records the output's SHA-256.
- The v10 receipts (`causal_executions/4b43aa2c…/`) and the v10 results are included, so the registry builder can rebuild both sources.

**Development evidence**

| Material | Files |
|---|---|
| Re-evaluation after the leak fix and the static calibration check | `BINARY_EVENT_REEVALUATION_AFTER_LEAK_FIX_20261028.md`, `binary_event_*` |
| MIMIC-only online-calibration development | `site_adaptive_event_dev_20261028.json` |
| Post-hoc OPS amendment | `site_adaptive_event_amendment_20261028.json` |
| Development-database evaluation | `site_adaptive_event_{eICU,NWICU,Emory}_20261029.json` |
| Codex pre-freeze review and its fixes | `CODEX_REVIEW_EVENT_PREFREEZE_20261029.md` |

**Code**

| Module | Role |
|---|---|
| `site_adaptive_event_audit.py` | Scorer |
| `site_adaptive_event_dev.py` | Development and stream scoring |
| `osler_jepa/online_platt.py` | Gupta & Ramdas 2023, Algorithm 1 |
| `osler_jepa/event_calibration.py` | Calibration estimates and gate |
| `lockbox_addendum.py` | Addendum tool |
| `build_event_lockbox_protocol_v11.py` | Protocol builder |
| `build_external_expert_promotion_registry.py` | Registry builder |

- Tests for all of these are included. All 42 tests pass when run from this folder.
- `rules/active/` is the DKA rule file that `osler_jepa` loads on import. It is not used by the event scorer, and the protocol does not hash it.

## Model weights

`adaptive_event_weights_20261029/` holds 10 joblib bundles, one per cell and MIMIC training sample, plus `MANIFEST.json`. The manifest records SHA-256, versions and training-data hashes.

**How the bundles were made**
- `adaptive_event_weights.py` refits the frozen static model with the frozen helpers and seeds. It writes a bundle only if the bundle reproduces the frozen fit's probability on every MIMIC row of the sample. The largest difference was 0.0.
- `adaptive_event_weights_eicu_check.py` compares each bundle with the scorer's MIMIC + site fit on eICU, a development database. It covers 60,007 rows at 1h and 60,035 rows at 3h, and the largest difference was 0.0 (`adaptive_event_weights_20261029_eicu_check.json`). The feature transform keeps only columns with support in the MIMIC fit rows, so the model does not depend on the site.

**What a bundle contains**

| Part | Contents |
|---|---|
| Feature transform | Medians, scales, clip bounds (measurement-only features) |
| Estimator | Logistic regression, C = 0.20 |
| Shrinkage | Fit prevalence and shrinkage factor (0.65–1.0), chosen on held-out MIMIC patients |
| Online Platt scaling | γ = 0.1, ρ = 100, radius 100, start (1, 0) |

**Loading**
- `joblib.load(path)` returns a dict.
- `adaptive_event_weights.predict(bundle, frame)` returns the static probability.
- `osler_jepa.online_platt.online_platt_scaling(anchor_hours=..., resolve_hours=..., probability=..., outcome=..., config=OnlinePlattConfig(gamma, rho, radius))` returns the site-recalibrated probability. It uses an outcome only once its resolve time has passed.
- The frame must be prepared like the scorer's (`normalize_cohort`).
- Use scikit-learn 1.7.2.

**Not in the bundles**
- Any site's online state, which starts at (1, 0) on deployment.

The weights are trained on MIMIC-IV (PhysioNet credentialed data). Keep them inside this private repository.

## Not included

- **Patient-level data.** MIMIC-IV, eICU, NWICU, Emory, SICdb and HiRID are not included, and neither are the extraction parquets.
- **Extraction reports.** The SICdb/HiRID reports are byte-identical to the ones in `../adaptive_lockbox_v10/`.

## Limitations

- **Prior use of the databases.** SICdb and HiRID were already scored under v10 for the continuous cells. This is a newly frozen binary-task evaluation on databases already used, not a first look at untouched ones.
- **Post-hoc choices.** OPS and the calibration tolerances were adopted post hoc on the development databases, so the development results are in-sample for these choices.
- **Retrospective replay.** The evaluation is a retrospective observation-time replay. It does not show real-time label latency or late corrections.
- **Site-level calibration only.** Calibration was checked pooled at the site level:
  - SICdb 3h: care units INBD and INIC fall outside the tolerances in 6 of 20 unit × sample checks;
  - HiRID has one care unit, so unit-level calibration is untested there.
- **Where the gain comes from.** The static model already beats both comparators on Brier, so OPS mainly fixes calibration.
- **Data provenance.**
  - The HiRID copy is third-party and cannot be verified against PhysioNet checksums.
  - The SICdb source is unconfirmed.
  - SICdb SpO₂ is an hourly aggregate.
- **Use.** `independent_validation: false`; `causal_claim_allowed: false`; `clinical_promotion_allowed: false`. Research use only.
