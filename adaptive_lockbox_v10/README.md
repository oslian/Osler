# Self-adaptive deployment form: SICdb and HiRID lockbox (protocol v10)

This folder holds the frozen self-adaptive deployment form, its one-shot lockbox evaluation on SICdb and HiRID, and the resulting replication registry. Files keep their original flat paths, so module imports and the protocol hashes resolve when commands run from this folder.

## Results

| | SICdb | HiRID |
|---|---|---|
| Evaluable continuous cells | 47 | 38 |
| Robust under `point_and_dtaci_adaptive` | **44** | **36** |
| Static v2 (no online updates) | 4 | 9 |

- **35 cells are robust on both databases.** Under the v10 decision rule they are independently replicated adaptive cells; see `adaptive_replication_registry_20261028.json`.
- Robust means the full gate passes in at least 4 of 5 MIMIC training samples.
- Median MAE gain over persistence across the 35 cells is 8.2% on SICdb and 9.2% on HiRID.
- Patient-level interval coverage is 0.88–0.91, against a 0.90 target.

Detail is in:
- `SICDB_LOCKBOX_RESULT_V10_20261027.md`
- `HIRID_LOCKBOX_RESULT_V10_20261028.md`
- `site_adaptive_lockbox_{SICdb,HiRID}_v10.json`

## Contents

**Protocol**
- `adaptive_model_lockbox_protocol_v10_20261027.json` and its markdown.
- The protocol pins:
  - 62 code and configuration files, all included here; hashes are verified;
  - 6 MIMIC-IV training parquets, which are not included.

**Lockbox records**
- One-shot ledger receipts are in `causal_executions/4b43aa2c…/`.
- Each database has an addendum receipt, written before any label was read, and a scoring receipt.

**Extractors and tests**
- `sicdb_observation_transition_extract.py`, `hirid_observation_transition_extract.py`, and their tests.
- Codex pre-addendum reviews: `CODEX_REVIEW_*`.

**Extraction reports**
- `{sicdb,hirid}_observation_transitions_{h}h_v1.json`.
- These hold support counts only; no patient identifiers.

**Weight exporter**
- `adaptive_cell_weights.py` and its test.

**Import support (added 2026-10-29)**
- `rules/active/` holds the DKA rule file that `osler_jepa` loads on import. The lockbox scorer does not use its contents.
- The extractors import 7 modules:
  - `eicu_body_system_configs.py`;
  - `eicu_{aki,sepsis,respiratory}_transition_extract.py`;
  - `eicu_demo_pretrain.py`;
  - `physionet2019_pretrain.py`;
  - `osler_jepa/biomarker_prediction.py`.
- About these files:
  - none is pinned by the protocol hashes or the addendum receipts;
  - all were last modified on or before 2026-10-22, before the v10 addendum;
  - without them the code in this folder does not import.
- All 39 tests in this folder pass.

**Registry builder**
- `build_external_expert_promotion_registry.py` and its test.
- It accepts a lockbox result only if its SHA-256 matches the ledger scoring receipt.

## Model weights

`adaptive_cell_weights_20261028/` holds the fitted weights of the 35 replicated cells, one bundle per cell and MIMIC training sample: 175 joblib files, plus `MANIFEST.json` with SHA-256, versions and training-data hashes.

**How the bundles were made**
- The scorer does not keep fitted models, so `adaptive_cell_weights.py` refits each cell's frozen recipe on the same MIMIC samples with the same seeds.
- A bundle is written only if it reproduces the frozen fit's calibration predictions. The bound is float32 precision relative to the largest prediction; the largest difference was 2^-12 on urine output in mL.
- A hurdle bundle keeps only the components of the policy the frozen fit selected.

**What a bundle contains**

| Part | Contents |
|---|---|
| Feature transform | Medians, scales, clip bounds |
| Estimators | Ridge, or the hurdle components |
| Delta clip | Clip bound on the forecast change |
| Online interval start | Point-correction scale, quintile interval scale, conformal calibration scores |

- The conformal calibration scores are residual/scale ratios of held-out MIMIC patients, with no identifiers.
- Recipes across the 175 bundles: 155 measurement-pure ridge, 15 change hurdle, 5 persistence.

**Loading**
- `joblib.load(path)` returns a dict; `adaptive_cell_weights.predict(bundle, frame)` gives the forecast.
- The frame must be prepared like the scorer's (`normalize_cohort`).
- Use scikit-learn 1.7.2.

**Not in the bundles**
- The day-one covariate-shift calibration covariates. They are row-level MIMIC features, so the frozen day-one weighting cannot be fully reproduced from the bundles.
- Any site's online state, which starts at deployment.

The weights are trained on MIMIC-IV (PhysioNet credentialed data). Keep them inside this private repository.

## Not included

- **Patient-level data.** MIMIC-IV, SICdb and HiRID are PhysioNet credentialed data; extraction parquets are not included either.

## Limitations

- **Data provenance (HiRID).** HiRID was scored on a third-party copy that cannot be verified against PhysioNet checksums; see `HIRID_LOCKBOX_PROVENANCE_20261028.md`.
- **Data provenance (SICdb).** The SICdb archive also appears repackaged; its source is unconfirmed.
- **Single department.** HiRID covers one department, so care-unit robustness rests on SICdb alone.
- **Glucose.** No glucose cell is replicated.
- **Use.** `causal_claim_allowed: false`; `clinical_promotion_allowed: false`. Research use only.
