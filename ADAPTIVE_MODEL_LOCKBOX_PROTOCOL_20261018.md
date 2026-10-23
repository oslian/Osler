# Adaptive Model Lockbox Protocol v3: SICdb, HiRID and Zigong (frozen 2026-10-18T02:39:02-07:00)

Machine-readable version with hashes: `adaptive_model_lockbox_protocol_v3_20261018.json` (v3 supersedes v2: the Zigong extractor was restricted to the laboratory table before any label was read). The scoring
scripts refuse to run on SICdb or HiRID without this protocol, or if any hashed file has changed.

## Prior use

none. Repository-wide search on 2026-10-18: HiRID is mentioned only as not run because the data was absent (GREYBOX_HYBRID_AUDIT.md) and SICdb/HiRID as future validation targets; neither database is on this machine.

## Question

Does the frozen self-adaptive deployment form pass robustly at untouched ICU sites, without any local calibration study?

## Frozen deployment form

1. Point model: frozen MIMIC-IV catalog recipe per cell (mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json), trained on MIMIC v4 data.
2. Recipe rule: osler_jepa.recipe_availability.deployable_recipe: treatment-context ridge falls back to measurement-pure ridge when the site records none of the target's permitted treatment features.
3. Online point recalibration: osler_jepa.site_adaptive_point.online_point_correction, affine_0.005 (site_adaptive_point_dev_20261016.json).
4. Day-one weighting: covariate-shift weights from the inputs of the first min(200, 25%) arriving stays (no labels).
5. Interval: v2 value-bin normalization + DtACI meta rate 20, site and care-unit levels (site_adaptive_dtaci_dev_20261017.json).
6. Comparison: static v2 interval on the same streams.

## Preconditions before any label is read

- data access approved under PhysioNet credentialing and each database's terms
- an extractor per database producing transitions in the MIMIC-IV contract: 6h anchor step, 6h look-back, label = nearest measurement within min(2, h/2) hours of anchor + h, 72h episode cap, plausibility cleaning as in mimiciv_observation_transition_extract.py
- variable mapping declared from the data dictionary only, following the harmonized definitions: glucose = chemistry laboratory glucose; point-of-care glucose -> glucose_poc; blood-gas glucose -> glucose_bg; nurse-validated vital signs where the database distinguishes them; urine output interval-normalized to mL/hour; MAP = measured mean pressure
- an addendum file recording the SHA-256 of each extractor and mapping file, written before the extracted labels are opened

## Evaluation

- Cells: all 49 catalog cells; continuous cells via site_adaptive_external_audit.py, binary event cells via multi_sample_external_audit.py.
- Stream: the database is one site; stays in fixed random arrival order (seeded), chronological within a stay; outcome resolves at anchor + horizon; at most 60,000 rows per cell.
- Samples: 5 MIMIC training samples of 15,000 rows (sample 0 = original seed).
- Gates: point: patient-cluster paired bootstrap CI upper < 0 vs persistence and vs the recipe's comparator; interval: patient-equal and care-unit-equal whole-stream coverage in [0.87, 0.93].
- Robust: full pass in >= 80% of samples (medical_jepa/AGENTS.md).

Primary endpoint: per database: number of evaluable cells that are robust under the adaptive deployment form.


## Zigong addendum (v3)

- **data access:** assumed: the user holds PhysioNet credentialed access and signed this dataset's DUA; archive SHA-256 matches the official SHA256SUMS
- **prior project use:** none (not present on this machine before 2026-10-18)
- **inspected before freeze:** file list, data dictionary, column names, laboratory item/unit/reference-range metadata; value magnitude of two non-target inputs only (nursing blood_sugar -> mmol/L; FIO2 -> empty). No target value was read. After the v2 extraction, only support counts and timestamps were inspected: nursing-chart MAP is empty; nursing ChartTime has day-level resolution (median distinct-time spacing 24 h, 23% missing, 6% before admission); laboratory LabTime is minute-level and complete. No target value was read.
- **extractor:** zigong_observation_transition_extract.py (mapping and unit factors inside; hashed below)
- **declared differences:** infection-only cohort (disease-enriched, like the eICU sepsis module); treatment context not extracted (prescriptions only): treatment-context recipes fall back to the measurement-pure ridge; chemistry glucose is sparse (most glucose is blood gas -> glucose_bg); glucose cells may lack support; times are hours from hospital admission; ICU stays are ICU-department transfer segments; nursing chart excluded (day-level timestamps cannot guarantee causal as-of inputs or hourly labels): vital-sign, urine-output and hypoxemia cells are not evaluable; vital-sign inputs are absent and imputed with MIMIC training medians; not mapped: BNP (MIMIC uses NT-proBNP) and CK-MB (activity vs mass)
- **extraction:** zigong_observation_transitions_{h}h.parquet for h in 1, 3, 6, 12, 24, 48 with 6h anchor step

## Decision, one-shot and re-freeze rules

- cells robust on both SICdb and HiRID are independently replicated adaptive cells; robust on one = candidate. Any registry change requires the user's approval.
- each database is scored once; only crashes may be fixed, without changing methods, gates or cohorts, and every fix is reported
- if any hashed file changes before scoring, a new protocol version must be frozen before any label is read

`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
