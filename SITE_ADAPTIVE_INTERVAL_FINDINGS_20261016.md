# Self-Calibrating Intervals for Deployment at Any Site (2026-10-16)

Goal approved on 2026-10-16: a model that works at any hospital and for any patient without a
local calibration study. It may learn from outcomes that arrive automatically after deployment.

## Method (`osler_jepa/site_adaptive_conformal.py`)

1. **Day one:** covariate-shift weighting of the MIMIC calibration scores, using only the new
   site's unlabeled inputs (weighted split conformal, Tibshirani et al. 2019).
2. **Afterwards:** online log-scale quantile tracking (P-term of conformal PID control,
   Angelopoulos, Candès and Tibshirani 2023; a variant of adaptive conformal inference, Gibbs and
   Candès 2021). Each forecast's outcome updates the interval multiplier once it is observed.
   Levels are additive: site, care unit, value group, patient. Level keys are site-local, so no
   cross-site vocabulary mapping is needed. The multiplier is unbounded, so it can follow error
   scales larger than anything seen in MIMIC; plain ACI over source-score quantiles could not
   (found while writing the unit tests).
3. Strict causality: an outcome is used only after anchor + horizon, or once the stream has moved
   to a later stay.

## Selection (MIMIC only, `site_adaptive_interval_dev.py`)

Two simulated deployments inside MIMIC: S1 time (fit 2008–2013, deploy 2017–2022) and S2 care unit
(deploy to CVICU + CCU); 47 cells × 2 = 94 streams; 20 candidates; predeclared rule (most streams
with patient- and care-unit-equal coverage in [0.87, 0.93]).

| Candidate | Passing streams | Width vs static |
|---|---|---|
| static v2 | 72/94 | 1.00 |
| day-one weighting only | 72/94 | 1.01 |
| **weighting + site (η 0.02) + care unit (η 0.01)** (selected) | **85/94** | 1.004 |
| other adaptive combinations | 83–85/94 | 1.00–1.03 |

The care-unit level was added to the candidate set after a one-cell MIMIC smoke test; no external
data had been read. Most of the gain comes from online tracking. Day-one weighting adds nothing
under MIMIC's mild internal shifts.

## External simulated deployment (`site_adaptive_external_audit.py`, non-independent)

Each eICU hospital (and NWICU) is an independent deployment starting from MIMIC calibration.
Streams use up to 60,000 rows per cell, stays in fixed random arrival order. v4 data. Each cell is
refit on 5 MIMIC samples; robust = full pass in ≥ 80% of samples. Coverage is measured over the
whole stream, including the adaptation period.

| | Static v2 | Self-calibrating | Change |
|---|---|---|---|
| eICU robust full pass (47 continuous cells) | 11 | **27** | +16, no losses |
| eICU robust interval pass | 12 | **30** | |
| eICU median patient / hospital coverage | 0.871 / 0.865 | **0.890 / 0.886** | |
| NWICU robust full pass (38) | 28 | **31** | +4, −1 (sodium@24h over-covers, 0.931) |
| NWICU robust interval pass | 29 | **32** | |
| Median width vs static | | eICU 1.08, NWICU 1.03 | |

eICU gains: urine_output@1h/3h, heart_rate@6h/12h, map@6h/12h/24h, potassium@6h/12h/24h/48h,
hemoglobin@12h, o2sat@12h, bicarbonate@24h, creatinine@24h, glucose@24h.

## What still fails

- **eICU, 20 cells.** In 16 the **point forecast** fails (point-gate share < 0.8: map@1h,
  platelets@1h/3h, glucose@3h/6h/12h, heart_rate@3h, hematocrit@3h/6h/12h, hemoglobin@3h/6h,
  potassium@3h, bicarbonate@12h, chloride@12h, map@48h). Intervals cannot fix a forecast that
  does not beat persistence. Only 4 are interval-only failures: glucose@1h, creatinine@6h,
  creatinine@12h, bun@24h (coverage 0.84–0.89).
- **NWICU, 7 cells.** Mostly small care units still under-covered (care-unit-equal 0.82–0.86;
  the care-unit step size 0.01 adapts slowly); glucose@1h point gate; sodium@24h over-covers.
- Interval calibration at new sites is now largely solved for cells whose point forecast
  transfers: 27 of 31 robust point-passing eICU cells pass fully. **The remaining bottleneck is
  point-forecast transfer.**

## Limits

- Non-independent: eICU and NWICU were used before. Independent validation of the frozen method
  needs an untouched database (SICdb or HiRID), under the multi-sample protocol.
- Arrival order is simulated (random stay order, one stay at a time); real sites have
  overlapping stays and calendar drift.
- The coverage guarantee of quantile tracking is long-run; short streams (about 280 rows per
  eICU hospital) spend much of their length adapting.

`causal_claim_allowed: false`. Nothing promoted; registries unchanged.

## Update: online point-forecast recalibration (2026-10-16)

Module `osler_jepa/site_adaptive_point.py`: online affine recalibration of the frozen forecast,
corrected = current + scale·a + (1 + b)·delta. `a` and `b` are updated by online median-regression
steps; b is bounded to [−1, 0.5], so b = −1 is persistence. The recipe's comparator receives the
same correction.

MIMIC-only selection (`site_adaptive_point_dev.py`, predeclared rule): `affine_0.005` (site
level), 83/94 full-pass streams vs 81/94 without correction, median MAE ratio 0.995.

External simulated deployment (5 MIMIC samples, non-independent). Static v2 and interval-only
results reproduced the previous run exactly.

| | Static v2 | Adaptive interval | + online point correction |
|---|---|---|---|
| eICU robust full pass (47) | 11 | 27 | **27** (+heart_rate@3h, −o2sat@12h) |
| eICU robust point pass | 31 | 31 | 32 |
| eICU median MAE / persistence | 0.892 | 0.892 | **0.875** |
| NWICU robust full pass (38) | 28 | 31 | **33** (+potassium@6h, +sodium@24h) |
| NWICU median MAE / persistence | 0.908 | 0.908 | 0.888 |

### Why the eICU point failures remain: the comparator gate, not persistence

Sample-0 breakdown of the 15 eICU cells still failing the point gate:

- **13 cells beat persistence on both patient and hospital axes** but do not beat their recipe's
  simpler comparator:
  - treatment-context ridge vs measurement-pure ridge: map@1h, glucose@3h/6h/12h,
    hematocrit@6h, map@48h;
  - change hurdle vs plain ridge: platelets@1h/3h, hemoglobin@3h/6h, bicarbonate@12h,
    hematocrit@12h (chloride@12h passes in sample 0 but not robustly).
- Only **hematocrit@3h and potassium@3h** fail against persistence (hospital axis only).

So at eICU sites the forecasts are useful (they beat persistence), but the extra structure of
the selected recipe (treatment context or change hurdle) does not transfer. The simpler
comparator model generalizes at least as well.

### Implication

The remaining point-gate failures are a **recipe-selection** problem, not a forecasting failure.
The adaptive answer is online model selection at each site: an exponentially weighted forecast
combination (Hedge) of the recipe and its simpler comparator, which is guaranteed asymptotically
to do no worse than the better of the two at that site. That makes the comparator gate redundant
for those cells, so the point gate would become "beats persistence (patient and hospital)". That
is a gate-definition change and needs approval.

## Update: online recipe/comparator combination (Hedge), 2026-10-17

- `osler_jepa.site_adaptive_point.online_expert_combination`: an exponentially weighted average
  forecaster over the online-corrected recipe and its online-corrected comparator.
- MIMIC-only selection (`site_adaptive_combination_dev.py`): every candidate tied at 86/94. Inside
  MIMIC the recipe beats its comparator in 98% of streams, so Hedge converges to the recipe. The
  predeclared rule therefore selected `none`.
- **Protocol deviation** (user decision, option 2;
  `site_adaptive_combination_decision_20261017.json`): Hedge adopted as a safety mechanism with
  η fixed a priori at 1.0.

External simulated deployment (5 MIMIC samples, non-independent). The first three configurations
reproduced the previous run exactly.

| Robust full pass | Static v2 | Adaptive interval | + point correction | + Hedge combination |
|---|---|---|---|---|
| eICU, original gate | 11 | 27 | 27 | 26 |
| eICU, persistence-only gate | 11 | 28 | 28 | **28** |
| NWICU, original gate | 28 | 31 | 33 | 33 |
| NWICU, persistence-only gate | 28 | 31 | 33 | **33** |

- **The combination adds nothing.** On eICU it gains bun@24h, hematocrit@6h, hemoglobin@3h and
  loses potassium@48h, urine_output@1h, urine_output@3h. Median MAE/persistence goes 0.875 → 0.880.
  On NWICU nothing changes.
- **Removing the comparator gate rescues only one cell** (map@1h). The cells that failed the
  comparator gate now pass the point gate (persistence share 1.0) but **fail the interval gate**:
  patient-equal coverage 0.83–0.89 and hospital-equal 0.77–0.87. The comparator gate was masking
  interval shortfalls.

### Remaining bottleneck

Of 19 eICU cells still failing, 18 now fail only on the interval. They sit at 0.83–0.89 over the
whole stream: glucose 1h/3h/6h/12h, platelets, hematocrit, hemoglobin@6h, creatinine@6h/12h,
bicarbonate@12h, chloride@12h, o2sat@12h, map@48h, urine 1h/3h, potassium@48h. Each eICU hospital
has about 280 rows, and the step size frozen on long MIMIC streams (η 0.02) adapts slowly. Tuning η
on eICU would be selection on a used database. The principled fix is an online step-size selector
(DtACI, Gibbs and Candès 2022), which chooses the step size per site without tuning.

### Recommendation

Hedge brings no benefit. By the simplicity rule (AGENTS.md rule 2) it should be removed, together
with the persistence-only gate exception that only applies to combination deployments. Pending
user decision.

### Decision 2026-10-17: Hedge removed

Approved by the user. `online_expert_combination`, its tests, `site_adaptive_combination_dev.py`,
the combination configuration in `site_adaptive_external_audit.py`, and the persistence-only gate
exception in `medical_jepa/AGENTS.md` were removed. The deployment form is the frozen recipe plus
online point recalibration (`affine_0.005`) plus the online interval tracker, with the original
point gate (persistence and comparator). The result files and
`site_adaptive_combination_decision_20261017.json` are kept as records.

## Update: DtACI step-size selection (2026-10-17)

- `osler_jepa.site_adaptive_conformal.dtaci_stream_widths`: DtACI (Gibbs and Candès 2022) over
  step-size experts {0.005, 0.01, 0.02, 0.05, 0.1, 0.2}. Weights follow exponential weights on
  pinball loss with fixed share 0.005, updated in log space for numerical stability (an
  underflow bug was found in the smoke test and fixed, with a test).
- MIMIC-only selection (`site_adaptive_dtaci_dev.py`, predeclared rule, 188 streams including
  300-row short pseudo-sites): **DtACI meta rate 20 adopted**, 163/188 passing streams (short
  80/94) vs 158/188 (short 75/94) for the frozen fixed step, median width 0.995.

External simulated deployment (5 MIMIC samples, non-independent; first three configurations
identical to the previous run):

| Robust full pass | Static v2 | Adaptive interval | + point correction | + point correction + DtACI |
|---|---|---|---|---|
| eICU (47) | 11 | 27 | 27 | **27** |
| NWICU (38) | 28 | 31 | 33 | **33** |

DtACI is neutral externally. eICU robust interval passes go 29 → 30, widths 1.5% narrower, median
coverage 0.890 → 0.885; no cell gained or lost a full pass. It stays the adopted interval tracker
under the protocol.

## Current deployment form and status

**Frozen MIMIC recipe → online point recalibration (`affine_0.005`) → day-one covariate
weighting → DtACI interval tracking (site and care-unit levels).** No local calibration study is
needed; every component learns only from outcomes that have already occurred.

| | Static v2 | Self-adaptive model |
|---|---|---|
| eICU robust full pass | 11/47 | **27/47** |
| NWICU robust full pass | 28/38 | **33/38** |

Remaining eICU failures (20):

- **Comparator gate (15):** the recipe beats persistence but its extra structure (treatment
  context, change hurdle) does not beat the simpler comparator at eICU sites. Recipe complexity
  selected inside MIMIC does not transfer.
- **Interval, short streams (5):** glucose@1h, creatinine@6h/12h, o2sat@12h, bun@24h. Whole-stream
  coverage is 0.86–0.89; about 280 rows per hospital leaves little time after adaptation.

Remaining NWICU failures (5): small care units under-covered (care-unit-equal 0.81–0.85) and
glucose@1h point.

All external numbers are non-independent re-checks. Independent validation of this frozen
deployment form needs an untouched database (SICdb or HiRID) under the multi-sample protocol.

## Update: recipe re-selection by robustness under MIMIC shift (2026-10-18)

`recipe_shift_selection_dev.py` (MIMIC only; predeclared rule: the simplest recipe passing the
full gate in both S1 and S2, else the simplest with the most passes). 21 of 47 cells changed:
measurement-pure ridge 22 → 42, hurdle 19 → 4, treatment-context ridge 6 → 0, plain ridge 0 → 1.
Inside MIMIC the simpler recipes cost MAE: median about +3%, up to +11.5% (glucose@1h).

External simulated deployment (point recalibration + DtACI, 5 MIMIC samples, non-independent;
unchanged-recipe cells reproduce the previous run exactly):

| | Catalog recipes | Shift-selected recipes |
|---|---|---|
| eICU robust full pass (47) | 27 | 27 (+hematocrit@6h, +hemoglobin@6h, +map@1h; −sodium@24h, −urine_output@1h, −urine_output@3h) |
| eICU robust point pass | 32 | **40** |
| eICU changed cells, median MAE/persistence | 0.852 | 0.874 (worse) |
| NWICU evaluable | 38 | **44** (6 former treatment-context cells no longer need treatment data) |
| NWICU robust full pass | 33 | 35 (4 of the 6 newly evaluable pass; common cells 33 → 31, −bun@24h, −sodium@24h) |
| NWICU changed common cells, median MAE/persistence | 0.894 | 0.925 (worse) |

Reading:

- Simpler recipes resolve the comparator-gate problem (eICU robust point passes 32 → 40), but those
  cells then fail on the interval (coverage 0.84–0.89). Full passes do not increase.
- The complex recipes are **more accurate** externally too (lower MAE). Their extra structure is
  not significantly better than the comparator on the gate, but it is not useless.
- The real gain is **portability**: measurement-only recipes can be deployed where treatment
  data is missing (NWICU +6 evaluable cells, 4 robust).

## Overall plateau

Every adaptive refinement after the interval tracker (point recalibration, Hedge, DtACI,
recipe re-selection) leaves eICU at 27/47 robust full passes. The binding constraint is
whole-stream interval coverage in disease-enriched eICU cohorts with about 280 rows per hospital
(0.84–0.89). Further progress from the same non-independent data is unlikely to be meaningful;
the next evidence must come from an untouched database.

## Final deployment form and lockbox protocol (2026-10-18)

**Recipe by data availability** (`osler_jepa/recipe_availability.py`): the treatment-context ridge
falls back to the measurement-pure ridge when a site records none of the target's permitted
treatment features. All other recipes are kept, since they tolerate missing treatment features.
The rule is deterministic and uses inputs only. eICU records treatment, so it keeps every catalog
recipe and its result is unchanged (27/47). NWICU does not, so its 6 treatment-context cells fall
back.

Final NWICU run (`site_adaptive_final_external_NWICU_20261018.json`; catalog cells identical to
the previous DtACI run, fallback cells identical to the re-selection run):

| | Static v2 | Self-adaptive model |
|---|---|---|
| eICU robust full pass | 11/47 | **27/47** |
| NWICU robust full pass | 32/44 | **37/44** |

The evaluation scripts were generalized: `--external-template` replaces the NWICU-only option,
SICdb and HiRID gate axes were added, and NWICU-specific exclusions were replaced by support
checks plus the recipe rule. **Lockbox protocol:** `ADAPTIVE_MODEL_LOCKBOX_PROTOCOL_20261018.md`
(17 code and result files plus 6 MIMIC v4 training files hashed). The scoring scripts refuse to
run on SICdb or HiRID without it or after any hashed change.
