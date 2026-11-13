# Osler forecast API

> **Not runnable from this repository yet.** The API imports research modules, the model files (`adaptive_cell_weights_20261102/`, `m_weights_20261108/`) and the registry (`adaptive_replication_registry_20261102.json`), which are not published here. This directory documents the interface and its implementation.

Research-only (`causal_claim_allowed: false`, `clinical_promotion_allowed: false`). Two products:

| | Expert cells | Online marginal M |
|---|---|---|
| Endpoints | `POST /v1/expert/forecast`, `GET /v1/expert/cells` | `POST /v1/m/forecast`, `POST /v1/m/outcomes`, `GET /v1/m/cells` |
| Model | 45 frozen, independently replicated cells (`adaptive_cell_weights_20261102/`) | v13 lockbox form (`m_weights_20261108/`): expert anchor → own-trajectory residual → per-hospital online point correction + Mondrian MultiQT → Gaussian copula |
| Evidence | replicated on several external lockboxes | candidates from one external lockbox (INSPIRE) |
| Anchors | the transitions row format, 6 h apart | the JEPA examples row format, 3 h apart |
| State | none: day-one points, `interval: null` | per hospital, in SQLite; updated by posted outcomes |

## Run

```bash
cd medical_jepa
OSLER_API_ALLOWED_ORIGINS=https://your-site.example .venv/bin/uvicorn osler_api.app:app --host 127.0.0.1 --port 8000
```

The server binds to this computer only. Interactive docs: `http://127.0.0.1:8000/docs`. Browser pages on other sites
can call it only if their origin is listed in `OSLER_API_ALLOWED_ORIGINS`. M's state lives in
`OSLER_API_STATE` (default `osler_api_state/m_state.sqlite3`). It holds stay keys, pending forecasts and outcomes.
Loading fails if any weight file, the registry, the frozen configurations or the scikit-learn version (1.7.2) differ
from the manifests.

## Anchors (both products)

`anchors` is the stay's anchor stream from its first anchor on, evenly spaced. Each anchor is
`{"time": ..., "values": {...}}` with:

- `<variable>_t`: latest value;
- `<variable>_age_hr`: hours since that value was measured;
- `hist_*`: observed treatment context.

Missing values stay missing (`null` or omitted), never 0. The server recomputes previous-anchor values, slopes, hours
since the first anchor and (for M) the `_observed_*` flags. Caller copies are dropped and listed in
`recomputed_columns`. Names the model never reads are listed in `ignored_columns`. For M, a column outside the
examples' state columns (`GET /v1/m/cells` → `input_columns`) is ignored, because it was always missing where M was
fitted.

## Expert cells

Returns, for each cell, the median of the five MIMIC training samples' day-one forecasts and all five. The median
itself was not scored in a lockbox. The interval is `null`, because the validated interval (DtACI) needs each hospital's
outcomes. `flags` lists lockboxes where a cell did not replicate.

## M: forecast, then outcomes

1. `POST /v1/m/forecast` with `hospital_id`, a pseudonymous `stay_id` and `anchors`. Optionally add `"joint": true`.
   Each anchor of a stay is forecast once, in time order.
2. Keep the `forecast_id`. When a forecast cell's target is measured, `POST /v1/m/outcomes` with:
   - `forecast_id`;
   - `cell` (for example `map@6h`);
   - `value`;
   - `measured_at`.

   The measurement must be after the anchor and within the target's strict label tolerance of the cell's
   `forecast_time`, as in the training labels (for example 0.25 h for MAP and heart rate, 1 h for glucose, 2 h for
   potassium and bicarbonate, 4 h for creatinine). One outcome per cell. If several measurements qualify, send the one
   closest to `forecast_time`.
3. An outcome changes the hospital's state when the cell is next forecast at that hospital, in the order the forecasts
   were issued. This is where the frozen replay applies it.

Output rules:

- **Warm-up:** the hospital's first 200 stays get no output (`status: "warm_up"`), but their forecasts and outcomes
  still train the hospital's state.
- **Cohort:** an anchor with fewer than 2 organ modules observed is outside M's cohort (`status: "outside_cohort"`)
  and is not issued.
- **Per cell:**
  - `point` is the median over the five samples of the corrected point;
  - `samples` holds each sample's expert anchor, residual point, corrected point and 15 quantiles;
  - `own_trajectory_status` is the registry status of the residual point.
- **Events:**
  - a probability (share of the 50 members) is output only where the registry's single-event entry is a `candidate`;
  - every other event is `null` with its status, including `potassium >= 6.0` and `bicarbonate < 18`, which are not
    calibrated.
- **Joint:** members coupled by the Gaussian copula per block vector (`cross_target`, `across_horizons`), per sample.
  No joint event probability is output.
- **Known limits:** listed in `boundary.known_limits`.

Forecasts with no outcome after 7 days are dropped with their pending state.

## Tests

```bash
.venv/bin/python -m unittest test_osler_api test_osler_api_online test_osler_api_m
```
