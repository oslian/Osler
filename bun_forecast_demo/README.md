# Osler Personalized BUN 3h Forecast

A small research website that accepts one patient's renal trajectory as CSV
and calls the serialized `patient_state_precision_bun3_v1` artifact. It returns
the population point, personalized three-hour BUN point, and the artifact's 90%
patient-specific conformal interval.

`BUN@6h` motivated this focused website because it was the strongest renal
coupling candidate. It is not relabeled as a three-hour model. Exact three-hour
forecasting uses the separately validated `BUN@3h` artifact.

## CSV contract

At least six rows are required.

Required columns:

- `hours_since_onset`
- `bun` in mg/dL

Optional measurements:

- `creatinine` in mg/dL
- `urine_output` in mL per source observation window
- `potassium`, `bicarbonate`, `sodium` in mEq/L
- `map` in mmHg

Optional observed treatment context:

- `hist_fluids`
- `hist_vasopressor`
- `hist_diuretics`
- `hist_renal_replacement`
- `hist_nephrotoxin`

Blank cells remain missing. The runtime sorts timestamps, carries the latest
measured value forward, derives measurement ages, and never turns an unknown
treatment into zero.

## Run

Use an isolated environment because the serialized scikit-learn pipeline was
created with version `1.7.2`.

```bash
python3 -m venv .venv-bun-demo
.venv-bun-demo/bin/pip install -r bun_forecast_demo/requirements.txt
.venv-bun-demo/bin/uvicorn bun_forecast_demo.app:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860`.

## Docker / Hugging Face Spaces

The included Dockerfile runs from the repository root and exposes port 7860:

```bash
docker build -f bun_forecast_demo/Dockerfile -t osler-bun3 .
docker run --rm -p 7860:7860 osler-bun3
```

The service parses uploads in memory and does not persist patient identifiers
or row-level data.

## Boundary

- factual research forecast only
- no clinical claim
- no causal or counterfactual claim
- no treatment recommendation

## External accuracy audit

`evaluate_local_patients.py` evaluates the exact website artifact on a local
transition cohort with six same-stay causal history rows. It reports MAE,
RMSE, bias, error-threshold hit rates, direction accuracy, delta correlation,
patient-equalized baseline comparisons, and empirical conformal coverage.

```bash
python -m bun_forecast_demo.evaluate_local_patients \
  --cohort /path/to/3h_transitions.parquet \
  --output /tmp/bun3_external_evaluation.json \
  --target-tolerance-hours 1.5
```

The current independent MIMIC-IV result is documented in
`BUN_FORECAST_DEMO_EXTERNAL_EVALUATION_20260814.md`. It must not be interpreted
as causal or clinical validation.

## Website examples and languages

The website can switch between Traditional Chinese and English without
reloading the selected CSV or forecast result. It includes five synthetic,
de-identified teaching trajectories:

- stable renal trend
- progressive BUN rise
- recovering AKI
- sparse measurements
- renal replacement context

The examples contain no real patient rows or identifiers. The public example
catalog is available at `GET /api/examples`; each CSV is downloaded through an
allow-listed `GET /examples/{id}.csv` endpoint.
