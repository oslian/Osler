# Oslian forecast API (snapshot 2026-11-12)

A self-contained copy of the forecast API with the code, registry and model files it needs. Research-only:
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`. No patient data is included.

| Path | What it is |
|---|---|
| `osler_api/` | FastAPI service: 45 frozen expert cells (stateless) and the online marginal M (per-hospital state). See `osler_api/README.md`. |
| `OSLIAN_PRD_20261111.md`, `OSLIAN_FEATURE_LIST_20261111.md` | Product requirements and feature list. |
| `AGENTS.md` | The engineering rules and the architecture this code follows. |
| `adaptive_replication_registry_20261102.json` | The replication registry (which cell is replicated, candidate or not promoted). |
| `adaptive_cell_weights_20261102/`, `m_weights_20261108/` | Model files, each with a manifest of SHA-256 checksums. |
| everything else | The research modules and configuration files the API imports, at their original relative paths. |

## Why this is a separate folder

Several of these modules also exist in `osler/` in the version the v10 and v11 lockbox protocols froze by SHA-256
(`frozen_code_sha256`). Sixteen of them differ here because development continued. Overwriting the `osler/` copies would
break those bindings, so this snapshot lives beside them. Nothing in `osler/`, `model/` or `demo/` is changed.

## Run

```bash
cd api
python -m unittest test_osler_api test_adaptive_cell_weights
uvicorn osler_api.app:app --host 127.0.0.1 --port 8000
```

Needs Python with the packages the modules import (fastapi, pydantic, scikit-learn in the version the manifests name,
pandas, numpy, pyarrow, joblib, duckdb). Run from this folder: paths are relative to it. The state database
(`osler_api_state/`) is created on first use and is not tracked.

## Known limits

- The expert bundles carry `interval.calibration_scores`: unlabelled, row-level calibration error scores from MIMIC-IV
  patients, with no identifiers, in the same format as `osler/adaptive_cell_weights_20261028/`.
- `creatinine@48h` (stateless expert form) forecasts a rise of about 0.16-0.20 mg/dL for normal current values where
  the observed mean is about 0.03; found post hoc on development data on 2026-11-12.
- This is a snapshot of files copied from a larger working tree. Scripts here that read patient data need files that
  are not published.
