# Development

## Layout and imports

The application package is `backend/oslian_demo`. There is no editable install:
the package is put on the path explicitly, which keeps the checkout movable — an
editable install records an absolute path and breaks the moment the directory is
renamed or copied.

- **Tests** — `pyproject.toml` sets `pythonpath = ["backend"]`, so
  `python -m pytest tests` works from the repository root with no setup.
- **Running** — `scripts/start-demo.ps1` sets `PYTHONPATH` to `backend` and runs
  `python -m oslian_demo`.
- **By hand** — `$env:PYTHONPATH = "backend"` then `python -m oslian_demo`.

The research submodule is never on that path. The Flask process must run from
this repository alone; the release worker loads the checkout in its own
subprocess.

Rules the package keeps:

- No module rewrites `sys.path`, with one deliberate exception:
  `monitoring/worker.py` runs as a subprocess *inside* the pinned release
  checkout and must let that checkout shadow every same-named module.
- No module imports from another repository. `shared/ontology.py` exists so the
  Medication Safety engine does not have to.
- No module reads an environment variable except through `config.py` — the
  exceptions being `wholebody.RuntimeConfig.from_env` and
  `eventlog.StreamEventLogger.from_env`, which are the runtime's and the logger's
  own contracts and are listed in `config.py`'s docstring.
- No module computes a package path itself; it imports the constant.

## Everyday commands

```powershell
# start (fixture mode)
.\scripts\start-demo.ps1

# start against the locked model release
.\scripts\start-demo.ps1 -Mode release -ReleasePython C:\path\to\python.exe -ReleasePythonPath C:\path\to\overlay

# the whole demo suite
.\scripts\test-demo.ps1

# one file, verbose
.\scripts\test-demo.ps1 tests\test_stream_api.py -v

# include the release-worker test
.\scripts\test-demo.ps1 -Mode release -ReleasePythonPath C:\path\to\overlay

# model runtime
.\scripts\setup-model-runtime.ps1
.\scripts\check-model-runtime.ps1 -ReleasePythonPath C:\path\to\overlay
.\scripts\sync-model-research.ps1
```

Every script takes `-Python` (or `$env:OSLIAN_PYTHON`) if `python` on `PATH` is
not the interpreter you want, and none of them installs, upgrades or downgrades a
package.

## The test suite

| File | Covers |
|---|---|
| `tests/test_shared_ontology.py` | the state vocabulary this repository owns matches the knowledge files exactly |
| `tests/test_stream_fhir_mapping.py` | R4 structural validation, obs-6/obs-7, code and unit mapping, exact conversions |
| `tests/test_stream_ledger.py` | the append-only ledger, idempotent delivery, amendments, model input, organ state, exact per-run replay |
| `tests/test_stream_runtime_contract.py` | the 41-cell registry, statuses, null intervals, fail-closed gating |
| `tests/test_stream_api.py` | HTTP + SSE contract, anchors, reconnect, medication-safety isolation |
| `tests/test_stream_functions_retro.py` | the functional-group document and the retrospective source |
| `tests/test_stream_eventlog.py` | the two console stages and structural redaction |
| `tests/test_stream_ui_contract.py` | source-level UI contract, accessibility hooks, safety wording, frontend asset paths, anatomy asset rules, and a secret / local-path / PHI scan over the backend package and the frontend |

`test_stream_ui_contract.py` asserts against the source text of the page, the CSS
and the ES modules. If you rename a module under `frontend/js/stream/`, add it to
the `MODULES` dict there so the scan keeps covering it.

## Frontend

`frontend/` is a plain tree with no build step. Flask serves it at `/static`, so
`frontend/css/app.css` is `/static/css/app.css`. The page names four assets, and
a test asserts each one both appears in the markup and exists on disk — the cheap
guard that catches a move.

See [FRONTEND.md](FRONTEND.md) for module boundaries and the React assessment.

## Regenerating fixtures

```powershell
$env:PYTHONPATH = "backend"
python -m oslian_demo.monitoring.build_synthetic_fixture --write
python -m oslian_demo.monitoring.build_release_contract_fixture --release-dir backend\research\wholebody-jepa --commit <sha> --write
```

The release contract fixture must be produced **through the worker**, inside the
locked checkout, so its manifest and registry hash are the real ones. Do not
hand-edit it: the contract tests exist to catch exactly that. If you regenerate
it for a new release, `backend/model-lock.json` must be updated in the same
commit — `WholeBodyRuntime` refuses to start if the two disagree.

## Anatomy assets

Do not edit `frontend/js/stream/anatomy.js` to move an organ. Positions live in
`frontend/assets/anatomy-motion/organ-layout.json`.

When adding, replacing or removing anything under `frontend/assets/anatomy-motion/`,
update `PROVENANCE.json` (the directory's record of record) and, for an organ
illustration, `organs/SOURCES.json`. A licence is recorded per asset, never per
collection, and a recommended credit line is kept even where the licence does not
compel one.

## Browser notes

- Chrome caches ES modules per document. After editing a file under
  `frontend/js/`, open a **new tab** — a plain reload can run the old module.
- The Flask dev server does not hot-reload this package. Restart it after a
  backend edit.
- `requestAnimationFrame` is paused while a browser pane is hidden, so an
  automation that awaits a frame will hang. Force a paint instead.

## Local runtime state

`var/` holds the SQLite ledger and its `-wal` / `-shm` companions, and is
git-ignored in full. To start from an empty ledger, stop the server and delete
`var/oslian_stream.sqlite3*`; the next start recreates it and replays the fixture
from the beginning.

## What must not be committed to this branch

- model weights or any `.pt` / `.pth` / `.ckpt` / `.joblib` file — they live in
  the submodule, verified there against a SHA-256 manifest
- research findings, validation result JSON, audit or training scripts
- the SQLite ledger and anything else under `var/`
- scratch directories from other checkouts
- `.env` — only `.env.example`, which holds variable names and example shapes
- absolute personal paths, in code, scripts or documentation. The UI contract
  test scans the backend package and the frontend for Windows user paths, conda
  paths, private keys, API keys and bearer tokens, and fails on any of them.
