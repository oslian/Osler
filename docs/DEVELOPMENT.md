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

# either mode on a ledger of your own (created if missing, never overwritten)
.\scripts\start-demo.ps1 -Database .\var\oslian_stream_ranges_v2.sqlite3
.\scripts\start-demo.ps1 -Mode release -Database .\var\oslian_stream_ranges_v2.sqlite3 -ReleasePython C:\path\to\python.exe -ReleasePythonPath C:\path\to\overlay

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
| `tests/test_stream_trajectory.py` | the patient-trajectory chart, executed under Node: one time axis, the two regions, the four forecast states, the frozen historical snapshot, no interval band, accessible description, one-column phone layout. Skips with a reason when `node` is not on `PATH` |
| `tests/test_stream_signals_view.py` | the indicator view-model, executed under Node: the flag read only from the observation's own interpretation or supplied range, current / previous / forecast as three comparisons, the forecast horizon, the reading order, one organ-attention rule behind the rail, the figure and the report, a report frozen when it is opened, and the charts' range band and single-observation rule. Skips without `node` |
| `tests/test_stream_observation_statements.py` | the passthrough of each observation's `interpretation` and `referenceRange` onto the state and timeline documents; nothing decided server-side |
| `tests/test_stream_resource_versions.py` | one content per (id, version): idempotent re-delivery, a differing payload refused with `ResourceVersionConflict`, a new version accepted, amendments still ordered, an old ledger meeting the revised fixture without a mixed timeline, ranges present from the first event |
| `tests/test_reference_ranges.py` | the reference-range registry (one cited public source per interval), the fixture builder writing it into `Observation.referenceRange` with a derived, marked `interpretation`, the fixture reproducible byte for byte, and the state carrying it all through |
| `tests/test_ward_ranking.py` | the ward attention order: bands from published cutpoints, order within a band, urine output reported and never ranked, fail-closed unscored stays |
| `tests/test_ward_cases.py` | the retrospective per-stay cases behind the ward rows, opened by `retrospective:<case id>` |
| `tests/test_ward_view.py` | the ward fixture and `/api/ward`: provenance, fail-closed, no runtime dependency |

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
git-ignored in full. To start from an empty ledger, pass `-Database` with a new
path, or stop the server and delete `var/oslian_stream.sqlite3*`; the next start
recreates it and replays the fixture from the beginning.

One (id, version) holds one content. `Repository.insert_raw()` returns the
existing row for an identical resource, refuses a different one with
`ResourceVersionConflict`, and inserts a new version normally. When a fixture
revision changes a resource, the builder raises that resource's `meta.versionId`
(see `FIXTURE_REVISION` and `REVISION_NOTES` in `build_synthetic_fixture.py`),
and a stored session is resumed only onto the fixture revision it was opened on;
`tests/test_stream_resource_versions.py` covers all of it, including an old
ledger meeting the revised stream.

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
