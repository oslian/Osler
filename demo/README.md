# Oslian Demo

A research-only demonstration of two independent clinical-informatics workflows
that share a browser shell and nothing else:

- **Medication Safety** — a deterministic symbolic engine ranks drug candidates
  for an indication and explains the ranking. An LLM only parses free text and
  narrates a result that is already fixed; it never changes the ranking.
- **Monitoring** — a FHIR R4 stream is validated, ledgered, normalised and
  forecast by the validated forecast API in `api/`, and drawn on an anatomical figure.
  Two data modes share one pipeline, one chart and one runtime contract: a
  **live synthetic stream** and a **retrospective replay** of a deidentified
  public eICU demo stay.

The drug engine never reads the stream, and the stream never touches the drug
ranking. There is a test that proves it.

---

## ⚠️ Research-only boundary — read this first

This is a **methodology and architecture demo, not a clinically validated tool.**

- Drug `delta` values and disease perturbations are **hand-authored estimates**
  marked `review_status: unreviewed`. Direction (↑/↓) is meaningful; magnitudes
  are soft.
- `demo_clinical_data.json` is **illustrative, hand-written label text — not real
  FDA labels.**
- The live stream is a **synthetic FHIR R4 fixture**, not an Epic or hospital
  feed, and contains no real patient data. The retrospective case is public,
  HIPAA safe-harbor deidentified eICU CRD Demo data.
- Forecast cells are **factual forecasts**: points only. Every interval is
  `null`, because the validated interval is fitted online on each hospital's own
  outcomes, which a demo stream does not have. Every response carries
  `causal_claim_allowed: false`.
- The 45 forecast cells were frozen and then replicated on external hospital
  databases under one-shot protocols. That is research evidence about retrospective
  data. It is **not** clinical validation, and the demo must never be described as
  a production runtime or a clinically validated model.

No diagnosis, prescribing, clinical claim, causal claim, counterfactual,
treatment recommendation or clinical alert is authorized anywhere here.

---

## Structure

```text
frontend/                 everything the browser loads
  index.html              the single page: Home · Medication Safety · Monitoring
  css/                    app.css, stream.css
  js/                     shell.js and stream/{api,store,format,chart,anatomy,render,main}.js
  assets/anatomy-motion/  the body baseplate, the organ artwork and the layout manifest

backend/
  oslian_demo/            the Flask application package
    app.py                create_app(): the only entry point
    config.py             every path and environment variable, in one place
    __main__.py           python -m oslian_demo
    medication/           Medication Safety blueprint, engine/ and data/
    monitoring/           Monitoring blueprint: the FHIR streaming pipeline, fixtures/
    shared/ontology.py    the state vocabulary this repository owns

tests/                    the demo test suite
scripts/                  start and test
docs/                     ARCHITECTURE.md, DEVELOPMENT.md, FRONTEND.md
var/                      local runtime state (SQLite ledger); never committed
```

Flask serves `frontend/` as its static folder and its template folder, so the
front end stays one obvious directory instead of being buried inside the Python
package. `backend/oslian_demo` imports nothing from any other repository.

---

## Where the demo and the model live

The demo is the `demo/` folder of the **oslian/Osler** repository and owns no
model. Forecasts come from the forecast API in `api/` of the same repository, a
separate process the demo calls over HTTP:

```text
https://github.com/oslian/Osler.git
  demo/    this application
  api/     the forecast API: 45 replicated expert cells, their weights and registry
  model/   the September whole-body JEPA release (research comparison; not served here)
  osler/   the earlier research tree
```

`backend/oslian_demo/monitoring/fixtures/forecast_api_contract_v1.json` pins what
the demo may show: the 45 cells and the SHA-256 of the registry they come from.
In attached mode the API's own cell list and registry hash are compared with it
before the first forecast and on every reply; any difference fails the run closed.

### Clone

```powershell
git clone https://github.com/oslian/Osler.git
cd Osler\demo
```

---

## Quick start (detached mode)

Detached mode is the safe default: no API, no weights, no configuration. Every
run completes with 45 `degraded` cells and `forecast_point: null`, and the UI says
*forecast API not attached*.

```powershell
python -m pip install -r requirements.txt
.\scripts\start-demo.ps1
```

Open <http://127.0.0.1:5000>. Home → **Open monitoring** → **Play** for the live
synthetic stream, or switch to **Retrospective replay** → **Next observation**.

Chat and free-text case parsing need an LLM key — paste it in the top-right box
(OpenAI by default, or pick Gemini), or set `OPENAI_API_KEY`. Without a key the
app still runs fully: rule-based parsing plus the complete reasoning graph; only
the chat degrades to a notice.

Without the launcher:

```powershell
$env:PYTHONPATH = "backend"
python -m oslian_demo
```

To run against a ledger of your own choosing (a fresh one, or one kept apart
for a demo), pass `-Database`. The parent directory is created when missing; an
existing ledger is never deleted or overwritten, and the start banner prints the
ledger actually in use:

```powershell
.\scripts\start-demo.ps1 -Database .\var\oslian_stream_ranges_v2.sqlite3
```

---

## Attached mode

Start the forecast API first, from `api/` (see `api/README.md`; it needs the
research dependencies and `scikit-learn==1.7.2`):

```powershell
cd ..\api
python -m uvicorn osler_api.app:app --host 127.0.0.1 --port 8000
```

Then start the demo against it:

```powershell
$env:OSLIAN_FORECAST_RUNTIME = "api"
$env:OSLIAN_FORECAST_API_URL = "http://127.0.0.1:8000"
.\scripts\start-demo.ps1
```

A forecast is issued at an **anchor**: every 6 hours from the encounter start,
which is how the rows the cells were fitted on are spaced. Each anchor holds a
variable's latest value from the 6 hours before it; an older value is missing at
that anchor. The first forecast therefore appears 6 hours into the stream, a
run's input cutoff is an anchor time, and each cell's horizon counts from it.
Past 72 hours of stay the cells were never fitted, and runs return degraded cells
with `fallback_reason: outside_fitted_episode_window`.

---

## Verifying health

```powershell
curl.exe http://127.0.0.1:5000/api/model/health
```

Detached mode answers HTTP 503 with `"status": "runtime_not_attached"` and
`"ready": false`. That is the correct answer, not a failure.

Attached mode must answer HTTP 200 with `runtime_mode: api`,
`runtime_attached: true`, `status: healthy` and `ready: true`. Anything else means
the API was unreachable or served other cells or another registry; `reason` and
`checks` in the same response say which.

### The 45 cells

The contract has 45 target × horizon cells over 15 targets (MAP, heart rate,
respiratory rate, SpO₂, glucose, potassium, sodium, chloride, bicarbonate,
creatinine, BUN, hemoglobin, hematocrit, platelets, urine output). Each one is
`independently_replicated` in the registry. A cell's `status` is:

- `model`: the API returned a point for this anchor;
- `missing`: the target had no value in the 6 hours before the anchor, so there
  is nothing to forecast from;
- `degraded`: no forecast was requested (detached, or past the fitted window).

There is no carried-forward baseline and no event probability in this runtime.
`lower` and `upper` are always `null`. All 45 carry `causal_claim_allowed: false`.

---

## Synthetic stream and retrospective replay

Both modes are sessions of the same pipeline; switch with the mode control, or
deep-link.

- **Synthetic** — 147 Observation events over ~30 clinical hours (about 10
  minutes at Normal demo). The fixture deliberately includes duplicate delivery,
  an amended version, an out-of-order lab, a missing UCUM unit,
  `dataAbsentReason`, an unmapped code, an `entered-in-error` retraction, °F and
  mmol/L conversions, a blood-pressure panel and urine volume over
  `effectivePeriod`. Transport: **Play / Pause**, **Next event**, and
  **Normal demo** or **Fast demo**.
- **Retrospective** — the deidentified public eICU CRD Demo 2.0.1 stay, converted
  to FHIR R4 at replay time and revealed one row per **Next observation**.
  Withheld rows never enter the session, the DOM or the model input. Transport:
  **Next observation**, **Normal demo** or **Fast demo**.

### Playback speed

A speed is a **simulated arrival clock over a fixture**. It changes when the
demo hands each resource to the pipeline; it never touches the clinical
timestamps inside the resources, and it is not a statement about how fast a
hospital delivers data. So the interface offers two:

| Control | Clock | Where |
|---|---|---|
| **Normal demo** | 1× | main transport, default |
| **Fast demo** | 5× | main transport |
| **Stress test** | 20× | folded into **Developer / QA** |

**Stress test** is a load tool: it drives ingest, the run scheduler and the SSE
channel far faster than any real feed, to prove the demo stays responsive. It is
kept out of the main transport so nobody reads 20× as a clinical rate. The
backend accepts all three speeds exactly as before — `POST
/api/demo-streams/<id>/speed` still takes `1`, `5` or `20`.

Which speeds appear is decided by the **source**, not by the page: each source
returns a `playback` block from `describe()` with `kind`, `step`, `speeds` and
`developer_speeds`, and the page renders what it finds. A real hospital feed
declares `kind: "realtime"` with no speeds, so it would render no speed control
at all — data would simply appear at its actual arrival time. That connection is
a contract boundary today, not an implementation.

**Reset** opens a new epoch: the ledger keeps everything and the session starts
again from the hour-0 baseline, with live and historical organ state both
cleared.

Deep links:

```text
#stream?mode=retrospective&view=functions&function=renal_urinary&organ=kidneys&metric=creatinine&run=<run_id>
```

---

## Changing the model

A model change is a change to `api/` (its weights and registry). It is not shown
by the demo until the contract is regenerated from the running API:

```powershell
$commit = git log -1 --format=%H -- ..\api
$env:PYTHONPATH = "backend"
python -m oslian_demo.monitoring.build_forecast_contract_fixture --api-url http://127.0.0.1:8000 --api-commit $commit
```

Run the tests and commit the contract.

---

## Environment variables

All optional. With none set the demo serves detached mode on `127.0.0.1:5000`
with its ledger in `var/`. Copy `.env.example`; `.env` is git-ignored and must
never hold a real key.

| Variable | Values | Default |
|---|---|---|
| `OSLIAN_HOST` | bind address | `127.0.0.1` |
| `OSLIAN_PORT` | HTTP port | `5000` |
| `OSLIAN_STREAM_DB` | SQLite ledger path (`start-demo.ps1 -Database` takes precedence) | `var/oslian_stream.sqlite3` |
| `OSLIAN_STREAM_LOG` | `events`, `off` | `events` |
| `OSLIAN_STREAM_LOG_FORMAT` | `pretty`, `json` | `pretty` |
| `OSLIAN_STREAM_LOG_PAYLOAD` | `summary`, `full` | `summary` |
| `OSLIAN_FORECAST_RUNTIME` | `detached`, `api` | `detached` |
| `OSLIAN_FORECAST_API_URL` | the forecast API in `api/` | `http://127.0.0.1:8000` |
| `OSLIAN_EPIC_PRIVATE_KEY_PEM` | referenced by name only; never read | — |
| `OSLIAN_PYTHON` | interpreter the PowerShell scripts use | `python` |
| `LLM_PROVIDER` | `openai`, `gemini` | `openai` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | chat and free-text parsing | — / `gpt-4o-mini` |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | the same, Gemini provider | — / `gemini-2.5-flash` |

---

## Tests

```powershell
.\scripts\test-demo.ps1
```

or directly:

```powershell
python -m pytest tests -q
```

`pyproject.toml` puts `backend` on the path, so no install step is needed. The
attached runtime is tested against a stub of the forecast API on a local port, so
the suite needs no model files.

---

## Stopping the demo

One process. `Ctrl+C` in the window running it stops Flask, the playback threads,
and the SSE bus. The forecast API is a separate process; stop it in its own window.

If a port is left bound by an orphaned process:

```powershell
Get-NetTCPConnection -LocalPort 5000 -State Listen | Select-Object OwningProcess
Stop-Process -Id <pid>
```

SQLite leaves `-wal` and `-shm` files beside the ledger in `var/`; that is normal.
To begin from an empty ledger, start with `-Database` pointing at a new file, or
stop the server and delete `var/oslian_stream.sqlite3*`.

The ledger keeps one content per resource id and version. A fixture revision
that changes a resource's content also raises its `meta.versionId`, so an older
ledger accepts the revised stream as new versions; a session opened on an
earlier fixture revision is left paused in the ledger rather than resumed onto
the new one. A resource that arrives with an id and version the ledger already
holds under different content is refused with a *resource version conflict*
(the stream event names the source system, resource type, id and version; the
session-creation route answers 409) instead of being served from the old copy.

---

## Common problems

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'oslian_demo'` | `backend` is not on the path. Use `.\scripts\start-demo.ps1`, or set `$env:PYTHONPATH = "backend"`. |
| `flask is not installed for ...` | That interpreter has no Flask. `python -m pip install -r requirements.txt`, or pass `-Python`. |
| `port 5000 is already in use` | An earlier run is still bound. Stop it (above) or pass `-Port`. |
| `frontend is missing` | The checkout is incomplete — `frontend/index.html` must exist beside `backend/`. |
| `/api/model/health` returns 503 `runtime_not_attached` | Detached mode. Correct and intended; set `OSLIAN_FORECAST_RUNTIME=api`. |
| Attached health is `degraded`, `ready: false` | The API is not running at `OSLIAN_FORECAST_API_URL`, or it serves other cells or another registry than the contract. Read `reason`. |
| No forecast in the first hours of a stream | Intended: the first anchor is 6 hours after the encounter start. |
| Every request takes ~300 ms in the browser, even when idle | You opened `http://localhost:<port>`. The server binds IPv4 loopback only, so on Windows the browser tries `::1` first and pays the fallback on every request — measured at 314 ms median via `localhost` against 15 ms via `127.0.0.1`. Use the URL the launcher prints (`http://127.0.0.1:<port>`). Werkzeug's development server always closes the connection, so keep-alive cannot amortise it. |
| The page looks stale after editing an ES module | Chrome caches modules per document. Open a new tab, or hard-reload. |
| Backend edits have no effect | The server does not hot-reload. Stop it and start it again. |

---

## Why the demo does not copy the weights

The weights live in `model/models/`, verified by a SHA-256 manifest. Copying them
into `demo/` would fork that verification: a file in two places drifts, and the
demo could then load an artifact nobody validated. Instead the demo points at
`model/`, checks the pinned commit and every hash on every start, and refuses to
run — rather than degrading quietly — when either disagrees.

The pinned release passes its own manifest, load and end-to-end validation, but
does **not** satisfy the formal architecture rule
(`AdjacentThreeLevelResidualAdapter`, canonical organ mapping). It is approved
only as a bounded research-demo runtime, through an explicit launch profile, and
is never written to a promotion registry.
