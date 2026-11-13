# Oslian Demo

A research-only demonstration of two independent clinical-informatics workflows
that share a browser shell and nothing else:

- **Medication Safety** — a deterministic symbolic engine ranks drug candidates
  for an indication and explains the ranking. An LLM only parses free text and
  narrates a result that is already fixed; it never changes the ranking.
- **Monitoring** — a FHIR R4 stream is validated, ledgered, normalised and
  forecast by the WholeBody runtime adapter, and drawn on an anatomical figure.
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
- Forecast cells are **factual forecasts**. Persistence is a baseline
  fallback, not a prediction. A cell shows a 90% interval only when the release
  emits one from that cell's own externally audited calibrator; every other
  interval is `null`. Event cells are calibrated probabilities. Every response
  carries `causal_claim_allowed: false`.
- The pinned model release is approved **only as a bounded research-demo
  runtime**. It must never be described as a formal adjacent-three-level model, a
  production runtime, or a clinically validated model.

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
  model-lock.json         the only model release the release runtime may load

tests/                    the demo test suite
scripts/                  start, test, and the three model-runtime scripts
docs/                     ARCHITECTURE.md, DEVELOPMENT.md, MODEL_RUNTIME.md, FRONTEND.md
var/                      local runtime state (SQLite ledger); never committed
```

Flask serves `frontend/` as its static folder and its template folder, so the
front end stays one obvious directory instead of being buried inside the Python
package. `backend/oslian_demo` imports nothing from any other repository.

---

## Where the demo and the model live

The demo is the `demo/` folder of the **oslian/Osler** repository. The model it
serves is `model/` in the same repository; the demo never copies a weight, it
records *which* model it serves in `backend/model-lock.json`:

```text
https://github.com/oslian/Osler.git
  demo/    this application
  model/   the whole-body JEPA release (model/models/) and its source
  osler/   the earlier research tree
```

The lock pins the last commit that changed `model/` and every artifact's SHA-256.
Any later change to `model/` makes release mode refuse to start until the lock is
regenerated on purpose (see [docs/MODEL_RUNTIME.md](docs/MODEL_RUNTIME.md)).

### Clone

```powershell
git clone https://github.com/oslian/Osler.git
cd Osler\demo
```

The repository root carries `.gitattributes` with `* text=auto eol=lf`, so the
hashed JSON artifacts in `model/models/` check out byte for byte on Windows.

---

## Quick start (fixture mode)

Fixture mode is the safe default: no model process, no weights, no
configuration. Every run completes with 41 `degraded` cells and
`forecast_point: null`, and the UI says *runtime not attached · gated*.

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

## Release mode

There is nothing to initialise: `model/` is part of the clone. Start the release
runtime with:

```powershell
.\scripts\start-demo.ps1 -Mode release -ReleasePython C:\path\to\python.exe -ReleasePythonPath C:\path\to\sklearn-overlay
```

With its own ledger as well:

```powershell
.\scripts\start-demo.ps1 -Mode release -Database .\var\oslian_stream_ranges_v2.sqlite3 -Python C:\path\to\python.exe -ReleasePython C:\path\to\python.exe -ReleasePythonPath C:\path\to\sklearn-overlay
```

The release worker needs an interpreter that can import `model/` with
`scikit-learn==1.7.2` — the version the serialized population anchors were fitted
with. `-ReleasePythonPath` prepends a dependency overlay if your main interpreter
has a different scikit-learn. Both also read
`OSLIAN_WHOLEBODY_PYTHON` / `OSLIAN_WHOLEBODY_PYTHONPATH`.

Full verification in one command:

```powershell
.\scripts\check-model-runtime.ps1 -ReleasePythonPath C:\path\to\sklearn-overlay
```

---

## Verifying health

```powershell
curl.exe http://127.0.0.1:5000/api/model/health
```

Fixture mode answers HTTP 503 with `"status": "runtime_not_attached"` and
`"ready": false`. That is the correct gated answer, not a failure.

Release mode must answer HTTP 200 with all five of:

| Field | Expected |
|---|---|
| `runtime_mode` | `release` |
| `runtime_attached` | `true` |
| `status` | `healthy` |
| `ready` | `true` |
| `scikit_learn` | `1.7.2` |

Anything else means a fail-closed check failed; `checks` in the same response
names which one — pinned commit, model manifest, a SHA-256, module origin, the
scikit-learn pin, the registry count, or the smoke prediction.

### The 41 cells

The pinned registry has 41 target × horizon cells, and the split is a property of
the release, not a runtime decision:

- **28 model-capable cells** (`can_move: true`), chosen per cell by the release's
  unified source selector: 11 `frozen_whole_body_jepa`, 11
  `unified_shared_jepa_target_head`, 4 `multisource_shared_jepa_target_head`, and
  the 2 `hypoxemia_event` cells from `calibrated_event_risk_head`.
- **13 persistence-fallback cells** (`can_move: false`,
  `source_model: patient_persistence_anchor`) — the target × horizon did not pass
  its validation gate, so the cell carries the patient's last observed value
  forward and says so. **Persistence is a safety policy, not a prediction.**

**25 continuous cells** may show a 90% interval: the release emits `lower`/`upper`
only from that cell's own calibrator (`externally_audited_mimic_split_conformal_90`
or `unified_source_disjoint_calibration_90`). The runtime refuses any other bound —
on a cell the catalog does not mark `interval_capable`, on a baseline, or one that
does not contain its point — and fails the run closed. The two event cells are a
calibrated probability in [0, 1] and never carry an interval. A head that declines
to move is reported by the release as persistence with
`fallback_reason: continuous_head_abstained`. All 41 carry
`causal_claim_allowed: false`.

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

A model change is a commit to `model/`, and it is not live in the demo until the
lock is regenerated in a separate commit:

```powershell
# 1. change and validate the model, from model/
cd ..\model
python -m scripts.validate_release
git commit -am "model: ..."

# 2. regenerate the contract fixture through the worker, from demo/
cd ..\demo
$commit = git log -1 --format=%H -- ..\model
$env:PYTHONPATH = "backend"
python -m oslian_demo.monitoring.build_release_contract_fixture --release-dir ..\model --commit $commit --write
```

Then update `backend/model-lock.json` (commit and artifact hashes from
`model/models/MODEL_MANIFEST.json`), run the tests in fixture and release mode,
and commit the lock and fixture together, touching nothing under `model/`.

---

## Environment variables

All optional. With none set the demo serves fixture mode on `127.0.0.1:5000`
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
| `OSLIAN_WHOLEBODY_RUNTIME` | `fixture`, `release` | `fixture` |
| `OSLIAN_WHOLEBODY_RELEASE_DIR` | release directory | `../model` |
| `OSLIAN_WHOLEBODY_PYTHON` | interpreter for the release worker | this interpreter |
| `OSLIAN_WHOLEBODY_PYTHONPATH` | dependency overlay for that interpreter | — |
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
release-worker test skips itself in fixture mode; `.\scripts\test-demo.ps1 -Mode release`
runs it against the locked commit.

---

## Stopping the demo

One process. `Ctrl+C` in the window running it stops Flask, the playback threads,
the SSE bus and — in release mode — the worker subprocess.

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
| `no models directory in ...` | `OSLIAN_WHOLEBODY_RELEASE_DIR` points somewhere other than `model/`. |
| `model/ was last changed in <sha>` | `model/` changed after the lock was written. Re-lock deliberately (Changing the model). |
| `model files do not match model-lock.json` | Almost always a CRLF checkout. The setup script forces LF; fix the checkout, never the check. |
| `/api/model/health` returns 503 `runtime_not_attached` | Fixture mode. Correct and intended; start with `-Mode release`. |
| Release health is `degraded`, `ready: false` | A fail-closed check failed. Read `checks` and `reason`. |
| `scikit_learn` is not `1.7.2` | The worker interpreter has a different version. Pass `-ReleasePythonPath` with a 1.7.2 overlay. |
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
