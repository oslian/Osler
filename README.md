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
- Forecast cells are **factual point forecasts**. Persistence is a baseline
  fallback, not a prediction. Intervals are always `null`. Every response carries
  `causal_claim_allowed: false`.
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
  research/wholebody-jepa/  git submodule: the model release (see below)
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

## The `oslian-demo` branch, and one repository for two things

This branch lives in the **WholeBody-JEPA repository**:

```text
https://github.com/Chun0613-code/WholeBody-JEPA.git
```

`main` holds the model research — training scripts, extractors, evaluation
reports and the released weights. `oslian-demo` is an **orphan branch**: it
shares no history with `main` and inherits none of those files. Checking out one
never gives you the other's tree.

**Why one repository.** The demo's only reason to exist is to serve that model
release under a lock, and the lock is only meaningful if the demo and the release
move through the same review, the same access control and the same clone. The
release `NOTICE.md` requires the repository to stay private and the weights not
to be redistributed; a second repository would mean a second copy of that
obligation and a second place for it to drift. Keeping both here means the
submodule pointer, `model-lock.json` and the weights are one object under one
policy.

**Why the submodule must never point at `oslian-demo`.** The submodule points at
`main`. If it pointed at `oslian-demo`, that branch would contain a submodule of
itself: `git clone --recurse-submodules` would clone the branch, see the
submodule, clone the branch again, and never stop. Two things prevent it:
`.gitmodules` pins `branch = main`, and `setup-model-runtime.ps1` and
`check-model-runtime.ps1` both refuse to run if the locked tree carries a
`.gitmodules` of its own.

### Clone

```powershell
git clone --branch oslian-demo --recurse-submodules https://github.com/Chun0613-code/WholeBody-JEPA.git oslian-demo
```

Without `--recurse-submodules`, everything except release mode still works;
`backend/research/wholebody-jepa` is simply empty until you initialise it.

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

## Initialising the model runtime

Once per clone:

```powershell
.\scripts\setup-model-runtime.ps1
```

It initialises the one submodule, checks its remote, checks out the commit
`backend/model-lock.json` approves, forces an LF checkout (the JSON artifacts are
hashed byte for byte, so a CRLF checkout fails the manifest), verifies every
model file against the lock, and proves the locked tree declares no submodule of
its own. It installs nothing, fetches nothing newer and pushes nothing.

Then:

```powershell
.\scripts\start-demo.ps1 -Mode release -ReleasePython C:\path\to\python.exe -ReleasePythonPath C:\path\to\sklearn-overlay
```

With its own ledger as well:

```powershell
.\scripts\start-demo.ps1 -Mode release -Database .\var\oslian_stream_ranges_v2.sqlite3 -Python C:\path\to\python.exe -ReleasePython C:\path\to\python.exe -ReleasePythonPath C:\path\to\sklearn-overlay
```

The release worker needs an interpreter that can import the checkout with
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

- **27 model-capable cells** (`can_move: true`) — 16 from `frozen_whole_body_jepa`
  and 11 from `unified_shared_jepa_target_head`.
- **14 persistence-fallback cells** (`can_move: false`,
  `source_model: patient_persistence_anchor`) — the target × horizon did not pass
  its validation gate, so the cell carries the patient's last observed value
  forward and says so. **Persistence is a safety policy, not a prediction.**

All 41 report `lower: null`, `upper: null` and an explicit `interval_status`:
this release bundles no patient-specific conformal artifact, so no interval may
be claimed. All 41 carry `causal_claim_allowed: false`.

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

## Syncing the model research

```powershell
.\scripts\sync-model-research.ps1
```

A dry run: it refuses to touch a dirty submodule, fetches, and shows the locked
commit, the submodule HEAD, `origin/main`, the commits between them and the files
that differ under `models/`. It checks out nothing, rewrites no lock and pushes
nothing.

To try a candidate:

```powershell
.\scripts\sync-model-research.ps1 -Update
```

That checks the candidate out, runs the runtime contract tests against it, and
restores the locked commit either way. A candidate that fails must not become the
demo runtime. A candidate that passes is still not the runtime until you edit
`backend/model-lock.json` deliberately.

---

## Changing the model, and pushing

Model and training changes belong on their own branch of the same repository, not
in a demo commit.

```powershell
cd backend\research\wholebody-jepa
git switch -c model-runtime-fix origin/main
# ... change, test ...
git commit -am "fix: ..."
git push -u origin model-runtime-fix
```

Never commit on a detached HEAD, and always branch from `main`. After the model
branch is validated, come back out and record it on the demo branch — the
submodule pointer and `model-lock.json` in the same commit:

```powershell
cd ..\..\..
git add backend/research/wholebody-jepa backend/model-lock.json
git commit -m "chore: adopt validated model runtime <sha>"
```

Push the demo branch:

```powershell
git push -u origin oslian-demo
```

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
| `OSLIAN_WHOLEBODY_RELEASE_DIR` | release checkout | `backend/research/wholebody-jepa` |
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
| `no models directory in ...` | The submodule is not initialised. Run `.\scripts\setup-model-runtime.ps1`. |
| `release checkout is at <sha>` | The submodule drifted off the lock. Run `.\scripts\setup-model-runtime.ps1`. |
| `model files do not match model-lock.json` | Almost always a CRLF checkout. The setup script forces LF; fix the checkout, never the check. |
| `/api/model/health` returns 503 `runtime_not_attached` | Fixture mode. Correct and intended; start with `-Mode release`. |
| Release health is `degraded`, `ready: false` | A fail-closed check failed. Read `checks` and `reason`. |
| `scikit_learn` is not `1.7.2` | The worker interpreter has a different version. Pass `-ReleasePythonPath` with a 1.7.2 overlay. |
| Every request takes ~300 ms in the browser, even when idle | You opened `http://localhost:<port>`. The server binds IPv4 loopback only, so on Windows the browser tries `::1` first and pays the fallback on every request — measured at 314 ms median via `localhost` against 15 ms via `127.0.0.1`. Use the URL the launcher prints (`http://127.0.0.1:<port>`). Werkzeug's development server always closes the connection, so keep-alive cannot amortise it. |
| The page looks stale after editing an ES module | Chrome caches modules per document. Open a new tab, or hard-reload. |
| Backend edits have no effect | The server does not hot-reload. Stop it and start it again. |

---

## Why the weights are not on this branch

The weights live on `main`, versioned there and verified by a SHA-256 manifest
against a pinned commit. Copying them onto the demo branch would fork that
verification: a file in two places drifts, and the demo could then load an
artifact nobody validated. Instead the demo points at a checkout it does not own,
checks the commit and every hash on every start, and refuses to run — rather than
degrading quietly — when either disagrees.

The pinned release passes its own manifest, load and end-to-end validation, but
does **not** satisfy the formal architecture rule
(`AdjacentThreeLevelResidualAdapter`, canonical organ mapping). It is approved
only as a bounded research-demo runtime, through an explicit launch profile, and
is never written to a promotion registry.
