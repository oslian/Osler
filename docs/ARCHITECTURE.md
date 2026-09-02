# Architecture

Oslian Demo is one Flask process with two blueprints that share nothing but the
app object, serving one static frontend tree. This document covers the
internals; [../README.md](../README.md) is the starting point.

## Composition

```text
oslian_demo.app.create_app(config)
  ├─ medication.routes.bp                    Blueprint("medication")
  ├─ monitoring.api.create_blueprint(...)    Blueprint("stream")
  └─ GET /  ->  render_template("index.html")

  static_folder   = frontend/     served at /static
  template_folder = frontend/
```

`create_app()` is the only entry point. `python -m oslian_demo` and every test
build the app through it; there is no second app object, no module-level
`app = Flask(...)`, and no `sys.path` manipulation in the package.

`oslian_demo.config` owns every path and every environment variable, and draws a
line between three kinds: package data (knowledge JSONs, fixtures, approved
maps), the frontend tree, and machine-local runtime state under `var/`.

The one deliberate exception to the no-`sys.path` rule is
`monitoring/worker.py`, which runs as a subprocess *inside* the pinned release
checkout and must let that checkout shadow every same-named module.

## Repository independence

The Flask process imports from this repository only. The state vocabulary the
Medication Safety engine canonicalises against used to come from a research
package; it is now `oslian_demo.shared.ontology`, carrying the seventeen
canonical states the demo's own knowledge files actually name.
`tests/test_shared_ontology.py` reads those knowledge files and asserts that
every variable in them resolves exactly as before — and that the table carries
nothing the demo never asks for.

The only component that loads research code is the release worker, and it does
so in a separate interpreter, inside the submodule. See
[MODEL_RUNTIME.md](MODEL_RUNTIME.md).

## Medication Safety

```text
free text or form fields
   ↓ case_parser        LLM strict-JSON parse, or deterministic regex rules
structured fields
   ↓ case_targets       indication → physiological treatment target states
targets
   ↓ engine.reasoning_engine
       pathologic state(s) → mechanism → official label → patient safety gate
   ↓ disease_world      the local disease "wiki" from the organ JSONs
   ↓ agent.build_graph  the mind-map reasoning graph
bundle  →  /api/analyze (whole) or /api/analyze_stream (NDJSON, one line per step)
   ↓ llm_client         chat that may only explain the bundle it is given
```

The ranking is produced entirely by `engine/`, which is deterministic and has no
model and no network dependency. The LLM sits at two ends only: parsing input,
and narrating a result that is already fixed. `/api/chat`'s system prompt forbids
changing the ranking, inventing a dose, or recommending a drug outside the
candidate list, and the grounding block is the computed bundle.

| Module | Responsibility |
|---|---|
| `engine/reasoning_engine.py` | the five-layer orchestrator; `load_knowledge()` is the single reader of `medication/data/` |
| `engine/drug_identity.py` | canonical drug names, aliases, dedup |
| `engine/drug_profile.py` | per-drug mechanism profile |
| `engine/drug_safety_gate.py` | label-validation-aware dose and contraindication gate |
| `engine/patient_profile.py` | age, renal function, allergies, meds, flags |
| `engine/clinical_role.py` | rank priority by clinical role |
| `engine/clinical_data.py` | the openFDA label / FAERS fetcher (a maintenance CLI, not used at request time) |

## Monitoring

```text
ClinicalDataSource      MockFhirSource · RetrospectiveCaseSource · EpicFhirSource (contract only)
   |  R4 Bundle(searchset) + Observation envelopes
FHIR validator     ->   append-only raw ledger (SQLite, WAL)
   |  (code system, code, shape, unit) mapping registry
canonical observations  ->  delivered timeline + input watermark
   |  effective_time <= input cutoff only
WholeBodyRuntime adapter  ->  model runs + forecast cells (persisted)
   |  SSE (id: = sequence, Last-Event-ID reconnect)
anatomy · organ signals · cells · run history · provenance drawer
```

| Module | Responsibility |
|---|---|
| `source.py` | `MockFhirSource` fixture replay; `RetrospectiveCaseSource` (eICU demo rows → FHIR R4 at measurement time); `EpicFhirSource` SMART/pagination boundary, contract only |
| `fhir.py` | R4 structural validation, obs-6 / obs-7, RFC3339 parsing |
| `mapping.py` | versioned LOINC/UCUM → WholeBody variable rules with exact linear conversions |
| `repository.py` | all SQL: ledger, canonical rows, sessions, deliveries, events, runs, cells |
| `ingest.py` | validate → ledger (idempotent) → map → canonical row or mapping failure |
| `timeline.py` | delivered timeline, per-variable state, organ state, causal-prefix model input, exact per-run replay |
| `wholebody.py` | the single runtime adapter: fixture gate or pinned release worker |
| `worker.py` | the subprocess that runs inside the pinned release checkout |
| `session.py` | simulated arrival clock (1× / 5× / 20×), event bus, coalesced run scheduler |
| `anatomy.py` + `organ_target_map.json` | the explicit organ ↔ variable ↔ cell map (anatomical organs only) |
| `functions.py` + `functional_group_map_v1.json` | the 11 checkpoint modules as clinical-function groups with primary / secondary / related splits and organ links |
| `routing.py` | inverts both maps: variable → organ ids and functional group ids |
| `eventlog.py` | the two-stage terminal console (`event_emit`, `sse_send`) with structural redaction |
| `api.py` | the Flask blueprint: REST controls + SSE |

### Causality: the three invariants

**No future leak.** `timeline.build_model_input` takes only rows with
`effective_time <= input_cutoff`. A run records the cutoff it used.

**Exact historical replay.** A run also records `input_refs_json` — the exact
`Observation/<id>/_history/<version>` set it consumed.
`repository.observations_for_refs` reads those rows back by reference, and
`timeline.run_timeline` is the single basis for *every* historical view: the
signals, the charts, the variable state and the organ snapshot. "The current
timeline truncated at the cutoff" is deliberately not used, because it is a
different set — a resource delivered after the run but timestamped before its
cutoff would slip in, and an amendment would replace one of the run's inputs.

**Server-authored anatomy.** Every `observation.normalized` event carries
`organ_ids` and `functional_group_ids`, computed by `routing.variable_routing()`
from the two approved maps. No client may infer anatomy or function membership
from a variable name, and the organ snapshot in `/state` and `/forecasts` is
resolved through the same function, so the figure and the charts cannot disagree.

Organ freshness is measured on the **stream's clinical clock** (a 2-hour window),
never on wall-clock time, so a replay to the same cursor yields identical state.
Under a historical anchor, freshness is measured on that run's `input_cutoff`.

### Endpoints

```text
POST   /api/demo-streams                  {"source": "synthetic" | "retrospective"} get-or-create (idempotent)
GET    /api/demo-streams/sources          the two data modes and their provenance
GET    /api/demo-streams                  list sessions
GET    /api/demo-streams/{id}             session state
POST   /api/demo-streams/{id}/start|pause|reset   idempotent
POST   /api/demo-streams/{id}/step        deliver the next arrival group now
POST   /api/demo-streams/{id}/speed       {"speed": 1 | 5 | 20}
GET    /api/demo-streams/{id}/events      SSE: stream.state, fhir.resource.accepted, fhir.resource.rejected,
                                          observation.normalized, observation.unmapped,
                                          model.run.started | completed | failed
GET    /api/patients/{id}/state           clocks, watermark, per-variable state, organ state, run head
GET    /api/patients/{id}/timeline        the delivered timeline, or a run's exact inputs with ?run_id=
GET    /api/patients/{id}/forecasts       cells for the live head, or ?run_id= for a historical anchor
GET    /api/patients/{id}/forecast-history every run, newest first; ?cell= for one cell's history
GET    /api/patients/{id}/mapping-failures
GET    /api/fhir/{type}/{id}/_history/{vid}   the raw resource from the ledger (provenance drawer)
GET    /api/model/capabilities | health | anatomy | functions | mapping

GET    /api/cases                         preset cases, indications, whether an env LLM key exists
POST   /api/analyze                       one bundle
POST   /api/analyze_stream                NDJSON, one line per workflow step
POST   /api/chat                          explain the bundle; never changes it
```

### Event console

Starting the demo prints every event the backend produces and every event the SSE
generator actually sends: `event_emit`, one line per `EventBus.emit`, and
`sse_send`, one line per delivery, tagged `mode=live` for a fresh event and
`mode=replay` for ledger catch-up after a reconnect. Keep-alive comments are
transport padding and are never logged.

Redaction is structural rather than a filter: each event type has an explicit
field extractor and nothing outside it is printed.
`OSLIAN_STREAM_LOG_PAYLOAD=full` attaches the raw payload only for a synthetic
fixture source; any other source, including the Epic contract boundary, is forced
back to the redacted summary whatever the configuration says. A console write
that fails is swallowed, so logging can never break the stream.

### Fixtures

- `monitoring/fixtures/synthetic_patient_v1.json` — 147 Observation events over
  ~30 clinical hours, generated by
  `python -m oslian_demo.monitoring.build_synthetic_fixture --write`.
- `monitoring/fixtures/retrospective_eicu_demo_case.json` — the deidentified
  public eICU CRD Demo 2.0.1 stay, converted to FHIR R4 at replay time.
- `monitoring/fixtures/wholebody_release_contract_v1.json` — generated through
  the worker inside the pinned checkout: manifest, registry hash, the 41-cell
  catalog, 53 variables and one real input/output sample.

## Anatomy assets

The figure is a baseplate plus a placement manifest, never hand-drawn shapes.

- **Baseplate** — the accepted hand-fixed LadyofHats figure. Three generated
  display variants (`.back`, `.front`, `.vessels`) differ **only** in `display`
  flags, so organs can sit between the veins and the arteries.
- **Placement** — `frontend/assets/anatomy-motion/organ-layout.json`.
  `frontend/js/stream/anatomy.js` never computes a position: it reads each
  sprite's transform, clip and hit box from the manifest. Placement is
  translation plus uniform scale only; a paired organ is reflected about its own
  centre line.
- **Response** — a second clipped copy of the same artwork, tinted by an SVG
  filter that blurs the alpha and cuts the shape back out, so it reads as a halo
  and never repaints the organ. Only opacity and filter change.
- **Provenance** — `PROVENANCE.json` is the record of record for the directory,
  `organs/SOURCES.json` the per-organ record including each asset's view
  (`anterior` / `lateral`) and whether it is drawn. A contract test refuses any
  non-anterior asset in the drawn set.

To change a position, edit the manifest — never the renderer.
