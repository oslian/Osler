# Oslian

One repository for the Oslian research code, the whole-body model release, and the demo
that serves it. Research-only: no output here is a clinical claim, a causal claim or a
treatment recommendation (`causal_claim_allowed: false` everywhere).

| Folder | What it is | Rules to read first |
|---|---|---|
| [`model/`](model/) | Whole-body JEPA source, runtime weights, the 2026-09-02 release in `model/models/`, and the frozen adaptive lockboxes (v10, v11). Formerly `oslian/Model`. | [`model/AGENTS.md`](model/AGENTS.md), [`model/README.md`](model/README.md) |
| [`osler/`](osler/) | The earlier Osler research tree: the symbolic engine, the DKA world model, the eICU/MIMIC extraction and audit scripts, and the code the v10 lockbox froze. Formerly the root of this repository. | [`osler/AGENTS.md`](osler/AGENTS.md), [`osler/README.md`](osler/README.md) |
| [`demo/`](demo/) | The Oslian Demo (Flask + static front end): medication safety and FHIR stream monitoring. Its forecasts come from the forecast API in `api/`, checked against a pinned contract. Formerly the `oslian-demo` branch of WholeBody-JEPA. | [`demo/README.md`](demo/README.md), [`demo/docs/MODEL_RUNTIME.md`](demo/docs/MODEL_RUNTIME.md) |
| [`api/`](api/) | The Oslian forecast API (FastAPI) with the PRD, the 2026-11-02 registry and the model files it serves; a self-contained snapshot. | [`api/README.md`](api/README.md), [`api/AGENTS.md`](api/AGENTS.md) |

All three histories are kept: `git log -- model/`, `git log -- osler/` and `git log -- demo/`
show each project's own commits.

## Working in each folder

Run each project from its own folder. Scripts and frozen evidence bind paths relative to
that folder (protocol hashes, report `bindings`, lockbox `frozen_code_sha256`), and those
paths did not change when the projects moved here.

```powershell
cd model; python -m scripts.validate_release          # model release check
cd osler; python -m pytest                             # Osler regression tests
cd demo;  .\scripts\start-demo.ps1                     # the demo (detached mode)
```

Research documents in `model/` and `osler/` are indexed by topic in each folder's
`docs/INDEX.md`. A file that still sits at a project's root is bound by path from code or
frozen evidence; do not move it.

## The demo's forecast contract

`demo/backend/oslian_demo/monitoring/fixtures/forecast_api_contract_v1.json` pins the 45
cells the demo may show and the SHA-256 of the registry they come from. In attached mode
the demo compares the forecast API's own cell list and registry hash with it before the
first forecast and on every reply, and refuses to run if they differ. See
[`demo/README.md`](demo/README.md).
