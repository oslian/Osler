# Oslian

One repository for the Oslian research code, the whole-body model release, and the demo
that serves it. Research-only: no output here is a clinical claim, a causal claim or a
treatment recommendation (`causal_claim_allowed: false` everywhere).

| Folder | What it is | Rules to read first |
|---|---|---|
| [`model/`](model/) | Whole-body JEPA source, runtime weights, the 2026-09-02 release in `model/models/`, and the frozen adaptive lockboxes (v10, v11). Formerly `oslian/Model`. | [`model/AGENTS.md`](model/AGENTS.md), [`model/README.md`](model/README.md) |
| [`osler/`](osler/) | The earlier Osler research tree: the symbolic engine, the DKA world model, the eICU/MIMIC extraction and audit scripts, and the code the v10 lockbox froze. Formerly the root of this repository. | [`osler/AGENTS.md`](osler/AGENTS.md), [`osler/README.md`](osler/README.md) |
| [`demo/`](demo/) | The Oslian Demo (Flask + static front end): medication safety and FHIR stream monitoring. It loads `model/` through a pinned lock. Formerly the `oslian-demo` branch of WholeBody-JEPA. | [`demo/README.md`](demo/README.md), [`demo/docs/MODEL_RUNTIME.md`](demo/docs/MODEL_RUNTIME.md) |

All three histories are kept: `git log -- model/`, `git log -- osler/` and `git log -- demo/`
show each project's own commits.

## Working in each folder

Run each project from its own folder. Scripts and frozen evidence bind paths relative to
that folder (protocol hashes, report `bindings`, lockbox `frozen_code_sha256`), and those
paths did not change when the projects moved here.

```powershell
cd model; python -m scripts.validate_release          # model release check
cd osler; python -m pytest                             # Osler regression tests
cd demo;  .\scripts\start-demo.ps1                     # the demo (fixture mode)
```

Research documents in `model/` and `osler/` are indexed by topic in each folder's
`docs/INDEX.md`. A file that still sits at a project's root is bound by path from code or
frozen evidence; do not move it.

## The demo's model lock

`demo/backend/model-lock.json` pins the model by the last commit that changed `model/` and
by every artifact's SHA-256. Any later change to `model/` makes the demo's release mode
refuse to start until the lock is regenerated on purpose; see
[`demo/docs/MODEL_RUNTIME.md`](demo/docs/MODEL_RUNTIME.md).
