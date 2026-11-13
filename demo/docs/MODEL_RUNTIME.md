# Model runtime

The demo does not own a model. It points at `model/` in the same repository,
checks that release on every start, and refuses to run when anything disagrees.
This document is how that works and how to change it.

## The shape

```text
oslian/Osler
  demo/backend/model-lock.json     the only model state the release runtime may load
  model/models/                    the release: weights, heads, selector, calibrators,
                                   registry, MODEL_MANIFEST.json (SHA-256 of each file)
  model/whole_body_jepa/, model/osler_jepa/, model/whole_body_joint_jepa.py
                                   the code the worker imports, inside model/
```

The lock's `validated_commit` is the last commit that changed `model/`
(`git log -1 -- model`). In one repository HEAD moves with every commit, so the
model is identified by the commit that last touched it, plus every artifact hash.

## model-lock.json

`backend/model-lock.json` is the authority. It records the repository, the
validated commit, the release date, the manifest and registry SHA-256, every
model file's SHA-256 and byte count, the scikit-learn pin, the runtime contract
schema and the cell counts.

`WholeBodyRuntime` reads it at construction and passes its `validated_commit` to
the worker as the expected commit. It also checks the lock against the runtime
contract fixture — which was generated from that same release — and refuses to
start if the two disagree. Two independent records of the same release agreeing
is a check; one record is a hope.

## The fail-closed checks

Every one of these runs on every release start, in the worker, inside the
checkout. Any failure means the run is refused, not degraded quietly.

| Check | What it proves |
|---|---|
| `pinned_commit` | the checkout is at the commit the lock approves |
| `manifest` + `sha256:<file>` | every artifact is byte-identical to the manifest |
| `module_origin:osler_jepa`, `module_origin:whole_body_joint_jepa` | the modules loaded came from the checkout, not from a same-named package elsewhere |
| `scikit_learn_pin` | 1.7.2 — the version the serialized population anchors were fitted with |
| `registry_cells` | the registry has the cell count the manifest declares |
| `smoke_predict` | a real prediction returns all 41 catalog cells |

Output boundaries hold in every mode. A cell shows `lower`/`upper` only when the
release emits them from that cell's own calibrator with one of the
`interval_statuses_with_bounds` the lock names, the catalog marks the cell
`interval_capable`, and the bounds are finite and contain the point; any other
bound fails the run. Every other cell is `null` with an explicit
`interval_status`. Event cells are calibrated probabilities. Persistence is
labelled a baseline rather than a prediction, and `causal_claim_allowed` is always
`false`.

**CRLF is the usual failure.** The JSON artifacts are hashed byte for byte, so a
CRLF checkout fails the manifest. The repository root's `.gitattributes`
(`* text=auto eol=lf`) keeps them LF on every platform. Fix the checkout, never the
check.

## The scripts

### `check-model-runtime.ps1`

Full verification: the last commit that changed `model/` against the lock, no
uncommitted changes under `model/`, every file hash, the manifest and registry
hashes, the interpreters, the module origins, then a real server on a spare port
answering `/api/model/health` and `/api/model/capabilities` — so the final answer
comes from the running system, not from the script's own opinion. It stops the
server again.

## Changing the model

A model change is a commit to `model/`. It is not live until the lock is
regenerated, in a separate commit that touches nothing under `model/`:

1. Change and validate the model from `model/` (`python -m scripts.validate_release`), commit.
2. From `demo/`, regenerate the contract fixture through the worker with the new
   `git log -1 --format=%H -- ../model`.
3. Update `backend/model-lock.json` from `model/models/MODEL_MANIFEST.json`, run the
   tests in fixture and release mode, and commit lock and fixture together.

## Dependencies stay separated

`requirements.txt` and `pyproject.toml` at the root cover the demo only — Flask,
plus two optional LLM SDKs. The research code keeps its own
`model/requirements.txt` (torch, pandas, scikit-learn,
…) and runs in its own interpreter, chosen with `-ReleasePython` /
`OSLIAN_WHOLEBODY_PYTHON`. Merging the two would make the demo uninstallable
without a training stack, and would let the demo process import research code by
accident — which is exactly what the split prevents.
