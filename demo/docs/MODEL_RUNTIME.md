# Model runtime

The demo does not own a model. It points at one release of the oslian/Model
repository, checks that release on every start, and refuses to run when anything
disagrees. This document is how that works and how to change it.

## The shape

```text
oslian-demo branch of WholeBody-JEPA      (orphan, no research files)
  backend/model-lock.json                 the only release that may be loaded
  backend/research/oslian-model/          git submodule -> github.com/oslian/Model, branch main
      models/whole_body_joint_jepa.pt
      models/unified_shared_jepa_head_bundle.joblib
      models/multisource_shared_jepa_head_bundle.joblib
      models/unified_source_selector.json
      models/runtime_registry.json
      models/interval_calibrator.json
      models/event_risk_heads.joblib
      models/cross_target_nowcast.joblib
      models/cross_database_replication_registry.json
      models/external_validation_report.json
      models/FEATURE_CONTRACT.json
      models/MODEL_MANIFEST.json
```

oslian/Model is the team's private repository for the release, its source and its
weights. The demo pins one commit of it and records every file's SHA-256; it never
copies a weight into its own tree.

## Why the submodule stays one level deep

The submodule is a different repository, so it cannot contain this branch. The
demo still guarantees that a recursive clone stops one level deep:

1. `.gitmodules` declares exactly one submodule, with `branch = main`.
2. `setup-model-runtime.ps1` refuses to proceed if more than one submodule is
   declared, or if the locked tree carries a `.gitmodules` of its own.
3. `check-model-runtime.ps1` re-checks the second of those on every verification.

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
CRLF checkout fails the manifest. `setup-model-runtime.ps1` sets
`core.autocrlf=false` and `core.eol=lf` in the submodule and re-materialises the
files. Fix the checkout, never the check.

## The scripts

### `setup-model-runtime.ps1`

Run once per clone. Initialises the one submodule, verifies its remote URL,
checks out the locked commit, forces an LF checkout, verifies every file against
the lock, and proves the locked tree declares no submodule of its own. It
installs nothing, fetches nothing newer, updates nothing and pushes nothing.
`-Force` re-checks-out the locked commit, and is refused on a dirty submodule.

### `sync-model-research.ps1`

A dry run by default. Refuses to touch a dirty submodule; fetches; prints the
locked commit, the submodule HEAD, `origin/main`, the commits between them and
the files that differ under `models/`. It checks out nothing and rewrites no
lock.

`-Update` checks a candidate out, runs the runtime contract tests against it, and
restores the locked commit either way. A candidate that fails must not become the
demo runtime. A candidate that passes is still not the runtime: adopting it means
editing `backend/model-lock.json` deliberately and committing it together with
the submodule pointer.

### `check-model-runtime.ps1`

Full verification: git HEAD, remote, no self-submodule, every file hash, the
manifest and registry hashes, the interpreters, the module origins, then a real
server on a spare port answering `/api/model/health` and `/api/model/capabilities`
— so the final answer comes from the running system, not from the script's own
opinion. It stops the server again.

## Changing the model

Model and training changes belong on their own branch of oslian/Model.

```powershell
cd backend\research\oslian-model
git switch -c model-runtime-fix origin/main
# change, test
git commit -am "fix: ..."
git push -u origin model-runtime-fix
```

Never commit on a detached HEAD — the submodule sits detached at the locked
commit by design, so branch from `main` first. Once the model branch is
validated, record it on the demo branch in one commit:

```powershell
git add backend/research/oslian-model backend/model-lock.json
git commit -m "chore: adopt validated model runtime <sha>"
```

Never mix a model change into a demo commit: the demo branch records *which*
release it serves, never *what the release is*.

## Dependencies stay separated

`requirements.txt` and `pyproject.toml` at the root cover the demo only — Flask,
plus two optional LLM SDKs. The research code keeps its own
`backend/research/oslian-model/requirements.txt` (torch, pandas, scikit-learn,
…) and runs in its own interpreter, chosen with `-ReleasePython` /
`OSLIAN_WHOLEBODY_PYTHON`. Merging the two would make the demo uninstallable
without a training stack, and would let the demo process import research code by
accident — which is exactly what the split prevents.
