# Persistent Physiology State: Implementation and Pilot Findings

## Objective

Replace independent horizon guesses with one continuously evolving patient
state. Every target head should read the same body trajectory while organ
states exchange information only through an explicit sparse graph.

This remains factual research forecasting. It does not model treatment effects
or authorize clinical decisions.

## Implemented

- `PersistentPhysiologyState`: body latent, organ latents, organ uncertainty,
  and elapsed time.
- Predict-update observation update with explicit organ presence.
- Bounded recurrent state transition with at most one-hour internal steps.
- Sparse anatomical/functional organ graph.
- One shared future body trajectory consumed by all target-specific JEPA
  heads.
- Residual gate around the existing JEPA future latent. The new state must earn
  a departure instead of replacing the validated path at initialization.
- Exact linear conservation projector for explicitly defined hidden stores.
- Derived anion-gap and osmolality consistency audit.

Derived laboratory equations do **not** overwrite forecasts. A first pilot
showed that direct overwrite harmed anion gap because ICU inputs are
asynchronous and database definitions differ.

## Understanding Tests

| Test | Mechanical result | Capability conclusion |
|---|---|---|
| Mask one organ and use connected organs | Passed structural test | Information can cross the graph; real per-target recovery still needs external gates |
| Compare direct and staged rollout | Passed exactly | The recurrent state is path-consistent at the configured internal step |
| Generate all targets from one state | Passed integration test | All heads consume one shared body trajectory |
| Perturb a connected source | Connected organ state changes | This is model sensitivity, not a causal treatment claim |
| Enforce a declared linear total | Passed to numerical tolerance | Explicit hidden-store constraints can be exact |

Six new unit tests cover these properties.

## eICU Bounded Pilot

Data: 100 eICU patients from the canonical event ledger, with a fixed 70/30
patient split. Base JEPA and persistent-state JEPA used identical seeds,
training rows, targets, robust scaling, and three epochs.

### Direct replacement

Replacing the future latent outright caused broad negative transfer. More
epochs did not rescue it. This design was rejected.

### Residual state, conservative gate

- Evaluated cells: 114
- Cells with a lower point MAE than base JEPA: 67
- Cells lower than both base JEPA and persistence: 23
- Dense cells (at least 100 rows): 16/33 lower than base, 8/33 lower than both
- Learned state gate: 0.0299
- Median dense-cell delta versus base: approximately zero

The positive count is not sufficient evidence. Most changes are very small;
the candidate has not demonstrated a meaningful general advantage. It remains
`candidate_only`.

Artifacts:

- `artifacts/persistent_physiology_state_pilot.json`
- `artifacts/persistent_physiology_state_pilot_3epoch.json`
- `artifacts/persistent_physiology_state_residual_pilot.json`
- `artifacts/persistent_physiology_state_gate2_pilot.json`

## Current Verdict

The missing architectural object now exists: one persistent, auditable body
state with organ states and recurrent evolution. The pilot does **not** yet
show that this object improves real whole-body forecasting enough to replace
the current target router.

The next valid evidence step is a full multi-seed patient/hospital/time gate
with target-level ablations. Until then, the release must remain unchanged.
