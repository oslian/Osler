# Pure JEPA Body-System Conversion Findings

Date: 2026-07-13

## What changed

The DKA path now has a saved neural target-head checkpoint, and all available
eICU body-system transition cohorts can use the same pure-JEPA architecture:

`measurement-pure values + masks + ages + causal history`
`-> shared GRU physiology encoder`
`-> target-specific neural heads`
`-> factual future state`

No hand-built belief feature is passed to this path. The model is trained as a
factual forecaster only; it does not estimate treatment effects.

## DKA checkpoint

The pure-JEPA DKA target-head checkpoint was trained from 8,050 usable rows and
989 stays on top of the frozen measurement-pure v5 representation. Across seven
patient-held-out seeds, 9/10 evaluated targets significantly beat persistence:

- glucose, pH, bicarbonate, anion gap, potassium, MAP, sodium, osmolality, and
  urine output passed;
- creatinine remained non-promoted and must use persistence.

The checkpoint is `dka_pure_jepa_target_heads.pt`.

## Body-system screen

The common trainer produced candidate checkpoints for 15 eICU transition
modules:

| Module | Targets | Rows | Single patient-held-out screen |
| --- | ---: | ---: | --- |
| AKI | 14 | 142,117 | 14/14 lower MAE |
| Acute neuro | 12 | 68,957 | 10/12 lower MAE |
| Cardiac injury | 18 | 73,851 | 12/18 lower MAE |
| Cardiovascular instability | 13 | 69,764 | 12/13 lower MAE |
| Coagulopathy/heme | 18 | 92,199 | 18/18 lower MAE |
| Electrolyte/acid-base | 17 | 76,193 | 17/17 lower MAE |
| Endocrine stress | 16 | 71,382 | 11/16 lower MAE |
| GI/pancreatic/nutrition | 20 | 68,315 | 12/20 lower MAE |
| Hepatic failure | 14 | 97,844 | 14/14 lower MAE |
| Immune/inflammatory | 16 | 78,842 | 9/16 lower MAE |
| Integumentary/skin wound | 18 | 72,215 | 17/18 lower MAE |
| Musculoskeletal/rhabdo | 16 | 74,041 | 12/16 lower MAE |
| Respiratory | 15 | 83,116 | 15/15 lower MAE |
| Sepsis | 18 | 126,399 | 16/18 lower MAE |
| Toxic/metabolic | 19 | 51,633 | 19/19 lower MAE |

This is a broad candidate screen, not a promotion claim. Each module used two
training epochs and one patient-held-out split with a capped cohort. It still
requires seven-seed target gates, hospital-held-out evaluation, conformal
interval calibration, and rollout stability checks before any module can be
promoted.

## Full gate follow-up

The full gate was then run for all 15 modules with seven patient-held-out
seeds, a hospital-held-out split, and split-conformal coverage. The number of
targets passing the patient gate was:

| Module | Patient-gate targets / total | Hospital split | Hospital-significant targets |
| --- | ---: | --- | ---: |
| AKI | 6 / 14 | run | 12 |
| Acute neuro | 4 / 12 | run | 8 |
| Cardiac injury | 3 / 18 | run | 4 |
| Cardiovascular instability | 5 / 13 | run | 6 |
| Coagulopathy/heme | 5 / 18 | run | 8 |
| Electrolyte/acid-base | 3 / 17 | run | 10 |
| Endocrine stress | 4 / 16 | run | 5 |
| GI/pancreatic/nutrition | 3 / 20 | run | 6 |
| Hepatic failure | 3 / 14 | run | 10 |
| Immune/inflammatory | 5 / 16 | run | 6 |
| Integumentary/skin wound | 6 / 18 | run | 8 |
| Musculoskeletal/rhabdo | 1 / 16 | run | 2 |
| Respiratory | 5 / 15 | run | 6 |
| Sepsis | 10 / 18 | run | 12 |
| Toxic/metabolic | 6 / 19 | run | 10 |

This resolves the previous uncertainty about whether the other systems could
run without hand beliefs. They can run, but only the target-level cells that
pass the patient gate should be allowed to move. The rest must remain
persistence or `needs_calibration_audit`. A hospital point estimate is not
enough to promote a target by itself.

The resulting fail-closed registry is stored in
`pure_jepa_gate_registry.json`. A target is allowed to move only when all three
conditions are true: patient gate, hospital gate, and seven-seed conformal
coverage gate. Otherwise its runtime status is `fallback_persistence`.

## Runtime integration

`pure_jepa_runtime.py` now loads a body-system checkpoint and the registry
together. It accepts only a caller-supplied pre-anchor history, produces the
neural candidate, and then applies the registry per target. A non-promoted
target returns the latest observed value with source
`persistence_fallback`; it never silently uses a candidate checkpoint just
because the checkpoint exists.

The runtime currently exposes the factual point estimate and gate provenance.
It does not fabricate lower/upper bounds: interval values remain
`needs_runtime_calibration_artifact` until serialized conformal residuals are
attached to the runtime. This keeps a passing audit from being confused with
an already calibrated production interval.

## Boundary

- Hand-built beliefs remain available for historical comparison and older
  validated runtime paths, but they are not inputs to the new pure-JEPA
  checkpoints.
- The DKA checkpoint is saved and loadable, but remains candidate-only.
- The eICU transition files do not establish causal treatment effects.
- Some targets are sparse; a lower single-split MAE is not enough to promote
  them.
- A checkpoint being loadable is not the same as being promoted. The registry
  is the authority for whether a target may move.
