# MAP Native Event-Equal Training

## Intervention

Cloned all 21 original fixed HGB estimators, preserving features, source-only
channel support, patient partitions, labels, hyperparameters and evaluation.
Only fitting weights changed: each patient has equal total weight; within a
patient each distinct (stay, future-source-ID) has equal weight; queries sharing
that event divide its weight. Future values do not determine weights.
All original held-out MAEs reproduced, all parent bindings verified. Eight
focused tests pass, including patient/event hierarchy and stay namespaces.

## Completed Results

Seven-split median patient-equal MAP MAE, mmHg:

| Model | New source patients | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.44082 | 11.73923 |
| Original own | 8.78353 | 10.24211 |
| Original full | 8.74362 | 10.37322 |
| Event-equal own | 8.75946 | 10.23905 |
| Event-equal full | 8.74981 | 10.37609 |
| Event-equal support | 8.75408 | 10.36186 |

Patient-paired bootstrap: new full versus original full gains 0/7 in both
roles, harms 1/7 source and 0/7 hospital. New full versus matched own gains
0/7 both, harms 1/7 source and 0/7 hospital. Versus support gains/harms 0/7
both. Full versus persistence remains 7/7 gains in both roles.

## Decision

Event weighting is not a sufficient training repair for this MAP task. This
complements, rather than repeats, the earlier fixed-prediction scoring check.
It does not establish that repeated outcomes never matter. The weak generalizable
increment beyond own history remains unresolved; no sole biological cause is
established. Do not infer numerical physiology from support-control nonsignificance.

This is a raw fixed-lag tree diagnostic, not full anatomical JEPA training.
Seven overlapping splits and repeatedly inspected hospitals are not a fresh
external lockbox. No interval calibration, runtime changes or promotion.
Private model weights and row predictions remain outside version control.
Evidence: `artifacts/map_native_event_weight_training_20260912.json`.
