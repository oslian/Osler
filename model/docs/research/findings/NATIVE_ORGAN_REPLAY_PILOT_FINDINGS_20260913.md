# Native Anatomical Replay Learning Pilot

Completed both fixed8epoch arms,64fit/32tune/64new-source/64hospital patients,
one source-random eligible query each. Nine direct artifact bindings verified.
This is the first multi-patient real-label fit of the new observation replay
bridge, not a full-cohort experiment or a seven-seed promotion gate.

| Source-tune MAE | Identity graph | Canonical sparse graph |
|---|---:|---:|
| Epoch0 baseline | 8.34714 | 8.34714 |
| Epoch1 | 8.34968 | 8.34964 |
| Epoch8 | 8.41926 | 8.41953 |
| Selected epoch | 0 | 0 |

Both select the exact zero-start residual. Held-out selected predictions are
therefore identical to the frozen own-history baseline, not new confirmation of
learned graph efficacy. New-source MAE8.46155 versus persistence9.85938; hospital
10.72190 versus12.07813. With64patients, paired persistence comparisons cross0
in both roles. Do not substitute larger earlier-cohort significance for these
pilot results. Selected baseline/candidate equality is by construction.

The first observed failure is already source-tune selection: the trained
correction does not improve this small development set. This does not prove a
single biological/data/optimization cause. Training loss and epoch8 held-out
predictions were not retained; the selected checkpoint alone cannot diagnose
fit memorization, convergence or the trained graph's held-out gap. No test-set
tuning is authorized by this result.

Anatomical plausibility only defined the experimental sparse graph; no edge was
validated. This path retains all anatomical slots and scoped observation updates
but has no standalone proof the shared body state adds value. More patients and
matched training diagnostics remain required; no automatic success from scale.
No runtime, interval or promotion changes. Candidate checkpoints remain private.

Evidence: `artifacts/native_organ_replay_pilot_20260913.json`.
