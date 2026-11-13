# Native real auxiliary supervision

14fits complete, seven real/shuffled auxiliary label arms. Same native inputs,
source selection and original controls. Actual source-fit paired labels replace
initial-output imitation; auxiliary loss reused unchanged with weight.25.

| Hospital target | Real MAE | Beats glucose-only /7 | Beats persistence /7 | Beats shuffled /7 |
|---|---:|---:|---:|---:|
| Glucose | 42.3962 | 1 | 7 | 0 |
| Magnesium | .1814 | 7 | 7 | 5 |
| Potassium | .3924 | 7 | 7 | 6 |
| BUN | 3.5303 | 7 | 6 | 5 |
| Creatinine | .2420 | 7 | 0 | 0 |
| Phosphate | .8632 | 7 | 0 | 0 |
| Calcium | .3685 | 7 | 0 | 0 |

Source-evaluation real-vs-shuffled gains: Mg6,K6,BUN5,Ca5,phosphate3,
creatinine2,glucose0. Hospital real-vs-shuffled harm:BUN1,creatinine1,Ca2;
others listed0. Thus improvement over untrained other heads is real but cannot
be equated with validated physiology or incremental cross-organ information.
Both learned auxiliary arms can absorb population corrections. Glucose gets
no significant real-versus-shuffled gain in either role. No full repair.

This is positive evidence against retaining inaccurate old outputs as the sole
supervision: several other targets improve versus glucose-only7/7, and some
beat persistence. It does not prove shared context beats each target's own
history, or that all outputs are preserved. A matched own-target comparison
is required before interpreting auxiliary gain as whole-body coupling.
Initial-other-head weakness is an intentionally explicit comparator limitation.

All effects exploratory seven overlapping patient-split CIs, not independent
replications. Same potassium-selected query cohort and known-time caveats;
no new external lockbox, intervals, runtime or promotion changes. Base-model
scope, observational only. Prior snapshot auxiliary failure remains valid.

Artifacts: `artifacts/native_truth_auxiliary_20260911.json`,
`artifacts/native_auxiliary_placebo_20260911.json`.
