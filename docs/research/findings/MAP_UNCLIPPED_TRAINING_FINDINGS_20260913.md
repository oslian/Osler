# MAP Clock Retraining

## Seven-Split Completion

All 21 own/full/control fits completed, with 98 direct report dependencies
hash-verified. Same protocol and source-only settings, no post-seed7 changes.

| Seven-split median MAE | Original new patients | Corrected | Original hospital | Corrected |
|---|---:|---:|---:|---:|
| Own | 8.84140 | 8.96478 | 10.21067 | 10.26613 |
| Full | 9.22298 | 9.22632 | 10.57510 | 10.60183 |
| Control | 9.23789 | 9.23419 | 10.61140 | 10.63245 |

Corrected versus original: zero significant gains for all arms in both roles.
Own has 3 source/2 hospital harms; full 1/1; control 0/2. Corrected full versus
corrected own has no gains, 7 source and 5 hospital harms. Full-control has
no significant gains or harms in either role, not equivalence proof.

History clock clipping is therefore not a sufficient accuracy repair under
this fixed training policy. The representation correction remains valid, but
its impact on optimization and clock-feature use is not isolated. The wider
historical-path gap persists; no unique underlying mechanism has been proven.
Do not rerun identical clock repair as an untested explanation. No runtime,
calibration or promotion changed; reused evaluation is not a fresh lockbox.

Evidence: `artifacts/map_unclipped_training_{seed}_20260913.json`, seeds
7,19,31,43,59,71,83. The initial pending status below is superseded.

## Historical First Split

Seed7 own/full/control matched retraining completed under the frozen protocol.
All report dependencies verified. Unbounded history time only; no changes to
query, labels, other numerical observations or training settings.

| Arm | Original new-patient MAE | Corrected | Original hospital MAE | Corrected |
|---|---:|---:|---:|---:|
| Own | 8.80806 | 8.86079 | 10.21067 | 10.37726 |
| Full | 9.16712 | 9.18174 | 10.56384 | 10.65343 |
| Control | 9.10062 | 9.12843 | 10.54625 | 10.58307 |

No corrected arm significantly improves its original version. Hospital own
change +0.16659 CI[+0.04915,+0.29079], full +0.08959 CI[+0.00675,+0.17394];
control change +0.03682 CI[-0.04325,+0.11690]. New-patient differences all cross0.
Corrected full remains worse than corrected own: new-patient +0.32095
CI[+0.19662,+0.43715]; hospital +0.27617 CI[+0.00091,+0.54327].

This one-split result does not support clock clipping as a sufficient repair
of the expanded-history gap. It is not a seven-seed verdict, convergence test,
proof timing unimportant, or evidence to retain a known timing limitation in
future scientifically faithful models. The changed history time distribution
and finite optimization are part of the fixed intervention.

Remaining six seeds have not run. No runtime, calibrator or promotion changes.
Evidence: `artifacts/map_unclipped_training_7_20260913.json`.
