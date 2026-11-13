# BUN Sampler-Weighted Objective Reconciliation

Rescore the fixed selectedepoch7 checkpoint on the same529selected/9664broad
fit BUN pairs under patient-equal, expected bounded-sampler and realized
epoch1-7draw weights. Exact8epoch final RNG and all seen-patient sets reproduce.
Each group has3752BUN draws through selectedepoch7. No inference or training
repeated; uses private paired predictions from the completed fitting audit.

Body-off minus full loss:

| Group | Metric | Patient equal | Expected sampler | Realized draws |
|---|---|---:|---:|---:|
| Selected | MAE |-.014888|-.011938|-.010064|
| Selected | SmoothL1 |-.007650|-.006017|-.003992|
| Broad | MAE |-.030084|-.024860|-.019841|
| Broad | SmoothL1 |-.020583|-.017325|-.013471|

All three weightings retain the same sign. Bounded hospital sampling therefore
does not explain away the BUN penalty at this checkpoint. This is a deterministic
weighted-loss comparison, not a significance claim. It is NOT historical loss
at each optimizer step: all draws are rescored with final selected weights.

The shared multitarget objective is not BUN-only. The earlier all-target test
showed global dynamics deletion harms many targets. Remaining optimization
work must reduce the BUN contribution while preserving other target losses;
this result does not prove a feasible common descent direction, optimizer
convergence or unavoidable gradient conflict. Previous small gradient-probe
instability and global-branch deletion evidence remain applicable.

No runtime, weights, intervals or promotion changed. Exact sampler and
pair-set/draw-count assertions pass. Underlying weighted sampling and loss
helpers already have regression tests. Evidence:
artifacts/native_reconstruction_bun_weighted_objective_20260920.json.
