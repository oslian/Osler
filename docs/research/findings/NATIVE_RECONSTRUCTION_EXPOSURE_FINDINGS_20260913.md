# Native Reconstruction Training Exposure

## Question And Evidence

Did the selected checkpoints in the 35-target sampling pilot actually consume
the available patient-target supervision? Exact RNG replay uses the saved fit
sample ordering, event frequencies, sampling seed and selected epoch.
Evidence: `artifacts/native_reconstruction_exposure_20260913.json` and
`scripts/audit_native_reconstruction_exposure.py`.

| Arm and checkpoint | Updates | Unique fit pairs consumed / available |
|---|---:|---:|
| Event-weighted, selected epoch 4 | 256 | 124 / 879 |
| Balanced, selected epoch 2 | 128 | 115 / 879 |
| Event-weighted, final epoch 8 | 512 | 182 / 879 |
| Balanced, final epoch 8 | 512 | 369 / 879 |

Twelve targets received zero direct supervised draws by the selected
event-weighted checkpoint. Even its final checkpoint omitted five targets.
Balanced sampling covered all targets by its final checkpoint, but not all
patient-target pairs. Shared parameters can still change through other targets;
zero direct draws do not mean the entire encoder remained untouched.

## Interpretation

The pilot is not evidence of adequate optimization or convergence for every
target. This is a concrete exposure limitation, not proof that additional
training will improve heldout reconstruction or explain every previous failure.
Only 32 patients contributed fitting data; repeating their examples does not
increase independent patient support.

## Follow-Up And Boundary

`NATIVE_RECONSTRUCTION_FULL_COVERAGE_PROTOCOL_20260913.md` fixes a balanced
schedule that consumes every available fit pair each epoch and gives the
event-weighted control the same update budget. It preserves the previous
patient roles, examples, normalization and source-tune checkpoint selection.
Schedule and budget both change relative to the earlier pilot, so their
individual effects are not isolated. Training is pending completion as of this
entry. No release weights, serving behavior, intervals or promotion changed.
