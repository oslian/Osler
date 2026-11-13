# Available Data Versus Actual Training Scope

Completed a hash-checked reconciliation of source patient shards, source role
isolation, saved epoch exposure counters and all39target support counts.
Expanded selection patients do not overlap either fit group or the original
selected source group. Two invariant tests pass. This is a support audit, not
an accuracy experiment or newly discovered reason for every failure.

## Incomplete Exposure

The matched trial intentionally held compute and target membership fixed.
At selected epoch7, dense targets saw3752distinct broad-source patients; at
final epoch8,4288. For example:

| Target | Available broad fit | Seen by final epoch | Final coverage |
|---|---:|---:|---:|
| BUN | 9664 | 4288 | 44.37% |
| Creatinine | 9662 | 4288 | 44.38% |
| Glucose | 9541 | 4288 | 44.94% |
| Heart rate | 10096 | 4288 | 42.47% |

Sparse trained targets CPK, fibrinogen, lactate, lipase, triglycerides and
troponin already consumed their entire available broad fit pools at least once.
Their failures cannot all be attributed to unvisited source patients.

## Six Targets Not Trained In This Trial

| Target | Broad fit | Original broad selection | Expanded broad selection |
|---|---:|---:|---:|
| Amylase | 359 | 5 | 249 |
| Anion gap | 401 | 69 | 2863 |
| Direct bilirubin | 2408 | 5 | 404 |
| BNP | 1666 | 1 | 251 |
| LDH | 294 | 5 | 179 |
| TSH | 983 | 3 | 306 |

Allsixhave source data now, but inherited33-target eligibility excluded them
from the matched trial. They are untrained, not demonstrated negative models.
Original selected-source fit support is only3foraniongap,15forLDH,23foramylase;
equal group replication of such small sets is not an appropriate way to pretend
both groups have adequate support. Unsupported group-targets need explicit
status and must not silently receive thousands of repeated draws.

## Selection Support Is A Separate Limitation

The fixed90broad selection patients provide just6fibrinogen,7lipase,12CPK,
14triglyceride and18troponin examples. Expanded source-only selection has
514/455/751/488/811 respectively, among3323patients. This does not prove their
old scores are wrong, but makes repeated optimization on the small selection
cohort an avoidable uncertainty. The earlier source comparison deliberately
held selection constant and remains a valid matched comparison on its scope.

## Next Full-scope Path

First evaluate already-saved epoch trajectories on the expanded SOURCE selection
cohort to quantify selection stability without new training or external tuning.
Keep the old90-person scores as a paired audit and keep all evaluation cohorts
out of selection. Then a separately frozen complete-source training protocol can
include supported new targets and guarantee actual broad-patient coverage while
reporting additional compute and sparse-target repetition. Neither step may
retroactively relabel reused evaluation data as an untouched lockbox or promote
an untested head. Full-site, control and calibration gates remain necessary.

Evidence: artifacts/native_reconstruction_complete_source_scope_20260919.json.
No runtime, weights, intervals or promotion changed.
