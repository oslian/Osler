# MIMIC-to-eICU Site-Adaptive Rescue Findings

## Result

- Frozen zero-shot full external passes retained: `11/49`.
- Unresolved continuous cells tested: `38`.
- New site-adaptive 7/7 passes: `0`.
- Combined usable cells remain: `11/49`.
- This experiment does not alter the frozen runtime registry.

## What Was Tested

Two leakage-safe calibration variants were tested. Point recipes were fit on
MIMIC-IV only. Each of seven deterministic evaluations used a small set of
eICU hospitals for adaptation and at least ten different eICU hospitals for
evaluation.

1. Patient-grouped CV adaptive conformal was run over all 38 unresolved cells.
2. Leave-one-adaptation-hospital-out conformal with a richer severity scale
   model was run on the four cells closest to passing.

For the 25 interval-only cells, the existing frozen zero-shot point evidence
was retained and only the new interval evidence was re-evaluated. A changed
point source still had to pass both point and interval gates.

## Closest Cells

| Cell | Best interval splits | Point splits | Result |
|---|---:|---:|---|
| creatinine@6h | 6/7 | 7/7 | candidate only |
| BUN@24h | 6/7 | 7/7 | candidate only |
| creatinine@12h | 5/7 | 7/7 | candidate only |
| potassium@48h | 5/7 | 7/7 | candidate only |
| creatinine@48h | 4/7 | 7/7 | candidate only |

`creatinine@6h` failed one untouched-hospital split through broad
over-coverage (`0.949` patient-equal, `0.955` hospital-equal, `0.959`
care-unit-equal). The adaptation hospitals for that split simultaneously had
care-unit under-coverage, so one global interval adjustment cannot repair both
domains without evaluation-label tuning.

`BUN@24h` failed one split because late-time coverage was `0.941`, above the
predeclared `0.93` upper bound. The threshold was not relaxed.

## Treatment Schema Result

The six treatment-source cells were rebuilt using only target-permitted,
pre-anchor `hist_*` fields present with support in both databases. No treatment
source passed the full 7/7 point and interval gate. The shared schema mostly
contains treatment/evidence indicators; dose/rate fields are not consistently
available in both databases, so this does not establish dose-resolved transfer.

## Diagnosis

The remaining external failure is not one uniform problem:

- Point prediction transfers for many cells.
- Interval failures alternate between under-coverage near `0.85` and
  over-coverage near `0.95` across hospital splits.
- Care-unit and early/late calibration can move in opposite directions.
- Treatment-source selection is unstable when dose/rate semantics do not align.

This is cross-hospital observation-process and residual-distribution shift.
Further tuning on eICU evaluation labels would turn the external test into a
development set and is therefore prohibited.

## Decision

- Keep the original 11 full external passes.
- Keep all rescue results candidate-only.
- Do not automatically promote any site-adaptive cell.
- The next promotion test must use a new untouched database such as HiRID or
  AmsterdamUMCdb after a calibration policy is frozen without using that
  database's labels.

## Boundary

This is factual observation/forecast research only. It does not support causal,
treatment-effect, clinical-decision, or automatic runtime-promotion claims.
