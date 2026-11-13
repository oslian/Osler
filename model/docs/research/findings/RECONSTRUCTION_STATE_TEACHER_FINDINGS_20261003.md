# Shared-State Teacher Readability Does Not Repair Truth Accuracy

## Complete Fixed Experiment

All47 targets/all5 folds; 705 new readouts, 1,797,884 source query occurrences,
29,580 unique patients and720,470 patient-target pairs. Each original fit-only
raw265 HGB provides a fixed scalar teacher. Its predictions on fitting rows
train body64, process-only265 and within-component-shuffled-body64 students,
using the same100-iteration absolute-error budget. Encoder and inputs unchanged.

This differs from old forecast/latent-retention distillation: it tests whether
the same masked CURRENT body state can approximate the fixed raw readout's
function, with no raw numeric observations at student inference. Teacher fit
outputs are in-sample, NOT cross-fitted labels or independent physiological truth.
Original reference predictions, patient/component roles and hashes verify.

## Two Outcomes, Different Answers

| Equal-target macro MAE | Body trained on truth | Body trained on teacher | Raw teacher |
|---|---:|---:|---:|
| Against real truth, patient |0.857139|0.853788|0.825737|
| Against real truth, component |0.811241|0.811520|0.787947|
| Against teacher, patient |0.230677|0.197151|0|
| Against teacher, component |0.230200|0.193756|0|

Student imitation error decreases14.53%patient/15.83%component. All235fits
improve patient/bounded fitting imitation over the original body head;
232/235also improve component-equal fit imitation. This is not a stopped or
untrained optimization path. Query imitation is not exact: residual discrepancy
remains substantial, so complete input-information preservation is NOT proven.

23targets pass all four imitation comparisons against original body, process
student, shuffled-state student and fit-teacher-median at both levels:
albumin,alkaline_phos,ALT,AST,bicarbonate,BUN,calcium,creatinine,glucose,
heart_rate,hematocrit,hemoglobin,magnesium,o2sat,phosphate,platelets,potassium,
PTT,respiratory_rate,sodium,temperature,total_protein,WBC.

In contrast, ZERO targets pass the real-truth comparison against the existing
body-only head, ZERO against raw teacher, and ZERO the complete six-comparison
truth screen. Gains versus the older native head on ALT/AST/BUN/creatinine/Mg
are not an increment over the better body-only reference.

Among the23imitation passes,17have higher patient mean truth error and13higher
component mean truth error than body-only. These are mean harm counts, not23
independent trials or multiplicity-adjusted significance claims. Across all47,
21patient means and21component means worsen; the lists are not identical.

## Concrete Failure Cases

| Target | Body truth MAE (patient) | Student truth MAE | Body teacher MAE | Student teacher MAE |
|---|---:|---:|---:|---:|
| Albumin |0.340241|0.341758|0.171630|0.157286|
| Bicarbonate |0.608804|0.610201|0.166721|0.142416|
| BUN |0.614053|0.616881|0.234440|0.211600|
| Creatinine |0.873193|0.879645|0.298929|0.277007|
| Glucose |0.701064|0.701645|0.137954|0.116913|
| Hemoglobin |0.444886|0.449850|0.173782|0.147022|

BUN/creatinine imitation passes, but truth error increases at BOTH levels.
For creatinine the patient truth difference is+.006452, nominal95%CI
[+.004018,+.008820];10of12component means worsen. This is not only a marginal
positive improvement blocked by a conservative gate.

Troponin truth means improve relative to body-only but remain worse than native
and teacher; its component p=.00784 versus body misses the fixed family screen.
Ferritin also numerically improves body-only but still loses native. Neither
is promoted or used to select another loss after observing the source results.

## What Is Established, What Is Not

Established: a changed readout objective can recover more of the teacher's
predictable output from the same body64, with23held-component diagnostic passes.
That recovered approximation does NOT produce a stable truth-accuracy gain.
Thus the simplistic remedy "just teach the body head to copy the better raw
model" fails on this experiment. It is not a proof of no body signal.

Not established: whether irrecoverable compression, finite decoder capacity,
in-sample teacher bias or conditional distribution mismatch dominates. A smaller
teacher discrepancy alone does not identify which remaining discrepancy matters
to actual patient values. Neither a negative nor positive imitation result can
prove the encoder retained all relevant information.

There is a concrete mathematical failure mode even with an optimal teacher:
let60%of examples be groupA with Y=0(probability.51) or100(.49); groupB40%
always has Y=100. A group-aware absolute-error teacher predicts0forA and100forB
(truth MAE29.4). A less-informed constant model minimizing REAL truth chooses100
(MAE30.6). A constant student minimizing teacher-point MAE instead chooses0,
improves teacher imitation60->40, but worsens real MAE to69.4. Conditional medians
do not obey an iterated-expectation identity. A deterministic unit test verifies
this counterexample. It establishes a possible objective mismatch, NOT that
this exact generating process explains our clinical datasets.

## Decision

Candidate only. No new promotion, serving weight, interval, registry, forecast
or clinical claim. Do not conflate23imitation passes with23physiological targets
validated for use. Do not repeat point-teacher distillation as an untested fix.
The prior observation-skip result remains valid: explicit observed information
helps the body-only path, but a useful increment from body state beyond raw
observations remains unresolved. Any next representation experiment must retain
real-truth evaluation and matched raw/process controls, rather than optimize
and report only an auxiliary learning objective.

Evidence: `artifacts/native_reconstruction_state_teacher_assessment_20261003.json`
and five `native_reconstruction_state_teacher_fold*_20261003.json` reports.
17input/scoring/regression tests include the optimal-teacher counterexample.
All483files bound by the final assessment were independently rehashed.
All private predictions and fitted students stay ignored and local. Source
development has been repeatedly inspected; no external data was scored here.
