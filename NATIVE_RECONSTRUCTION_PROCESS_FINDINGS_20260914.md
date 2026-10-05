# Trained Observation-Process Control

Training: `artifacts/native_reconstruction_process_training_epoch8_20260914.json`.
Evaluation: `artifacts/native_reconstruction_process_evaluation_20260914.json`.
Same39 targets, source patients, shared architecture, initialization, optimizer,
balanced schedule and eight epochs as the expanded real arm. Only numerical
input channels are neutralized from training onward; observed masks, ages,
event times and remaining input channels are retained. No target-label shuffle.

All8epochs completed. Source selection chose epoch5, tune2.038137 versus
initial2.085252; real selected epoch7 at1.921813. The process model is learned
under its actual input distribution, unlike prior frozen neutralization.

## Heldout Comparison

Real beats process with nominal95% paired patient intervals in12evaluation
targets and9hospital targets. Seven overlap:

**albumin, ALT, AST, bicarbonate, heart rate, platelets, total protein.**

There are no significant real-vs-process harms in either role at the nominal
threshold. Lack of significance is not equivalence.

The process control itself beats population for15evaluation and9hospital
targets. Thus population superiority or a patient-permutation diagnostic alone
does not isolate numerical physiology from observational-process information.

| Target | Evaluation real / process | Hospital real / process |
|---|---:|---:|
| Albumin | 0.5494 / 0.6515 | 0.5580 / 0.6668 |
| Bicarbonate | 0.9038 / 0.9829 | 0.8591 / 0.9614 |
| Glucose | 1.3315 / 1.3601 | 1.6921 / 1.7042 |
| Creatinine | 1.6250 / 1.6078 | 1.3345 / 1.3595 |
| Hemoglobin | 0.6522 / 0.6790 | 0.6979 / 0.7211 |

Errors are normalized. Glucose, hemoglobin, hematocrit and creatinine do not
show significant real-over-process gains in both roles. Their numerical
dependencies in the frozen-input test did not guarantee incremental utility
over a retrained process model. MAP still lacks reliable population gain.

## Diagnosis And Boundary

There is now evidence that remaining predictive performance is a mixture of
observation-process information and numerical inputs, rather than exclusively
either. Seven targets show incremental numerical value under this matched
single-split comparison. This does not prove an organ causal mechanism or
attribute a fraction of all error to workflow: masks/cadence can also reflect
underlying disease and the model is pretrained, not learned from scratch.

Do not report the14 earlier nominal population gains as14 proven physiology-only
targets. Do not discard observation validity metadata to remove workflow; doing
so would erase information needed to interpret real measurements.

Results remain single reused-split, unadjusted multiple comparisons, current
masked reconstruction. Repeat-split and external/process-shift validation are
still required. No new forecast cells, intervals, runtime change or promotion.
