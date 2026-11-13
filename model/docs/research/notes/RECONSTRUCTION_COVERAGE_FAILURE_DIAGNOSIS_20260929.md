# Where the Coverage Candidate Fails

This diagnoses the completed47-target current-reconstruction experiment without
changing models, labels, support or the .05/47 descriptive screening threshold.
It is not a future-forecast diagnosis or a runtime promotion decision.

## First Failed Comparisons

-17 pass all existing descriptive comparisons.
-25 have lower mean errors against BOTH controls at BOTH patient and component
levels but fail at least one family-adjusted test.
-5 have at least one mean that is not better: BNP,myoglobin,O2sat,troponin I,TSH.

Among30 failures,29 fail real versus numeric-zero process at component level,
23 fail population at component level,5 population at patient level and2 process
at patient level. Counts overlap. Component means weight provenance connected
components equally, not hospitals or rows. No claim that each component is a
distinct independent biological population.

## MAP Explains the Distinction

MAP improves over the original model, but relative to its matched process-only
control its patient MAE delta is -0.19663, while component delta is -0.01653
with bootstrap95%CI[-0.15996,0.15584]. Its improvement is not established across
source components. A better old-model comparison does not establish reliable
use of numeric physiology beyond measurement structure.

## What Is and Is Not Located

Surface: the candidate remains unpromoted.
Direct statistical blocker: cross-component incremental numeric information,
not predominantly patient-average error. Five targets additionally have genuine
mean-performance failures and must not be treated as merely underpowered.
Root mechanism remains unresolved: heterogeneity, measurement support, model
misuse of values, and limited independent component information can all produce
these observations. Non-significance alone does not prove low power or absence
of signal. Numeric-zero controls retain masks/ages and other native structure;
their performance is not evidence that physician behavior causes the result.

## Next Discriminating Work

Use source-only predictions to ask whether numeric advantage changes with
observed input support/freshness, and compare its sign/stability with the outer
groups descriptively. Do not fit a selector on the outer failures. If adequate
numeric support still fails, inspect matched value-shuffle controls before
adding capacity. If support explains failure, test a prespecified support-aware
representation in source data, without dropping difficult targets or relaxing
the acceptance threshold. Independent confirmation is still required.

Artifact: `artifacts/native_reconstruction_coverage_failure_diagnosis_20260929.json`.
Reproduce: `python -m scripts.diagnose_coverage_failures`.
Three unit tests cover mean priority, multiple-test threshold and missing support;
the run also asserts exact agreement with every original screening decision.
No serving weights, registry or calibration changed.
