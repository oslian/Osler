# Epoch 10: Complete Paired Outer Assessment

All five frozen outer folds are complete: 1,800,214 patient-target pairs,
73,670 distinct patients and 47 masked-current reconstruction targets. This
is the prespecified epoch-8 to epoch-10 continuation, with no epoch selection
from outer results. Fold 2 reuses a certified equivalent fold-0 fit; the four
separately fitted folds overlap in training patients.

The interrupted fold-4 process had exited without a completed artifact. The
identical inference script was rerun and completed 318,474 pairs. All five
prediction files passed hashes, test-role membership, complete paired keys,
source provenance, exact fit-only scales and inference dependency checks.
Five synthetic tests cover the additional scoring-contract checks. These
checks found no version or provenance mismatch in the completed predictions.

## Results

| Equal-target normalized MAE | Epoch 8 | Epoch 10 | Relative reduction |
| --- | ---: | ---: | ---: |
| Patient | 0.920727 | 0.917355 | 0.366% |
| Provenance component | 0.896169 | 0.894102 | 0.231% |

22/47 targets improve their raw mean error at both weighting levels. Process
controls, which retain observation structure but zero numeric inputs, worsen
in aggregate: patient 0.970805 to 0.972799 and component 0.943497 to 0.945516.
The real model's mean improvement is small and not uniform.

The unchanged descriptive screen against process and fit-only population
controls passes 21/47 targets, previously 17/47. It retains 16 targets, loses
bicarbonate, and adds chloride, fibrinogen, platelets, PTT and uric acid.
None of those five additions also passes the incremental comparison against
its own epoch-8 real model at both patient and component levels. Crossing the
base screen therefore does not itself prove that continuation helped them.

ALT is the only target passing both the complete base screen and the
incremental epoch-10 versus epoch-8 screen under the fixed 47-target family
threshold (one-sided p < 0.05/47). ALT already passed the base screen at epoch 8.

| ALT raw MAE | Epoch 8 | Epoch 10 | Paired difference, bootstrap 95% CI |
| --- | ---: | ---: | --- |
| Patient, n=49,361 | 44.434185 | 43.584505 | -0.849680 [-0.922205, -0.776766] |
| Component, n=37 | 33.093955 | 32.518159 | -0.575795 [-0.884749, -0.265015] |

The component incremental p-value is 0.0005234. Of 37 components, 28 improve
and 9 worsen. Among the 27 components with at least 30 patients, 22 improve.
Deleting any one component leaves the mean incremental difference negative
(range -0.634581 to -0.514769). These are descriptive sensitivity checks, not
new eligibility rules or a reason to discard the nine harmed groups.

Bicarbonate's patient MAE worsens from 3.450134 to 3.462649 (paired difference
0.012514, bootstrap CI [0.006479, 0.018602]); component MAE rises from 3.977841
to 4.000399. It still has lower means than process/population, but its component
p-values do not pass the fixed family threshold. Longer training is therefore
not a universal repair.

## Remaining Failure Location

Of the 26 targets failing the base screen, 21 have lower mean error than both
controls at both levels but insufficient evidence under the fixed criterion.
Five have at least one nonnegative mean contrast: BNP, myoglobin, O2sat,
sodium and TSH. Overlapping blockers among the 26 failures are:

- Process comparison: 2 patient-level and 21 component-level failures.
- Population comparison: 4 patient-level and 23 component-level failures.

The principal measured bottleneck remains consistent incremental benefit
across source components. These counts do not distinguish limited precision,
measurement differences, case mix and model generalization as the sole cause.
The observed ALT response supports an optimization-budget contribution for
that target. It does not establish that more epochs will repair the others.
The small fixed source monitors also cannot replace the complete outer role:
they differ in both sampled identities and source distribution.

## Decision And Evidence

Keep epoch 10 as a research candidate. This is a successful conditional,
descriptive incremental ALT result, not a new forecast cell or runtime
promotion. The data have been repeatedly examined; there is no fresh external
lockbox, calibrated reconstruction interval, or clinical/causal claim here.
Release serving weights, registry and intervals are unchanged.

The full paired assessment is
`artifacts/native_reconstruction_continuation_outer_20260929.json`, generated
by `scripts/score_monitored_continuation_outer.py`. It contains all 47 targets,
5,000-replicate paired bootstrap intervals, per-component results, retained
and lost screens, and artifact bindings. `tests/test_continuation_outer_contract.py`
verifies rejection of misaligned rows, incorrect fit reuse, changed provenance,
modified dependencies and nonfinite predictions. Raw prediction rows stay
in the private data directory.

The next diagnostic should separate the remaining component-level uncertainty
from genuine mean harm using the fixed paired outputs and existing variance
audits before choosing another training change. No further epoch budget or
new target gate has been chosen from these outer results.
