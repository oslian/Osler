# Coverage Sampling: Complete Outer Development Contrast

## Scope

Five frozen outer folds, 1,800,214 patient-target pairs, 73,670 distinct people,
47 targets. Four arms: coverage real/process and original replacement
real/process, plus the fit-only population center. Same update budget and
observation contract; no post-selection tuning. Process controls zero numeric
inputs while retaining measurement structure, not a shuffled physiology control.

All output hashes, expected patient-target identities, labels and test roles
passed. Replacement predictions reproduce the original baseline exactly in all
five folds. Fold2 uses certified equivalent inference, not independent training.
The interrupted fold4 attempt had no completed artifact; the restarted identical
run completed 318474 pairs and exited successfully.

## Results

| Equal-target normalized MAE | Original | Coverage | Relative reduction |
| --- | ---: | ---: | ---: |
| Patient | 0.925567 | 0.920727 | about 0.52% |
| Provenance component | 0.901157 | 0.896169 | about 0.55% |

20/47 targets improve in mean at both levels. Mean improvement is not a
significant pass. The unchanged descriptive screen against process and
population controls rises from13 to17 targets. All original13 are retained.
Added: alkaline_phos, bicarbonate, heart_rate, serum_osmolality.
These four do NOT individually pass the incremental comparison with the old
real model. Crossing a screen threshold does not prove the change helped that
target significantly.

Only MAP passes the incremental paired screen against the old real model at
both patient and component levels with the existing47-target family correction.
Its raw MAE delta is -0.47955 (patient bootstrap95%CI[-0.51970,-0.43820]) and
-0.78874 (component CI[-1.23147,-0.37128]);6/36 components worsen.
MAP still fails the full base screen against process/population controls.
Therefore no target simultaneously establishes both a new incremental screen
win and the complete base screen in this experiment. No runtime promotion.

## Diagnosis

Matched-budget broader exposure modestly changes generalization but does not
uniformly repair it. The numeric-zero process model also improves on average,
so improved real-model error alone cannot establish stronger physiology learning.
The sampler changes both duplicate exposure and effective weighting; this is
not an isolated causal test of duplicated samples. Increased training exposure
is a contributing design factor, not a proved sole root cause.

Before further model changes, inspect which control blocks each failed target
and whether numeric information adds stable value beyond measurement structure.
Do not promote the new four based solely on crossing the descriptive threshold.

## Evidence and Limits

`python -m scripts.score_coverage_outer` generates
`artifacts/native_reconstruction_coverage_outer_evaluation_20260929.json`:
raw/normalized errors,5000 paired bootstrap replicates, fixed-family one-sided
tests, per-component harms, and retention of the original screen.
Inference is conditional/descriptive: training folds overlap and development
data have been repeatedly examined. This is masked CURRENT-value reconstruction,
not future forecasting, interval calibration or independent external validation.
Registry, serving weights and intervals remain unchanged.
