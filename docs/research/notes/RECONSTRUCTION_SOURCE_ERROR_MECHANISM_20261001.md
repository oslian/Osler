# Source Group Error Mechanism

## Question and Scope

The full epoch-10 source assessment found recurring group-level mean harms.
Are these only constant bias, or does the model also fail to distinguish
patients? Diagnose all 47 masked current-reconstruction targets in all five
frozen source-selection folds. No new architecture, neural fitting, epoch
selection, outer-label calibration, runtime modification or promotion.

This differs from the prior BUN global/group offset-transfer and affine tests:
it is a within-source-component diagnostic on the complete latest model query
grid, not a new transferable calibration method. Those previous negative
transfer results remain valid. Component identity is allowed ONLY to stratify
this diagnosis, not as a new physiological encoder input or runtime selector.

## Paired Local Bias Check

Assign each patient to one of two halves using the fixed hash
`source-residual-v1:<patient>`. The assignment is the same across targets and
model folds. For each target and source component, estimate median residual
using one half and apply it to the other, then reverse. Held-out labels never
affect their own corrected predictions. Apply the same procedure to real,
process and population outputs. For a constant population output this becomes
a local median estimated on the other half, a stronger diagnostic reference.

Retain every group. A missing half yields null, never a fitted score on the
same rows. Summaries below focus on >=30-person components already showing
higher real MAE at BOTH epochs 8 and 10. These are posthoc descriptive strata;
counts are target-fold-component occurrences, not independent hospitals or
experiments. No statistical-harm or efficacy claim is made.

| Reference | Recurring harm occurrences | Real improves after centering | Still worse after centering BOTH arms |
| --- | ---: | ---: | ---: |
| Process control | 56 | 42 | 35 |
| Population reference | 69 | 56 | 45 |

Thus constant bias is present and often reducible locally, but this correction
is not sufficient. Among the 26 prior outer base-screen failures, the analogous
counts are 45 -> 28 remaining versus process and 56 -> 39 versus population.
This subset does not determine fitting or select a correction.

Median within-group Spearman correlation is 0.191 in the process-harm set and
0.156 in the population-harm set. Corresponding prediction/truth IQR ratios
are 0.289 and 0.369. Neither statistic proves information absence or a scaling
bug: conditional point estimates need not reproduce marginal outcome spread.
No arbitrary output inflation is justified by these ratios.

## Countercheck: Is Patient Assignment Useful?

After the fixed cross-fitted centering, preserve every prediction value and
its calibration half, but calculate the exact expected MAE under random
permutation among patients within that same half/component. This isolates
assignment at fixed prediction marginals. The expectation is the mean of all
pairwise absolute errors, computed with sorted sums rather than a quadratic
matrix. Tests match a brute-force Cartesian calculation, including ties and
large common offsets. Constant population predictions have zero assignment
benefit as expected.

| Reference | Remaining local harms | Real assignment better than permutation |
| --- | ---: | ---: |
| Process control | 35 | 29 |
| Population reference | 45 | 37 |

The median real assignment advantage across all supported recurring-harm
occurrences is 0.03064 normalized MAE units (process-harm set) and 0.02554
(population-harm set). This is an exact descriptive expected-permutation
contrast, not a p value, trained placebo, or proof of cross-database transfer.
Positive assignment benefit can coexist with MAE worse than a constant.

## What Is Established, and What Is Not

Established for this diagnostic:

- Local constant-bias correction helps many groups but leaves substantial
  residual mean harms even under matched local correction of the references.
- Most of those remaining groups still contain useful patient assignment in
  the real output. "The model contains no individual information" is not an
  adequate explanation of all these failures.
- Source population uncertainty and numerical calibration cannot be reduced
  to a single universal offset or a claim that all output variance must grow.

Not established:

- These local offsets will transfer to an unseen hospital, or should be served.
- Weak ranking originates specifically in the encoder rather than the head,
  observational support, model objective or conditional population differences.
- Increasing capacity, adding more heads or inflating predictions fixes it.
- Any new target has passed a promotion gate, or future forecasting improved.

The remaining actionable distinction is between usable patient information
that is poorly decoded and information already lost before the readout.
Before another refit, compare the current frozen readout with matched raw-input
and latent probes on the same source roles/groups, consulting the prior raw/body
probe and constrained-readout negatives. A probe must not be selected using
these posthoc harms or presented as a new external result. This assessment
narrows explanations; it does not claim the unique root cause or a full repair.

## Verification and Artifacts

Original full component MAEs reproduce to 1e-12, all parent hashes match, and
all 47 targets are retained. Thirty tests pass, including held-label isolation,
constant-bias recovery, reversed ranking, insufficient support, duplicate
identity rejection and exact permutation-loss computation. No patient rows or
identifiers appear in the public aggregate artifacts.

- `scripts/diagnose_source_group_residuals.py`
- `scripts/diagnose_source_assignment.py`
- `artifacts/native_reconstruction_source_group_residuals_20261001.json`
- `artifacts/native_reconstruction_source_assignment_20261001.json`

Both commands finished. Runtime weights, intervals, registry and GitHub release
are unchanged. This remains factual, candidate-only research.
