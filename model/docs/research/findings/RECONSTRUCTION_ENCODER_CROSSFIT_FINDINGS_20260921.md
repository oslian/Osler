# Fresh Encoder Cross-Fit: Completed Outer Evaluation

## What Was Actually Tested

All five patient/hospital-component folds, both real and process-only arms,
eight fixed epochs. 72 arm-epochs were executed, another8 were reused only after
bitwise-equivalent full rerun evidence. Every trained arm has283008updates.
No outer score selected epochs, parameters or targets. Fresh learned parameters
and fitting-fold normalization replace inherited learned release weights and
pooled preprocessing. Variable/module architecture stays fixed.

This is target-and-sibling-masked CURRENT reconstruction, not future forecasting,
personalization, causal learning or a claim to replace the release. The future
JEPA latent loss is not trained by this current-reconstruction experiment.

Outer data:247611patient-target pairs from10717patients,33trained targets.
247610pairs produce both predictions; one empty supported history abstains and
stays in the denominator of coverage. The source partition had10737patients;
20have no labels among these33targets, not performance-based exclusions.
The compared baseline is a fitting-population center, NOT persistence, because
the target's own value/history is hidden in this task.

## Results

Target-normalized macro patient MAE (lower is better):

| Scope | Real physiology | Process-only | Fit population |
|---|---:|---:|---:|
| All |1.673005|1.748309|1.726672|
| Selected |1.874418|1.964726|1.954685|
| Broad |1.663383|1.738083|1.716379|

These units invert the fold scaling back to the earlier common normalized
scale. They are not percentages or clinical raw units. Real numerically beats
both controls in patient MAE for22/33all,19/33selected,23/33broad targets.
Component-equal numerical wins are23/33all,17/33selected,24/33broad.

Four targets pass the fixed descriptive patient/component screen against BOTH
controls in all and broad scopes: albumin, BUN, hematocrit, total protein.
Selected scope has0passes, and the intersection across selected/broad is0.

| Broad target | Real MAE | Process MAE | Population MAE |
|---|---:|---:|---:|
| Albumin |0.567691|0.782789|0.847934|
| BUN |0.899561|0.962809|0.955296|
| Hematocrit |0.744992|0.793339|0.848757|
| Total protein |0.550021|0.720869|0.838340|

The screen requires>=30patients,>=3whole-history hospital components, negative
patient/component means and one-sidedp<.05/39 against each control. It remains
DESCRIPTIVE DEVELOPMENT evidence: crossfit training sets overlap, fold0/1 share
one fitted function, and the entire source study has been inspected in prior
project development. Bootstrap intervals condition on fixed predictions and
do not capture retraining uncertainty. No external lockbox or full promotion.

## Where Failure Remains

| Scope | Screen passes | All comparison means improve but significance fails | At least one non-improving mean |
|---|---:|---:|---:|
| All |4|17|12|
| Selected |0|15|18|
| Broad |4|18|11|

All targets meet the overall scope's minimal support for this classification.
Insufficient per-fold labels noted by preflight remain real; pooling outer
predictions is not evidence that every fold independently validates a target.

Concrete distinctions:

- Selected albumin improves patient MAE by.07842 against process and.25149
  against population, but componentp=.0260/.0491 does not meet.05/39. This is
  insufficient multiplicity-adjusted evidence, not observed mean harm.
- Selected BUN improves by.03595 against process, but patient95%CI
  [-.07496,.00262] and componentCI[-.06649,.03391] cross zero.
- Selected total protein improves patient MAE by.16948 against process while
  the equal-component difference is+.11734: this is heterogeneous benefit,
  not only sample size or an excessively strict switch.
- Glucose selected real1.20584 is worse numerically than process1.18923 and
  population1.19437; broad real.96916 is also worse than population.96221.
  The experiment does not establish glucose reconstruction gains.
- Broad creatinine and PTT improve both means against both controls, yet one
  componentp=.001943/.002434 respectively exceeds.05/39. Do not retroactively
  loosen the rule or describe them as full passes.

## What Is And Is Not Localized

The new experiment removes the need to inherit a learned encoder that already
saw the outer source patients. Physiological numbers still provide some
held-out reconstruction information beyond measurement process. Therefore the
representation is not uniformly useless. Conversely, fresh initialization,
fold-local inputs and strict separation do not yield universal transfer: these
repairs are not sufficient to explain/fix every failure.

There is no matched warm-start arm under this exact protocol, so comparisons
to older reports cannot attribute improvement to reinitialization alone.
Eight fixed epochs may leave optimization limitations. Raw information limits,
clinical/measurement heterogeneity and representation limits are not uniquely
separated by this experiment. Root cause is still not a single proven defect.

Masking follows the existing `related_siblings` contract. Hematocrit hides
hemoglobin; BUN hides creatinine, glucose, sodium, osmolality and urine-related
siblings. Albumin and total protein are NOT mutual siblings in that contract
and remain allowable mutual inputs. Their relationship is not, by itself,
evidence of cross-organ understanding; a protein-panel-specific audit would
be needed before making that stronger claim. Do not call all four results
new independent-organ abilities or runtime forecast cells.

## Decision

Keep candidates separate from release. Do not add heads or loosen gates based
on these outer results. The next diagnostic must distinguish weak evidence
from actual sign reversals and account for source group and hospital support;
the fixed outer predictions now permit that analysis without retraining.
Any changed method would require new validation, not reusing these outcomes
as an untouched test. No new promotion, registry change, interval or clinical
claim follows.

Fifteen regression tests pass. All ten checkpoints have verified report hashes,
finite tensors, matched per-fold arm RNG/medians/dependencies and epoch8 state.
Both final-fold jobs and full inference exited normally. `git diff --check`
passes. Aggregate evidence is
`artifacts/native_reconstruction_encoder_crossfit_evaluation_20260921.json`;
patient predictions remain in ignored private storage.
