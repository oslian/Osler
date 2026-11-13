# Seed31 Reconstruction Replication

Training and evaluation completed on 2026-09-15, without interruption.
The frozen experiment identifier retains its 20260914 suffix:

- `artifacts/native_reconstruction_replication_v2_31_epoch8_20260914.json`
- `artifacts/native_reconstruction_replication_v2_31_evaluation_20260914.json`

## Fixed Comparison

Used the unchanged V2 runner and evaluator. All three arms completed eight
epochs, 288,192 updates each, with all 26,059 eligible fit patient-target pairs
consumed per epoch. Training/evaluation dependencies and checkpoint selected
scores were verified against hashes. No old shard-role assignments or seed7/19
fitted predictions were reused. The split changes initialization/sampling as
well as fit/tune/evaluation patient membership, not only patient composition.

Thirty-eight targets were trained. Anion gap remains explicitly unsupported
because source tune support is 26 patients, below the fixed minimum of 30.
It was not silently replaced by another variable or promoted from prior seeds.

| Arm | Selected epoch | Source tune normalized MAE |
|---|---:|---:|
| Real numbers and observation metadata | 6 | 2.177559 |
| Trained observation-process only | 8 | 2.296804 |
| Fit-label derangement placebo | 1 | 2.341291 |

Common initial score: 2.341645. Different source patients and the unsupported
target make these absolute scores incomparable to prior splits. Process selects
the last epoch; equal fixed exposure does not demonstrate convergence.

## Heldout Results

Evaluated 14,758 patient-target pairs. Real beats process under nominal 95%
paired patient intervals for 19 evaluation and 12 hospital targets; ten overlap:
albumin, ALT, AST, bicarbonate, chloride, creatinine, heart rate, magnesium,
PTT, total protein.

Requiring simultaneous superiority over population, process and placebo in
both roles leaves nine: albumin, ALT, AST, bicarbonate, chloride, creatinine,
heart rate, PTT, total protein. Magnesium does not meet the combined criteria.

The intersection of these combined comparisons across completed seeds7/19/31
remains four: albumin, AST, bicarbonate, total protein. Creatinine now meets
the comparisons in seeds19/31 but did not in seed7. This is not 7-seed promotion.

Real significantly harms the hospital bilirubin process-only comparator.
No nominal harms versus population or placebo were found in this split; absence
of a significant difference does not establish noninferiority or equivalence.
Hospital fibrinogen/LDH/TSH remain unsupported at 26/27/17 patients. All other
evaluated targets meet the minimum; the excluded source anion gap is still
part of the original catalog, with unavailable evaluation rather than a pass.

## Interpretation

Numerical shared-state information has repeatable utility for a subset under
current-value masked reconstruction. It is not universally useful: winners
and occasional harms change across splits. The same 194 hospital patients are
reused, and patient populations overlap across fits, so these are not three
independent studies. Comparisons are unadjusted for multiplicity and do not
erase the earlier multiplicity-sensitivity concern.

Seeds43/59/71/83 remain pending under the fixed protocol. Neither future-value
prediction nor conformal coverage is evaluated here. No new runtime weight,
registry, interval, serving change or promotion. Raw rows and checkpoints remain
private; only aggregate results are recorded.
