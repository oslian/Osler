# Robust Readouts of the Shared Body State

Date: 2026-09-03. Run completed. Exploratory; candidate-only.

## Result

Three additional readout families were trained on the same frozen shared JEPA
state: median regression, Huber regression, and small absolute-error gradient
boosted trees. **No new target passes the complete masked-current diagnostic.**
The prespecified eICU-only method selector also adds no complete pass.

Some averages improve, but trained-placebo, care-unit and tail checks show why
those improvements cannot be promoted. Existing runtime artifacts are unchanged:
**41 catalog cells, 28 movable and 13 persistence**. This experiment does not
add future forecast cells or uncertainty intervals.

## What Was Tested

The [pre-run protocol](../../../MASKED_ROBUST_READOUT_PROTOCOL_20260903.md) fixed methods,
grids and evaluation before the main run. The preceding
[generalization experiment](MASKED_GENERALIZATION_FINDINGS_20260903.md) supplies
frozen neural and ridge reference weights; their external MAEs are reproduced
and checked before reporting comparisons.

- All 15 distinct numeric catalog targets; 15 evaluated in MIMIC, 14 in NWICU
  because NWICU urine has no compatible support.
- Same 1,736 eICU fit patients, 1,737 tune patients and 1,288 hospital-test
  patients across 55 hospitals withheld from this intervention.
- One hash-selected eligible anchor per patient; at most 4,000 patients per
  external target. Current truth must be <=1h old, other inputs <=6h old.
- Target, near-deterministic siblings and all their histories hidden before
  computing the shared 64-dimensional latent. No new input features.
- Seven patient-bootstrap bags per real ensemble and per independently trained
  placebo ensemble. Real/control bags use identical recipient rows and labels;
  controls receive deranged patient contexts before resampling.
- Source-only scaling and tuning. Real and placebo complexity are independently
  selected; a second placebo has exactly the real-selected complexity. New
  methods must beat both controls, not just whichever is easier.
- Existing population, three wrong-patient, seven patient-split and every
  supported-care-unit screens remain unchanged. No external tuning, threshold
  relaxation, selective patient exclusion or new deployment gate.

Every configuration completed without a fitting/convergence failure. Grids:
median alpha 0.01/0.1; Huber alpha 0.1/1/10 with epsilon 1.35; boosted trees
7/15 leaves, 128 iterations, minimum leaf 50, L2 10. The existing maintained
scikit-learn 1.7.2 implementations were used; no package added.

These are readout experiments, not full JEPA retraining. Freezing the encoder
avoids the representation drift that damaged existing forecasts in the previous
joint-update experiments. Models are not inserted into serving, and no
unvalidated reconstruction is fed into forecasting.

## Aggregate Comparisons

Numbers below count lower **current reconstruction MAE** than the previous
source-selected-duration neural ensemble, not promotion passes.

| New method | eICU hospital test | MIMIC | NWICU |
|---|---:|---:|---:|
| Median regression | 5/15 | 9/15 | 6/14 |
| Huber regression | 1/15 | 1/15 | 0/14 |
| Small nonlinear tree readout | 5/15 | 9/15 | 8/14 |
| Source-selected portfolio | 3/15 | 5/15 | 3/14 |

For MIMIC, the pooled paired-bootstrap CI favors median regression on 4
targets and trees on 6. For NWICU, it favors each on 3. None passes the complete
cross-patient/care-unit/control diagnostic. Huber is not a general improvement.

The reference fixed-32 neural head retains its earlier NWICU creatinine
diagnostic pass. It does not pass MIMIC completely and is not a new result.
All three new methods and the source selector have zero full passes in each
database, including the eICU hospital-test cohort.

Source selection was frozen before external evaluation:

| Selected source | Targets |
|---|---|
| Previous latent ridge | Bicarbonate, chloride, potassium, respiratory rate |
| Previous source-selected neural | BUN, creatinine, glucose, heart rate, platelets |
| New tree readout | Hematocrit, hemoglobin |
| New Huber | MAP |
| New median regression | O2 saturation, sodium, urine output |

This selector is not an oracle picking each database's best score. It can choose
a source-optimal method that is worse externally, as MAP illustrates. No
external failures were used to replace its choices after the run.

## Useful Improvement, But Not Sufficient

| Reconstruction | Previous neural MAE | New MAE | New method |
|---|---:|---:|---|
| MIMIC hemoglobin (g/dL) | 1.924 | 1.759 | Trees |
| NWICU hemoglobin (g/dL) | 1.889 | 1.859 | Trees |
| MIMIC hematocrit (%) | 5.672 | 5.400 | Trees |
| NWICU hematocrit (%) | 5.386 | 5.365 | Trees |
| MIMIC chloride (mmol/L) | 4.809 | 4.784 | Median |
| NWICU chloride (mmol/L) | 4.690 | 4.624 | Median |

For MIMIC hemoglobin, trees improve on the previous neural head by -0.164
g/dL MAE, CI [-0.187, -0.141]. However, the difference against independently
selected trained placebo is +0.014, CI [-0.007, +0.034]: no demonstrated
patient-specific advantage against that control. In CVICU it is actually worse
than placebo: +0.273, CI [+0.227, +0.321], with 891 patients.

NWICU hemoglobin improves over the old neural head by -0.030 g/dL, CI
[-0.046, -0.013], and beats placebo overall. But CTICU is worse than placebo:
+0.249, CI [+0.172, +0.322], with 302 patients; SICU is also worse. This is not
only a small-sample near miss.

MIMIC hematocrit is better than the neural head but **worse than trained
placebo even overall**: +0.126 percentage points MAE, CI [+0.067, +0.191].
Its exact-complexity placebo gives the same qualitative failure. Better than a
previous model does not by itself demonstrate useful patient-state information.

Median-regression creatinine still has a MIMIC Neuro Stepdown failure:
delta +0.085 mg/dL vs placebo, CI [+0.010, +0.159], across 52 patients.
NWICU chloride has a favorable overall placebo comparison, but several unit
intervals cross zero. Real subgroup harm and inconclusive subgroup evidence
remain different failure types.

## Tail Error Explains Part of the Average Improvement

Fit-only target 10th/90th percentiles define low/middle/high evaluation strata.
These boundaries never enter model inputs, selection by external outcomes or
patient routing. They are cohort-relative values, not clinical normal ranges.

MIMIC hemoglobin, tree readout vs previous neural:

| Stratum | Patients | Previous MAE | Tree MAE |
|---|---:|---:|---:|
| Low | 602 | 3.421 | 3.379 |
| Middle | 3,180 | 1.586 | 1.349 |
| High | 218 | 2.718 | 3.268 |

NWICU hemoglobin shows deterioration at both tails despite average improvement:
low 3.157 -> 3.260, middle 1.381 -> 1.299, high 3.340 -> 3.456 g/dL.
For MIMIC hematocrit, middle MAE improves 4.562 -> 4.108, but low worsens
10.391 -> 10.501 and high 8.598 -> 9.942 percentage points.

These are descriptive tail point estimates, not separately confirmed tail
significance tests. They nevertheless prevent the misleading claim that the
average gains represent uniformly better recovery. Extreme true measurements
must not be assumed to be measurement noise. Shrinking toward central values
can improve typical cases while concealing important individual differences.

## Interpretation and Boundaries

This narrows the question rather than solving all generalization:

1. Changing the readout's objective and nonlinearity can improve some pooled
   errors without changing the shared encoder.
2. Those gains still do not establish consistent patient-specific benefit;
   trained placebos and care-unit analysis expose failures missed by averages.
3. Some methods improve the middle of the target distribution at the expense
   of the tails. "More robust loss" is not equivalent to better physiology.
4. The remaining differences cannot be attributed uniquely to architecture,
   noise, treatment, labels or sample size by this experiment. There is no
   justified claim that another head alone will resolve them.

MIMIC and NWICU are previously inspected exploratory validation data. The source
hospital split is new-intervention-heldout, not necessarily unseen by historical
pretraining. Seven evaluation splits overlap. Many method/target comparisons
use nominal CIs without multiplicity correction. No causal, clinical, next-test
timing, new forecast or calibrated-interval claims follow.

## Artifacts and Verification

```bash
python -m scripts.audit_masked_robust_readouts
```

The CLI defaults use the same licensed local event files as the previous run.
Input hashes, previous report/weights hashes and protocol/code hashes are
recorded. Existing output directories are rejected; choose a fresh
`--output-dir` for a rerun. Previous head choices must have been source-selected
and their release binding must match. Reference MAEs are reproduced exactly
within numerical tolerance; serving hashes are checked again on completion.

- Full aggregate report: `artifacts/masked_robust_readouts_20260903/report.json`
  (`complete: true`).
- Candidate parameters and source-only choices:
  `artifacts/masked_robust_readouts_20260903/readouts.joblib`.
- Implementation: `scripts/audit_masked_robust_readouts.py` and
  `scripts/masked_robust_readouts.py`.
- Regression tests: `tests/test_masked_robust_readouts.py`.

73 related tests pass. Release validation remains `status: ok`, 41 catalog
cells, 28 movable, 13 persistence. No patient rows were placed in the repository,
no serving models changed, and no commit or upload was performed.
