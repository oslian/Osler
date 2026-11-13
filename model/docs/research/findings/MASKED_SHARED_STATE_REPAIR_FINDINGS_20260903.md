# Shared-State Masked Reconstruction Repair

## Decision

**Completed training and evaluation; no runtime promotion.** The incumbent still
has 28 movable forecast cells and 13 persistence cells. This experiment concerns
15 distinct same-anchor numerical targets, not 41 independent nowcast targets.

The frozen representation contains recoverable information, but changing its
readout or adding missingness markers does not make every target transferable.
Reconstruction improvement is not future-forecast improvement.

## Verified Code Defect

The inspected `JointWholeBodyJEPAWorldModel.forward` passed a predicted future
latent to `_decode_latent`. Its `reconstruction` output was compared with current
anchor labels by the training loop, while the masked reconstruction audit read
the same heads from `context_latent`. These are different latent semantics.

The code now explicitly passes `reconstruction_latent=context_latent`. Future
value heads still receive the original future latent. No layer shapes or saved
release weights changed. Tests require current reconstruction to be independent
of requested future horizon and verify the intended anchor-state readout.

This establishes a defect in the inspected training path, not proof that this
was the sole cause of every historical checkpoint failure. B also receives
matched masking supervision; this experiment does not isolate the code fix
from that additional training.

## Controlled Experiment

| Arm | Shared encoder | Current reconstruction head | Input contract |
|---|---|---|---|
| A | Frozen release | Frozen release | Neutral masks/ages |
| B | Same frozen encoder | Retrained on masked anchor context | Neutral masks/ages |
| C0 | Jointly updated context encoder | Retrained on masked anchor context | Neutral masks/ages |
| C | Same joint intervention as C0 | Retrained on masked anchor context | Explicit observed/missing mask, neutral ages |

C0 isolates joint training from the explicit missingness intervention. No new
latent dimensions, organ graph, treatment features or future-value model were
added. C changes actual missingness markers as well as synthetic masks; it is
not a claim that workflow information has been eliminated. Its existing organ
pooling also responds to the explicit mask, so this is an input-contract
intervention, not an isolated missing-token embedding experiment.

All arms use the same starting checkpoint, immutable scaler, target/sibling
groups, patient split, paired minibatches and four-epoch budget. Heads use AdamW
with learning rate 0.003; context parameters use 0.0001. Each epoch has eight
128-row updates per supported target. C0/C additionally use weight-10
distillation of unmasked context, belief, uncertainty and organ features into
the original feature space. Future heads, EMA target encoder and observation
process heads remain frozen. This is reconstruction fine-tuning of a JEPA
checkpoint, **not a new end-to-end future-JEPA training run**.

Training uses the 3h slice of the local eICU canonical event table: 3,291 patients,
1,668 discovery patients and 1,623 development patients. Fresh supported fitting
patients range from 970 to 1,591 per target. The original encoder may previously
have seen development patients; those scores are source diagnostics, not a new
unseen-patient pretraining test. No MIMIC or NWICU examples fit any candidate.

An initial pilot consumed the random number stream differently in B because B
does not calculate distillation. It is retained in `artifacts/masked_abc_20260903`
but **superseded for arm comparisons**. The corrected paired-minibatch run is
`artifacts/masked_abc_paired_20260903/report.json`. Hyperparameters and gates were
not changed. C0/C produced the same losses after this sampling repair.

## Evaluation Contract

- Truth is a current measurement no older than one hour, not a future label.
- Target, all of its history, and predefined related/sibling variables are
  removed before encoding. The groups include conservative renal and acid-base
  groups, not only exact algebraic duplicates. Groups are unchanged across arms.
- Remaining inputs must satisfy the existing six-hour as-of age limit. Labels,
  future treatment, hospital and care-unit identity never enter the encoder.
- One outcome-independent hash-selected eligible anchor per patient, up to
  4,000 patients per database/target. Full available within-stay history is
  retained before selecting anchors. This is not a full-row census.
- Compare with the frozen population median and three same-model wrong-patient
  derangements. Require seven overlapping patient split CIs and every supported
  care unit CI below zero. Units need 30 patients, with at least two units;
  splits need 100 test patients. These are nominal paired-bootstrap diagnostics.
- A/B/C uses one optimizer seed, not seven independent training runs. The
  following ensemble experiment separately tests seven optimizer seeds.
- NWICU lacks a compatible urine target. Unsupported is not failed.
- MIMIC and NWICU have already been inspected. Neither is an untouched lockbox;
  no multiplicity-adjusted confirmatory or clinical claim is made.

## A/B/C Results

MAE in physical units; smaller is better. This table reports current masked
reconstruction, **not future forecast accuracy**.

| Target | MIMIC A | B | C0 | C | NWICU A | B | C0 | C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bicarbonate | 4.1507 | 3.7596 | 3.5826 | 3.4970 | 3.8665 | 3.6907 | 3.5362 | 3.4550 |
| BUN | 14.8580 | 12.6966 | 12.6784 | 12.7376 | 14.3438 | 12.9647 | 12.6342 | 12.2514 |
| Chloride | 7.0268 | 5.1095 | 5.1988 | 5.1894 | 7.6492 | 4.7593 | 4.8867 | 4.8643 |
| Creatinine | 0.8397 | 0.7386 | 0.7547 | 0.7609 | 0.8265 | 0.7097 | 0.6813 | 0.6725 |
| Glucose | 37.8789 | 35.6682 | 35.7196 | 35.5804 | 39.9922 | 39.3232 | 39.3800 | 39.4331 |
| Heart rate | 14.2363 | 13.9531 | 13.9636 | 13.9361 | 14.3161 | 13.9379 | 13.9005 | 13.9113 |
| Hematocrit | 4.9311 | 5.0447 | 4.6988 | 4.7105 | 5.4395 | 5.2222 | 4.9977 | 4.9661 |
| Hemoglobin | 1.6939 | 1.6815 | 1.5767 | 1.5657 | 1.8781 | 1.8301 | 1.7347 | 1.7219 |
| MAP | 12.8321 | 12.6199 | 12.3068 | 12.3325 | 11.8072 | 11.6559 | 11.5868 | 11.6335 |
| O2sat | 2.2459 | 2.2297 | 2.2283 | 2.2151 | 2.4093 | 2.3819 | 2.3898 | 2.3765 |
| Platelets | 73.3027 | 70.8476 | 69.6601 | 70.0339 | 74.7810 | 71.8939 | 71.1015 | 71.4782 |
| Potassium | 0.4636 | 0.4425 | 0.4416 | 0.4399 | 0.4375 | 0.4133 | 0.4135 | 0.4129 |
| Respiratory rate | 4.4709 | 4.3085 | 4.3215 | 4.2793 | 4.6879 | 4.4668 | 4.4690 | 4.4443 |
| Sodium | 4.8180 | 4.0726 | 4.0799 | 4.1178 | 4.5670 | 3.9296 | 3.9375 | 3.9924 |
| Urine output | 64.3611 | 67.0094 | 65.6733 | 65.4322 | unsupported | - | - | - |

- B improves MAE on 13/15 MIMIC targets and 14/14 supported NWICU targets.
- C0/C improve 14/15 MIMIC targets and 14/14 NWICU targets, relative to A.
- **Full diagnostic passes:** A none; B/C0 creatinine on NWICU only; C none.
- B passes all seven patient splits for six MIMIC targets before the unit gate;
  C0/C pass eight. On NWICU, each trained arm has nine such targets.
- No target passes the complete screen in both databases. Better than A does
  not imply better than the population baseline or all controls.

For example, chloride improves greatly relative to the original readout, yet
B's MIMIC MAE 5.1095 still exceeds the population median MAE 4.9595. This is
partly repairing a poor readout, not establishing reliable cross-organ recovery.

Energy-entropy effective rank of masked context was also recorded per task.
Median ranks on MIMIC are B 14.95, C0 15.43, C 16.05; on NWICU they are 16.62,
16.68 and 17.72. This is not evidence of a simple rank-collapse explanation, but
rank alone does not establish semantic physiology or held-out mode stability.
No promotion threshold was retroactively chosen from these ranks.

## Ordinary Forecast Retention

The same trained candidates were evaluated with observed targets and the
existing per-cell source selector. MIMIC shared-head fit patients were excluded
using the release-bound patient exclusion cache. No source re-selection was
allowed. Current nowcast predictions were not fed into future forecasts.

All 41 catalog cells were accounted for: 38 numeric cells were evaluable on
MIMIC, 37 on NWICU. Two binary event cells need separate interval-event labels;
NWICU urine is unsupported; platelets@1h has no eligible unused MIMIC support.

**B forecasts are bit-for-bit identical to A on every evaluated row/cell.**
Only reconstruction weights changed, so the existing future feature path and
calibrators are untouched. C0/C do not have this protection:

| Diagnostic | C0 | C |
|---|---:|---:|
| MIMIC cells with positive pooled MAE-delta CI | 6 | 7 |
| NWICU cells with positive pooled MAE-delta CI | 8 | 14 |
| MAP@24h MIMIC delta / 95% CI | +0.0865 / [0.0373, 0.1392] | +0.5295 / [0.4067, 0.6535] |
| MAP@24h NWICU delta / 95% CI | +0.1627 / [0.1258, 0.2006] | +0.2646 / [0.1893, 0.3346] |

These are nominal regression diagnostics, not adjusted multiple-comparison
claims. Component distillation was not enough to preserve all old forecast
readouts. C also changes missingness support at inference, so the experiment
does not uniquely attribute its regressions to optimizer interference.

Existing intervals were deliberately not reused for modified C0/C representations
or any new masked nowcast. No new calibrated interval capability is claimed.

## Seven-Seed Readout Follow-up

After the paired run, a declared follow-up held the encoder completely frozen
and averaged seven reconstruction heads with equal, fixed weights. All seven
heads read the same context latent; there are not seven body models. A paired
capacity-matched placebo ensemble uses the same architecture, initial heads,
optimizer budget and labels, with patient contexts deranged during training and
independently deranged during evaluation. This is stronger than the earlier
same-model wrong-patient control. Population and the three original controls
are still required as well.

The source patient split is fixed at seed 73; optimizer seeds are 7, 19, 31, 43,
59, 71, 83. Each target/head has 32 updates of 128 examples at learning rate
0.003. All 15 target heads and their placebos were fitted before external
evaluation. No best seed, weight or target was selected using external errors.
Because the follow-up was motivated by already inspected results, it is still
exploratory external validation, not a new pristine test.

| Creatinine masked reconstruction | Original A | Ensemble | Population | Trained placebo |
|---|---:|---:|---:|---:|
| MIMIC, 4,000 patients | 0.8397 | 0.7502 | 0.7978 | 0.7893 |
| NWICU, 4,000 patients | 0.8265 | 0.7188 | 0.7747 | 0.7667 |

NWICU creatinine passes the complete diagnostic including the trained placebo.
Its pooled MAE delta versus A is -0.1077, 95% CI [-0.1145, -0.1014]; versus the
trained placebo, -0.0480 [-0.0535, -0.0423]. Six of seven individual optimizer
seeds pass the full NWICU population screen; the pre-fixed equal ensemble passes
the ensemble screen. This is not seven out of seven individually validated
models and is not a future creatinine@horizon promotion.

MIMIC creatinine improves overall, but the full screen still fails. In Neuro
Stepdown (52 patients), ensemble minus population delta is +0.0181 with CI
[-0.0164, +0.0512]. Against the trained placebo the same unit is significantly
worse: +0.0635 [0.0214, 0.1017]. Several other units have CIs crossing zero.
Therefore this cannot honestly be explained as only insufficient sample size.

No other target passes the complete two-database screen. Before unit gates,
six MIMIC and eight NWICU targets pass all seven patient comparisons including
the trained placebo. No thresholds were relaxed to convert these into passes.

## What Is Resolved

1. Current reconstruction and future latent semantics are now explicitly
   separated in the code; same-anchor heads are trained on their tested latent.
2. Full target/history/sibling masking is matched between fitting and testing.
3. Explicit missingness has been experimentally tested, not assumed helpful.
4. Frozen readouts improve several targets without perturbing deployed forecasts.
5. Seven optimizer seeds and an actually trained equal-capacity placebo have
   been evaluated, with both gains and regressions retained.

## What Remains

- Subgroup generalization, including genuine unit-specific error increases.
- Some masked relationships do not beat even a population prior, despite large
  improvement over A. Conditioning information removed by the conservative
  group mask may genuinely be important; failure does not prove impossibility.
- Joint representation updates change the features used by frozen forecasting
  readouts. Good nowcast does not guarantee forecast retention.
- Candidate nowcasts still lack compatible uncertainty calibration and a new
  untouched cohort. Temporal drift and masked-mode stability are not certified.

The defensible current choice is to retain the incumbent runtime, keep these
reconstruction candidates separate from deployment, and not claim that all
body states or future forecast cells now pass. Future follow-ups need a fixed
source-only development decision and genuinely new external evaluation, not
repeated tuning to these unit failures.

## Reproduce and Artifacts

Run from this repository with its installed dependencies:

```bash
python -m unittest tests.test_masked_abc tests.test_release_masked_cells
python -m scripts.train_release_masked_abc --output-dir artifacts/my_fresh_masked_abc
python -m scripts.audit_masked_readout_ensemble --output-dir artifacts/my_fresh_readout_ensemble
python -m scripts.validate_release
```

CLI input paths refer to local credentialed canonical tables and can be
overridden. Outputs must be fresh directories to prevent accidental overwrite.
The scripts preserve release hashes and write only aggregate reports and model
weights, not patient tables or identifiers. Source/external input hashes and
settings are recorded in the reports.

- `artifacts/masked_abc_paired_20260903/report.json`: all arm/target/unit CIs,
  rank diagnostics, and ordinary forecast retention results.
- `artifacts/masked_abc_paired_20260903/{B,C0,C}.pt`: candidate state dictionaries
  bound to the original checkpoint hash. C requires its explicit-mask contract;
  these are not drop-in replacements for files in `models/`.
- `artifacts/masked_readout_ensemble_20260903/report.json`: seven-seed and trained
  placebo results, including optimizer-level and unit-level failures.
- `artifacts/masked_readout_ensemble_20260903/readouts.pt`: only reconstruction
  readout weights, fixed seeds and equal-mean aggregation, bound to the frozen
  encoder. They consume the masked 64-D context, **not the 311-D forecast feature
  block**, and must not be used to fabricate an observed anchor or future value.

No release weights, source selector, production calibrator or GitHub upload was
changed by this experiment. No next-measurement-time model was trained.
