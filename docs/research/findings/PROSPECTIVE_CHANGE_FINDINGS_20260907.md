# Frozen JEPA: Prospective Change and Risk-Binned CQR

## Decision

Completed the prespecified glucose@3h / MAP@3h experiment. Neither cell passes
the diagnostic screen. No release weights, source registry, or serving intervals
were changed. This is a candidate-only readout experiment, not a new body model.

MAP's median readout improves point MAE versus release and persistence on NWICU,
but does not beat the target-history-only control. Prospective change detection
does not improve. Glucose does not improve over release. These results do not
support the claim that the only remaining problem is choosing persistence.

## What Was Tested

Frozen shared JEPA plus the existing 311-feature target contract and seven causal
target-history summaries. New scikit-learn HGB readouts predict signed-delta
quantiles (.05/.50/.95) and probability of a research-sized future change.
This is not end-to-end JEPA retraining. Median changes can reverse direction or
exceed the old expert's magnitude; there is no [0,1] amplitude restriction.

Six methods: direct head, same-capacity shuffled-history head, same-estimator
history-only head, linear extrapolation, unchanged release, and persistence.
Current values remain the patient's own in the shuffled-history control.

All head fitting, Platt calibration, alarm cutoffs, risk bins and CQR radii use
eICU only. Candidate artifacts were saved and hashed before NWICU was read.
Input pairs require measured future events, target-specific timing tolerance,
and current measurement age <=1h. Future outcomes never select predicted-risk bins.

## Support

| Target | Fit patients / rows | Probability validation patients | Interval calibration patients | Patient test | Hospital test | NWICU test |
|---|---:|---:|---:|---:|---:|---:|
| Glucose | 690 / 2,262 | 278 | 138 | 265 | 106 | 500 |
| MAP | 1,287 / 6,175 | 487 | 246 | 499 | 258 | 500 |

Head-development roles are patient-disjoint; patients seen in held-out hospitals
cannot enter fitting or calibration. Primary evaluation uses one hash-selected
eligible anchor per patient. Chronological replay includes 1,944 glucose and
5,238 MAP NWICU anchors. Counts across targets are not unique-patient totals.

## Point Predictions

MAE in physical units; lower is better. The history-only control uses the same
HGB estimator settings but only seven patient-history features, not JEPA features.

| Target / dataset | Direct | Release | Persistence | History-only | Shuffled history |
|---|---:|---:|---:|---:|---:|
| Glucose / eICU patient | 29.04 | 29.90 | 33.51 | 27.94 | 30.61 |
| Glucose / eICU hospital | 32.80 | 34.14 | 35.65 | 31.64 | 33.69 |
| Glucose / NWICU | 30.07 | 29.91 | 31.72 | 29.93 | 30.44 |
| MAP / eICU patient | 9.58 | 9.97 | 10.66 | 9.54 | 9.83 |
| MAP / eICU hospital | 8.82 | 9.03 | 10.16 | 8.76 | 9.24 |
| MAP / NWICU | 9.06 | 9.50 | 9.94 | 8.98 | 9.33 |

The machine-readable report is authoritative for exact values.

NWICU paired patient-bootstrap differences, direct minus comparator:

| Target | Versus release, 95% CI | Versus persistence, 95% CI | Versus history-only, 95% CI |
|---|---|---|---|
| Glucose | +0.156 [-1.236, +1.551] | -1.654 [-2.777, -0.518] | +0.138 [-0.582, +0.883] |
| MAP | -0.439 [-0.702, -0.157] | -0.879 [-1.238, -0.514] | +0.083 [-0.098, +0.266] |

MAP improves over release by about 4.6% MAE, but its seven overlapping NWICU
evaluation resamples beat release in only 5/7 and shuffled history in 3/7.
These are NOT seven independent training runs. Glucose does not significantly
beat release or shuffled history on NWICU. No additional shared-state benefit
over the history-only readout is established for either target.

## When To Move

Research changes mean |future-current| >=20 mg/dL for glucose and >=10 mmHg for
MAP. These are not clinical alarm definitions. The classifier predicts magnitude
events, not their direction; signed-direction metrics are reported separately.

| NWICU target | Direct recall at <=5% FPR | Release-score recall | History-only recall | Linear-trend recall |
|---|---:|---:|---:|---:|
| Glucose | 10.1% | 20.2% | 9.3% | 10.9% |
| MAP | 12.1% | 13.3% | 14.5% | 14.5% |

This table is a held-out ROC ranking diagnostic, not a threshold fitted for
deployment. At the actual frozen eICU alarm cutoffs, the direct head's NWICU
FPR/recall are glucose 3.16%/5.67% and MAP 2.75%/5.78%.
There is no demonstrated gain in prospective detection at the fixed FPR budget.
Better MAP point MAE does not imply better early warnings.

Chronological replay groups repeated anchors by the same future measurement.
It cannot establish that alarms precede biological onset, or measure sensitivity
during unmeasured periods. Late-in-stay slices are not calendar-time holdouts.

## How Much Confidence

- Glucose: only 138 calibration patients in total; neither predicted-risk bin
  reaches the prespecified minimum of 100. New CQR intervals abstain everywhere.
  This is insufficient calibration support, not measured zero coverage. The
  unchanged release interval has 89.6% NWICU coverage on this sample.
- MAP: direct CQR NWICU marginal coverage is 90.0%, versus release 89.8%.
  Low/high predicted-risk coverage is 87.54% / 94.12%; the high bin misses the
  project's [87%,93%] operating band on the conservative side.
- MAP source hospital coverage is 93.41%. NWICU care-unit coverage includes CICU
  80.65% (31 patients) and NSICU 84.31% (51 patients). These small-slice estimates
  have wide CIs; nevertheless they do not satisfy the required point-coverage gate.
- MAP average interval width is 40.02 mmHg versus release 39.09 mmHg, about 2.4%
  wider, not narrower. Central 90% interval score is 60.26 versus 63.95 (point
  estimates only, no paired significance claim for this score).

Stable-case NWICU direct MAE minus persistence is +3.756 mg/dL for glucose
(95% CI [2.631,4.940]) and +0.288 mmHg for MAP ([-0.098,0.659]). The predeclared
upper-CI tolerances are 1 mg/dL and 0.5 mmHg, respectively; neither passes.
Thus there are several substantive failures, not just a marginal coverage miss.

## Interpretation and Boundaries

The signed-delta implementation works, and the old amplitude restriction is
removed. That did not solve early change recognition or establish that the
frozen shared representation adds predictive information beyond personal history.
MAP's modest point improvement warrants preserving this candidate, not promotion.
Glucose's risk-binned calibration needs more eligible source calibration patients
under a separately fixed protocol; do not lower support thresholds after testing.

All evaluation is conditional on a later measurement being available. The original
encoder's training-patient overlap has not been re-audited. NWICU has been examined
in prior research, so this is transfer validation, not a pristine external lockbox.
Full seven-training-seed, calendar-time and original runtime gates are not complete.
No causal, clinical, next-measurement-time, or onset-detection claim is allowed.

## Artifacts and Reproduction

- Protocol: `PROSPECTIVE_CHANGE_PROTOCOL_20260906.md` (frozen before execution).
- Implementation: `whole_body_jepa/prospective_change.py`.
- Runner: `scripts/audit_prospective_change.py`.
- Aggregate report: `artifacts/prospective_change_20260906/report.json`.
- Frozen source-stage report: `artifacts/prospective_change_20260906/development.json`.
- Candidate heads: `artifacts/prospective_change_20260906/candidate_heads.joblib`.
- Regression tests: `tests/test_prospective_change.py`; full suite 220 tests passed.

From repository root, with existing dependencies and source parquet files:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python -u -m scripts.audit_prospective_change --output artifacts/prospective_change_replication
python -m unittest discover -s tests -q
```

Use a new output directory; completed runs are never overwritten. Reports contain
aggregates and hashes, not patient rows. Candidate serialization requires the
existing scikit-learn version and matching frozen release; do not load untrusted
joblib files. Final checks verified unchanged release and candidate hashes.
