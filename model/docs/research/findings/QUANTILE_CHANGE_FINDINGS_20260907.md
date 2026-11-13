# Quantile Distribution and Pooled CQR: Completed Follow-up

## Decision

No new promoted cells. Both glucose@3h and MAP@3h remain candidate-only for this
experiment. Release still has 27 movable and 14 persistence-fallback cells.
The useful partial result is MAP interval calibration, not newly learned shared
physiology or reliable early-warning superiority.

We tested the three requested questions separately: prospective change risk,
signed change magnitude, and uncertainty conditioned on information available
before prediction. Protocol: `QUANTILE_CHANGE_PROTOCOL_20260907.md`.

## Controlled Experiment

Same frozen shared JEPA, same input contract, same measured label policy and
patient/hospital role hashes as the preceding experiment. Targets remain at 3h.
New readouts use 19 HGB quantiles (.05 to .95), with identical estimator settings
for direct, shuffled-history and target-history-only controls. Risk comes from
the distribution's two change tails and source-only probability calibration.
Current release, persistence, linear trend and the previous head are retained.

Two predeclared interval methods are both reported, without choosing a winner
on NWICU: pooled additive CQR, and pooled width-normalized CQR. Each uses at least
100 independent calibration patients, one anchor per patient. Pooled calibration
does not promise risk-conditional coverage; the same risk-bin and care-unit
evaluation requirements remain in force. Risk groups are frozen from the old
head, so changing the new distribution cannot redefine evaluation groups.

Glucose uses 690 fitting, 278 probability-validation, 138 calibration, 265 patient-
test and 106 hospital-test patients. MAP uses 1,287 / 487 / 246 / 499 / 258.
NWICU primary evaluation is 500 patients per target; chronological replay contains
1,944 glucose and 5,238 MAP anchors. Counts across targets may overlap.
All candidate heads and calibrators were serialized and hashed before NWICU read.

## 1. Can It Recognize Change Earlier?

Recall at an empirical <=5% false-positive budget, NWICU:

| Target | Previous independent classifier | New distribution risk | Release score | New history-only | New shuffled history |
|---|---:|---:|---:|---:|---:|
| Glucose | 10.12% | 13.36% | 20.24% | 12.96% | 10.53% |
| MAP | 12.14% | 14.45% | 13.29% | 15.03% | 8.09% |

Paired patient-bootstrap differences versus the previous classifier are +3.24
percentage points [-5.91,+13.48] for glucose and +2.31 [-4.65,+10.41] for MAP.
Neither is significant. MAP beats shuffled history on this ranking metric by
6.36 points [1.01,12.02], but not release or history-only. This is not a general
shared-state advantage.

These ROC cutoffs use held-out labels for a ranking diagnostic, not deployment.
At the actual frozen source cutoffs, NWICU FPR/recall are:

- Glucose: 8.70% / 21.86%. The FPR exceeds the 5% source budget.
- MAP: 1.22% / 8.09%. Low FPR, but most eligible changes are still missed.

Glucose also exceeds the source budget in held-out eICU hospitals (10.20% FPR).
Thus higher raw alert recall cannot be presented as a free sensitivity gain.
Research thresholds remain glucose 20 mg/dL and MAP 10 mmHg, not clinical alarms.
Replay lead time is to a recorded future measurement, not verified biological
onset; unmeasured episodes are not evaluated.

## 2. Does It Predict The Magnitude Better?

NWICU physical-unit MAE:

| Target | New direct | Previous direct | Release | Persistence | New history-only |
|---|---:|---:|---:|---:|---:|
| Glucose | 29.991 | 30.068 | 29.912 | 31.722 | 29.874 |
| MAP | 9.057 | 9.062 | 9.501 | 9.941 | 8.984 |

The raw q05/q50/q95 estimators are EXACTLY equal to the previous estimators on
all validation inputs (maximum absolute difference zero for both targets and
all three fitted sources). Tiny point differences result from sorting 19
quantiles instead of three, not from newly learned median physiology.

MAP still improves versus release: paired MAE -0.444, CI [-0.703,-0.166]. It does
not beat history-only (+0.073, CI [-0.112,+0.252]) or the previous direct head
(-0.005, CI [-0.027,+0.018]). Glucose does not beat release (+0.079, CI
[-1.285,+1.481]). The added distribution resolution has not solved point accuracy.

Stable-case excess MAE versus persistence is glucose +3.590 (CI [2.408,4.782])
and MAP +0.267 (CI [-0.113,0.636]); upper bounds exceed the predeclared 1 mg/dL
and 0.5 mmHg stable-harm limits. That failure is independent of interval coverage.

## 3. Are Intervals More Useful?

| NWICU target / method | Coverage | Mean interval width | Central 90% interval score |
|---|---:|---:|---:|
| Glucose / pooled CQR | 93.8% | 153.80 mg/dL | 201.88 |
| Glucose / normalized CQR | 94.6% | 158.23 mg/dL | 200.87 |
| MAP / release | 89.8% | 39.09 mmHg | 63.95 |
| MAP / pooled CQR | 90.2% | 38.85 mmHg | 59.18 |
| MAP / normalized CQR | 90.0% | 38.57 mmHg | 59.09 |

Glucose now has enough patients for a pooled calibrator; the prior two bins
each lacked 100. This fills the calibration computation gap, not the validation
gap: the new intervals are overly conservative relative to the project's 87-93%
band. Both glucose interval-score differences versus release have CIs crossing
zero. No new glucose interval is enabled in serving.

MAP normalized width is about 1.34% smaller than release, a small descriptive
reduction, not a large precision breakthrough. Its paired interval-score change
versus release is -4.859 (CI [-11.147,-0.167]); pooled is -4.769
([-10.812,-0.235]). Neither method beats the history-only or shuffled control
significantly on interval score. The previous three-quantile head with the SAME
pooled calibration is essentially as good (actually slightly better in interval
score), locating this improvement in calibration rather than 19-quantile fitting.

Both new MAP methods pass NWICU low/high pre-anchor risk-bin point coverage:
pooled 90.42% / 89.84%; normalized 90.10% / 89.84%.
But NSICU is 86.27% (51 patients), below 87%, for both. Source hospital marginal
coverage is 93.41% pooled and 93.023% normalized, above 93%; the source low-risk
bin is also too conservative. Small domain slices have wide sampling uncertainty.
These are explicit operating-gate misses, not proof of catastrophic miscalibration.

## Interpretation

There is a reusable partial calibration improvement for MAP, but no complete
promotion and no convincing improvement over personal-history controls. Further
quantile resolution alone is not supported as the next accuracy lever. Fixed
source alert thresholds also do not reliably transfer for glucose.

The next distinct question would require genuinely new evidence or a separately
fixed intervention/context representation, not picking another threshold on these
same test outcomes. Preserve this result; do not change the gate to make it pass.

This is exploratory transfer validation: NWICU and eICU test results have been
seen in earlier development, and frozen encoder training overlap remains
unverified. Nominal CIs are not multiplicity-adjusted promotion evidence. Seven
evaluation resamples are not seven independent training runs. No full calendar-
time validation, new untouched lockbox, onset labels or causal claims are added.

## Reproduction and Verification

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOKY_MAX_CPU_COUNT=4 python -u -m scripts.audit_quantile_change --output artifacts/quantile_change_replication
python -m unittest discover -s tests -q
python -m scripts.validate_release
```

- Distribution helpers: `whole_body_jepa/quantile_change.py`.
- Runner: `scripts/audit_quantile_change.py`; common scoring extracted from the
  existing prospective evaluator with a parity regression test.
- Results: `artifacts/quantile_change_20260907/report.json`.
- Frozen source report: `artifacts/quantile_change_20260907/development.json`.
- Candidate weights: `artifacts/quantile_change_20260907/candidate_heads.joblib`,
  tied to the prior local candidate and release via hashes; research-only.
- New tests cover tail atoms, ordered quantiles, paired risk comparison,
  independent-patient calibration, heterogeneous widths, and evaluator parity.
- Full regression suite: 226 tests passed. Release validation and diff checks
  passed; the serving catalog remains 41 cells (27 movable, 14 fallback).

Existing dependencies suffice. No patient rows are exported and no files are
uploaded. Candidate and release hashes are verified unchanged after evaluation.

The CQR approach follows [Romano et al.](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html);
cross-database or per-patient coverage is tested empirically, not guaranteed.
