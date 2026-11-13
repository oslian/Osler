# Additional-Domain Masked Reconstruction

## Completed Evidence

All five outer inference folds completed with the frozen epoch8 models.
1,800,214 accepted patient-target pairs cover 73,670 people, 47 targets and
38 connected hospital components. Exact TEST identities, raw labels, positive
fit scales, finite predictions, no cross-fold patient/component overlap and
all prediction/dependency hashes were verified before scoring. There are no
missing accepted predictions. Previously rejected anchors remain excluded by
the fixed data contract: 5,467 no-fresh-label and 4,429 no-independent-history.

The target and its near-deterministic siblings, including their histories,
are masked. This is CURRENT-value reconstruction, not future forecasting.
The numeric physiology arm is compared with a capacity-matched process-only
arm (same masks/times, numeric values removed) and fit-only population center.
Process-only is not shuffled-subtype placebo or own-history persistence.

## Results

| Fit-scale normalized target macro MAE | Physiology | Process-only | Population |
|---|---:|---:|---:|
| Patient equal within each target | 0.925567 | 0.975469 | 0.978512 |
| Component equal within each target | 0.901157 | 0.945913 | 0.955007 |

Different targets' raw units are NOT averaged. Per-target raw MAE and paired
patient/component bootstrap intervals, all component results and supports are
in the machine-readable report. Macro figures are descriptive point estimates.

13/47 pass the predeclared DESCRIPTIVE screen: albumin, ALT, AST, bilirubin,
BUN, calcium, creatinine, hematocrit, INR, phosphate, potassium, total protein,
WBC. Each has at least30patients/3components, negative paired means versus
BOTH controls at patient and component aggregation, and all four one-sided
p-values below0.05/47. The prior helper's39-target denominator was not used.

Example raw-unit patient MAEs (physiology/process/population):
- Creatinine:0.6334/0.6585/0.6594.
- BUN:11.0815/12.0034/12.1810.
- Potassium:0.4335/0.4485/0.4475.
- INR:0.3248/0.3386/0.3608.

Passing pooled/equal-component tests does not mean every component improves.
Albumin and total protein remain mutual inputs; these successes are not proof
of independent organ physiology or identifiable internal stores.

## Failure Localization

21 nonpassing targets have lower average errors in ALL four comparisons,
but at least one component-level significance test fails: alkaline_phos,
amylase, bicarbonate, bilirubin_direct, cortisol, cpk, esr, fibrinogen,
free_t4, glucose, heart_rate, hemoglobin, ldh, lipase, magnesium, platelets,
ptt, respiratory_rate, serum_osmolality, temperature, uric_acid.
This localizes unstable/insufficient between-component evidence, not a proven
sample-size-only cause. Lipase also fails a patient-level significance test.

13 targets have at least one non-improving MEAN, not merely a narrow CI miss:
anion_gap, bnp, chloride, ferritin, lactate, map, myoglobin, o2sat, ph, sodium,
triglycerides, troponin_i, tsh. MAP is worse than both controls at both levels;
raw patient MAE13.2662 vs13.2048/13.1654. Sodium is also worse at patient level
(3.6324 vs3.6097/3.5903). More patients or a looser significance threshold is
not demonstrated to fix these targets. Myoglobin/pH have limited domain support.

## Interpretation and Remaining Work

Confirmed: under this broad cohort and corrected observation contract,
numeric cross-variable histories add held-out reconstruction information for
some targets. Shared representation is not universally devoid of useful signal.
Still unresolved: why the remaining numeric responses fail to transfer or add
value, and whether a source-only multi-component calibrator can transport.
Prior one-component calibration failed; this new experiment has multiple
reserved calibration components but has NOT fitted/evaluated that repair yet.

Do not interpret13versus an older experiment's4 as a matched causal improvement:
cohort, target family, scaling and support differ. The old insulin-selected
cohort support problem is not solved by this broader population. Eight fixed
epochs with replacement sampling are not proof of convergence/full exposure.

Training folds overlap; fold2 explicitly reuses an exactly equivalent fit.
The statistics condition on these trained functions and are descriptive,
not independent training replications or a fresh external lockbox. Existing
MIMIC/NWICU outcomes have been repeatedly inspected. No forecasting, interval,
clinical or causal claim follows. No runtime promotion, weights, registry or
calibrator changes. Candidate status remains.

Report: `artifacts/native_reconstruction_additional_evaluation_20260924.json`.
Protocol: `RECONSTRUCTION_ADDITIONAL_EVALUATION_PROTOCOL_20260924.md`.
Scorer: `scripts/score_additional_reconstruction.py`.
