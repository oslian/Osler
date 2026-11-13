# Shared Body State Reconstruction Findings

Date: 2026-08-30

## Question

Does one shared whole-body state contain recoverable current physiology, or is
the apparent signal only target leakage, deterministic formulas, or hospital
measurement workflow?

This audit is deliberately separate from future forecasting. It evaluates the
current shared state only.

## Leakage-safe task

For every target:

1. remove the target from the anchor and the complete causal history;
2. remove near-deterministic sibling variables as well;
3. train one joint encoder on eICU only;
4. reconstruct the hidden current value on MIMIC-IV;
5. compare against the eICU population center and an equal-capacity placebo
   trained after independently permuting each physiological variable;
6. require all seven patient splits and the care-unit gate to pass.

Only observed values with measurement age at most one hour are accepted as
reconstruction truth. The model receives no future value or future treatment.

## Training correction

The earlier pretraining and the external audit did not ask exactly the same
question. Pretraining mainly hid isolated random values, while the audit hid a
target together with all deterministic siblings and its complete history. It
could also reconstruct stale measurements as if they were current truth.

The corrected objective uses:

- target-aware structured block masking on every training row;
- low-rate additional random masking;
- occasional whole-organ masking;
- fresh measured truth only;
- one shared body latent with target-specific reconstruction heads.

This is a training-contract correction, not an increase in latent size.

## Strict external result

Training: full eICU cohort available to the event contract, 3,288 patients and
94,179 rows. External validation: MIMIC-IV, 984 patients and 16,561 rows.

| Target | MIMIC rows | Patients | Shared-state MAE | Population MAE | Placebo MAE | Gain vs population |
|---|---:|---:|---:|---:|---:|---:|
| ALT | 6,325 | 506 | 8.421 | 8.535 | 8.531 | 1.34% |
| AST | 6,337 | 509 | 12.062 | 12.289 | 12.227 | 1.85% |
| Bicarbonate | 11,725 | 956 | 0.847 | 0.937 | 0.960 | 9.65% |
| Ionized calcium | 3,914 | 394 | 14.764 | 15.081 | 14.810 | 2.10% |
| pH | 4,520 | 599 | 0.780 | 0.799 | 0.792 | 2.43% |
| Phosphate | 14,557 | 929 | 0.859 | 0.892 | 0.882 | 3.74% |

All six targets passed every one of the seven patient-level paired bootstrap
tests and every care-unit gate against both references.

The artifact is
`artifacts/shared_state_reconstruction_strict_full_eicu_external_20260830.json`.

## Boundaries and negative results

- 41 targets had sufficient external support; 6/41 passed the strict gate.
- Anion gap passed all patient bootstrap tests but failed the care-unit gate,
  primarily in CVICU and one MICU/SICU partition. It remains candidate-only.
- Bilirubin passed 4/7 strict partitions; PaCO2 passed 1/7.
- Glucose, heart rate, MAP, creatinine, blood counts and most sparse markers did
  not show stable target-and-sibling-masked reconstruction from the shared
  state.
- A hospital-adversarial screen reduced the passing set to MAP and temperature.
  It was rejected because removing site information also removed useful
  physiological structure.
- Results varied when the cohort and masking contract changed. Therefore this
  audit establishes external patient/care-unit evidence for six targets, not
  training-seed stability or final release promotion.

## Interpretation

The shared state is no longer supported by bicarbonate alone. It carries a
reproducible acid-base/mineral and liver-injury subspace that crosses databases
and care units without access to target siblings. The strongest effect is
bicarbonate.

It is not a universal lossless representation of every measured variable.
Variables dominated by patient-specific baselines, acute external events, or
weak conditional relations cannot be reconstructed honestly from the rest of
the body at one anchor. They must remain observed, target-specific, or missing.

This result validates a partial shared current-state representation. It does
not validate shared future dynamics, causal treatment effects, or replacement
of the existing per-target forecast router.

## Follow-up methods, 2026-08-31

Several explanations for the remaining failures were tested without weakening
the gate.

| Method | Strict result | Decision |
|---|---:|---|
| Three-seed neural ensemble | 4 targets | Reject; initialization variance was not the main bottleneck |
| Inverse-frequency structured masking | 4 targets | Reject; dense-target dominance did not explain the failures |
| Hospital-adversarial latent | 2 targets | Reject; it removed useful physiology with site information |
| Target-specific Ridge decoder | 2 targets | Reject; decoder capacity was not the missing ingredient |
| Rank-32 factor analysis | 4 targets | Reject; explicit independent-noise modeling did not beat PCA |
| Rank-32 PCA shared modes | 4 targets after leakage correction | Retain as a complementary shared-state component |

The PCA rank screen produced 4 passing targets at rank 16, 7 before leakage
correction at rank 32, and 1 at rank 64. Rank 32 gave the best bias-variance
tradeoff. Scaling the frozen rank to the full eICU cohort preserved the result.

An audit then found that raw, weight-normalized and three-hour urine output were
not hidden together. Their apparent 40% reconstruction gain was a unit/
aggregation shortcut. All urine representations are now one sibling group;
after rerunning the complete gate, the legitimate low-rank results are:

| Target | Low-rank MAE | Population MAE | Placebo MAE | Gain vs population |
|---|---:|---:|---:|---:|
| Anion gap | 0.996 | 1.133 | 1.154 | 12.02% |
| Bicarbonate | 0.851 | 0.937 | 0.957 | 9.25% |
| PaCO2 | 0.758 | 0.805 | 0.811 | 5.77% |
| Phosphate | 0.770 | 0.892 | 0.893 | 13.66% |

The strict artifact is
`artifacts/low_rank_shared_state_rank32_strict_full_eicu_external_20260831.json`.

The nonlinear shared state and linear modes provide evidence for eight unique
targets in total: ALT, AST, anion gap, bicarbonate, ionized calcium, pH, PaCO2
and phosphate. This is complementary evidence from two views of one current
body state, not proof that a combined runtime model has passed. A combined
readout must be selected on discovery data and validated on a new untouched
database before release.

## Cross-fitted hybrid audit

A target-routed hybrid was then tested rather than assumed. Three
patient-disjoint eICU folds generated out-of-fold neural and rank-32 PCA
predictions. A two-input Ridge readout was fitted on one disjoint half of the
OOF patients, while the other half selected neural, PCA, or fusion. The chosen
source and coefficients were frozen before the MIMIC-IV audit. The same route
and capacity were used for the permuted-physiology placebo.

Only bicarbonate and phosphate passed all seven patient gates and every
care-unit gate. The hybrid therefore reduced, rather than expanded, the strict
passing set. It is rejected as a runtime path.

The failure identified a more specific domain-shift problem: the source
ranking learned in eICU was not stable in MIMIC-IV. For example, eICU selected
PCA for ALT even though the nonlinear state was the better MIMIC source;
PaCO2 showed the opposite pattern. A linear fusion cannot solve a source
ordering that changes by database and care environment.

The audit artifact is
`artifacts/hybrid_shared_state_strict_full_eicu_external_20260831.json`. The
executable hybrid branch was removed after rejection. MIMIC-IV has now been
used to inspect both component rank and hybrid behavior, so it cannot be used
again to tune source routing and still be called an untouched lockbox. A
frozen source policy requires validation in HiRID, AmsterdamUMCdb, or another
independent database; none is currently present locally.

## NWICU independent lockbox, 2026-09-01

The complete Northwestern ICU database was subsequently added through a
native item-dictionary adapter and used without changing the frozen neural or
rank-32 methods. The nonlinear state passed ALT, bicarbonate and total protein.
The rank-32 modes passed 16 targets. Four source-target relations now reproduce
in both MIMIC-IV and NWICU: nonlinear ALT and bicarbonate, plus rank-32 anion
gap and phosphate.

The detailed extraction contract, support counts, MAEs and boundaries are in
`docs/research/findings/NWICU_EXTERNAL_VALIDATION_FINDINGS_20260901.md`. NWICU has now also been
opened as a lockbox and cannot be used to tune a combined source router while
remaining an untouched external validation set.
