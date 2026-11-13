# Distributional Expert Readouts of the Shared Body State

Date: 2026-09-04. Run completed. Exploratory; candidate-only.

## Decision

Adding multiple prediction experts to the same frozen 64-dimensional whole-body
JEPA state does not add a complete masked-current reconstruction pass. This is
true for a free three-component Laplace mixture and for directly supervised
three- and five-expert variants. No model is promoted and the serving release
remains unchanged at **41 catalog cells: 28 movable and 13 persistence**.

The experiment supports a narrower conclusion. The shared state contains weak
information about which target-value stratum a patient belongs to, and multiple
experts produce meaningfully different candidate values. The legal pre-anchor
router cannot select among those candidates consistently across patients and
care units. Five experts do not solve that assignment problem.

This is a masked-current reconstruction experiment, not future forecasting.
It adds no forecast cell, interval, causal, clinical or next-measurement claim.

## Fixed Comparison

The two experiments use the same release-bound shared JEPA state, patient rows,
target and sibling masks, source roles and external datasets as the preceding
generalization audit.

1. `wide_point`: capacity-matched point reader.
2. `single_laplace`: one conditional Laplace density.
3. `mixture_laplace`: free three-component Laplace mixture.
4. `guided_3`: three Laplace experts with source-fit truth-tertile supervision.
5. `guided_5`: five Laplace experts with source-fit truth-quintile supervision.

The guided loss is mixture negative log likelihood plus `0.25` times the gate
classification loss. Truth bins are computed from source fitting data only and
are never available at inference. Guided readers have 3,561 and 3,535 trainable
parameters, respectively, compared with 3,565 for the point control. Seven
training seeds are averaged. Training duration is selected on eICU tune patients
only, before MIMIC or NWICU evaluation.

Every candidate is compared with population, wrong-patient and independently
trained capacity-matched placebos; seven patient splits; every supported care
unit; the previous frozen heads; the point and density controls; uniform expert
routing; and three wrong-patient routing permutations. The outcome-aware oracle
sees the true target and is reported only as a non-deployable upper bound.

## Main Result

| Evaluation set | Guided 3 full passes | Guided 5 full passes | Source-selected guided passes |
|---|---:|---:|---:|
| eICU hospital test | 0 | 0 | 0 |
| MIMIC-IV | 0 | 0 | 0 |
| NWICU | 0 | 0 | 0 |

All 15 numeric targets were evaluated in eICU and MIMIC. Fourteen were
evaluated in NWICU because urine output lacked compatible support. Source-only
selection chose `guided_3` for seven targets and `guided_5` for eight.

Lower point MAE occurs often enough to show that the new heads are not inert,
but never survives the complete diagnostic:

| Candidate | eICU lower MAE than free mixture | MIMIC | NWICU |
|---|---:|---:|---:|
| Guided 3 | 7/15 | 9/15 | 8/14 |
| Guided 5 | 7/15 | 7/15 | 12/14 |

These are point-estimate counts, not promotion passes. Against every frozen and
capacity-matched comparator, both guided families have zero complete passes in
all three datasets. Their own population/placebo/care-unit diagnostic also has
zero passes.

NWICU creatinine is the clearest example of both the promise and the limit:

| Reader | MAE (mg/dL) |
|---|---:|
| Previous fixed-32 neural head | 0.7104 |
| Previous source-selected neural head | 0.6569 |
| Free three-component mixture | 0.6457 |
| Guided three-expert mixture | 0.6505 |
| Guided five-expert mixture | 0.6449 |
| Outcome-aware five-expert oracle | 0.2268 |

The legal five-expert router is slightly better on average than the free
mixture, but the difference is too small and heterogeneous for the full gate.
The oracle gap is large because the true target can identify the best component;
that information is unavailable before the measurement and cannot be treated as
a model result.

## What the Router Learned

The experts did not collapse to a single branch.

| Evaluation set | 3-head bin accuracy | Effective heads | 5-head bin accuracy | Effective heads |
|---|---:|---:|---:|---:|
| eICU hospital test | 0.426 | 2.79 | 0.275 | 4.64 |
| MIMIC-IV | 0.420 | 2.72 | 0.270 | 4.49 |
| NWICU | 0.428 | 2.77 | 0.278 | 4.60 |

Chance accuracy is 0.333 for three heads and 0.200 for five. The above-chance
accuracy and broad expert use show that the shared state carries some target-
stratum information. However, legal routing has zero complete wins over uniform
routing or all three shuffled-routing controls in eICU and NWICU. A single
MIMIC target/control comparison passes one routing check, but no target passes
the complete incremental gate. Pooled wins do not generalize uniformly across
patient splits and care units.

The five-head model therefore creates more available alternatives without making
the assignment reliable. The limiting step is not merely decoder capacity or an
insufficient number of prediction heads. It is the stability and identifiability
of the patient-specific target stratum from legally available shared state.

## MIMIC Reconstruction Boundary

The pre-run protocol required every frozen MIMIC reference MAE to reproduce to
absolute tolerance `1e-6`. The original temporary transition file was no longer
available, so it was rebuilt from raw local MIMIC events with the frozen
September workflow. Patient support and scores are semantically close, but 17 of
87 frozen checks exceed the fixed tolerance. Median absolute mismatch among
failed checks is `6.21e-6`; the maximum is `7.44e-4` for heart rate.

The threshold was not relaxed. The report is deliberately marked
`complete: false`, and the MIMIC guided results above are diagnostic only. eICU
and NWICU reproduce their frozen references strictly. This experiment cannot
use a reconstructed near-match to manufacture a formal cross-database pass.

## Interpretation

The hospital-department analogy is only partly supported:

- Different departments, represented by expert heads, can generate useful and
  distinct candidate answers from one shared body state.
- The shared state does not yet provide a robust dispatcher that selects the
  correct department for each patient across units and databases.
- Directly teaching low/middle/high target assignment improves classification
  above chance, but does not make routing stable enough for promotion.
- Increasing three experts to five increases diversity and the oracle ceiling;
  it does not increase complete passes.

The next justified representation question is whether legally available owner-
organ state or better measurement-pure temporal evidence makes the target
stratum identifiable. Repeating larger mixtures, changing cluster initializers
or selecting heads from these external outcomes is not supported by this result.
Any next experiment must remain source-selected and separately test whether the
new routing evidence, rather than extra capacity, explains the gain.

## Artifacts and Verification

- Free-mixture protocol: `MASKED_MIXTURE_READOUT_PROTOCOL_20260903.md`
- Guided-assignment protocol: `MASKED_GUIDED_MIXTURE_PROTOCOL_20260904.md`
- Free-mixture report and weights:
  `artifacts/masked_mixture_readouts_20260903/`
- Guided report and weights:
  `artifacts/masked_guided_mixture_20260904_v2/`
- Implementations: `scripts/masked_mixture_readouts.py`,
  `scripts/audit_masked_mixture_readouts.py`,
  `scripts/masked_guided_mixture_readouts.py`, and
  `scripts/audit_masked_guided_mixture.py`
- Regression tests: `tests/test_masked_mixture_readouts.py` and
  `tests/test_masked_guided_mixture.py`

The reports contain release, input, protocol, code and weight hashes. MIMIC and
NWICU were previously inspected and are not untouched lockboxes. No patient rows
are stored in the repository, no serving artifact is changed, and no promotion
is allowed by either experiment.
