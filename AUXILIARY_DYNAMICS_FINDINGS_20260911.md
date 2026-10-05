# Auxiliary Dynamics Diagnosis

Source-only diagnostic of the 21 completed auxiliary-budget checkpoints. No
training, runtime, calibration or promotion change. Source-tune patients also
selected the glucose checkpoints; these are not unbiased final-test results.

## Question

Does auxiliary improvement represent conditional dynamics rather than correction
of an unused head's bias? Compare against current value plus the patient-weighted
median change learned only from source-fit measured pairs. Report patient-equal
movement, signed error and delta correlation separately.

## Results

Potassium beats this constant-change control in 6/7 patient split bootstrap CIs.
The control's median change is zero. It beats shuffled supervision in only 4/7
and has not passed an own-potassium-history or external comparison. Thus some
conditional signal is present, but cross-organ attribution is not established.

Bicarbonate, BUN, calcium and hematocrit beat the glucose-only auxiliary outputs
in 7/7, yet beat the constant-change control in 0/7. Improvement over a weak
unused head is not sufficient evidence of useful dynamics. This does not prove
that their entire improvement is merely centering.

For potassium, retrospective changes above the source-fit upper quartile have
median-across-seed MAE 0.647 versus persistence 1.010. The remaining observations
have MAE 0.306 versus 0.281; auxiliary is numerically worse in 6/7 splits there.
These are descriptive, future-defined subgroups, not deployable routing rules
or subgroup significance claims. They cannot show advance recognition of change.

Predicted movement is smaller than realized movement. Conditional uncertainty
can produce this pattern even for an optimal point forecast; global amplitude
inflation is not justified. Magnesium/phosphate lack minimum patient support
for bootstrap comparisons and are not counted as failures.

## Localization And Limits

The earliest measured full-context failure remains source-fit versus new-patient
reversal, before external transfer. Auxiliary supervision learns some targets
but does not improve primary glucose significantly (0/7). Neither universally
dead gradients nor lack of all physiological signal explains the result.
No unique root cause is proved. Remaining questions include whether useful
conditional information exceeds own-target history and whether it transfers.

Evidence: `artifacts/encoder_auxiliary_dynamics_20260911.json`.
Reproduce: `python -m scripts.audit_auxiliary_learning --dynamics`.
The artifact binds the script and existing source/checkpoint inputs. No new
MIMIC or NWICU evaluation was used to select this diagnostic.
