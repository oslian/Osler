# Additional-Domain Matched Reconstruction Training

Scope is all47 supported current-value targets, five already frozen component
folds, real versus process-only arms, eight fixed epochs. The36 targets with
adequate support in every role are not the only trained targets. No future
JEPA objective or untouched external evidence is claimed. Old selected-insulin
population is not replaced by this broad population.

Use existing fresh_encoder and NativeBodyReconstruction/NativeOrganStateResidual
with schema-only release metadata. Identical seed19 initialization and sampling
for both arms. Never load learned release parameters. Read raw histories from
the exact full SQLite cache; scale using this fold's fit-only median/IQR.
Target and siblings stay masked throughout history. Process control sets only
normalized numeric values to zero, retaining masks, ages and event times.

Match the prior fixed optimization budget:1072 replacement draws per target
per epoch, AdamW1e-4, SmoothL1beta1, gradient clip1, four CPU threads, eval-mode
dropout convention with gradients enabled. Patient-target pairs are unique;
bounded [.5,2] weights balance full-history hospital components within each
target's FIT pool. This differs from the old selected/broad mixture; it tests
the broader population and must not be called its same-cohort replication.
Initialize readout weights zero and bias to fit-patient median in fold units.
Record number of unique fit examples seen; do not call this eight passes
through every sample or claim convergence. No held-out outcomes choose budget.

Only fit patients contribute gradients, normalization, baseline or sampling.
Selection, calibration and test remain separate. Do not evaluate test until
all scheduled training completes. Epoch staging is execution only; no winner
selection by intermediate fit loss. Preserve exact optimizer/RNG/model states
and hash-bound dependencies for resume.

Compare real/process/fit-center on identical outer examples, patient-equal and
component-balanced, separately by target; normalize aggregate errors using
fit-only scales and also report raw units. Fold correlations remain explicit.
Calibration comes only after frozen model choice; no runtime promotion is
authorized by this experiment. Later untouched external validation is still
required. A negative finite-budget result does not establish absent physiology.
