# Local Organ Exchange: Frozen JEPA Structural Pilot

## Frozen question and scope

Does an explicitly local, recurrent organ graph add predictive value beyond
the same owner-only shared JEPA, without a global-body information bypass?
This is a bounded structural experiment, not a new release or a complete
physical-body simulator. All nodes use the SAME frozen JEPA weights, with
non-owner measurements removed before encoding the entire patient history.
There is one body/organ trajectory and five owner-kidney target readouts:
creatinine, BUN, urine output, potassium, bicarbonate around nominal 6h.

Existing PyTorch predict-update, recurrent rollout, canonical organ contracts,
patient bootstrap and conformal helpers are reused. No dependencies added.
Organ ownership is an observational hypothesis, not exclusive biological control.
No conservation is imposed on abstract latents or asynchronous laboratory values.

## Arms fixed before results

None, sparse canonical graph, predeclared node-permuted graph, dense graph.
Same parameter count, initialization, optimizer, batches and epoch count.
The permutation preserves edge count and degree multiset, not per-organ degree;
it is a topology control, not proof of any particular biological edge.
No body-summary-to-organ feedback and no body-summary-to-target readout.
Frozen owner JEPA, persistence, and the released router policy are references.
If a nominal 6h task has no shipped registry entry, its router reference is only
persistence, NOT evidence of evaluating an existing shipped cell. Report that
status explicitly. The primary comparison remains sparse vs matched no-edge.
Residuals start at exactly zero and have direct value and target-JEPA losses.
Internal states retain uncertainty bookkeeping, not calibrated physiological truth.

## Data and evaluation

Use existing eICU AKI 6h repaired transitions and source-supported inputs only.
Exclude subjects and stays listed in the pretrained checkpoint's event-example
file before sampling. This is a conservative exclusion, not knowledge of every
historical checkpoint or hyperparameter exposure. No patient identifiers enter
the model. Keep histories within the same encounter and at/before anchor.
Choose one anchor per patient deterministically, before examining future values;
future-value availability may determine eligibility, never favorable errors.
Reserve whole hospitals, then seven patient-disjoint train/calibration/test splits.
Bounded cohort: up to 1800 development and 600 hospital subjects.
Fixed eight epochs; no external-based early stopping or graph selection.

Use measured anchor/future pairs, strict/exact source quality, actual event delta_t
when recorded (positive, <=48h). Source-only conformal radius per target/split.
Report patient paired bootstrap, per-unit support, coverage and interval width.
Every target must beat all controls; passing point checks alone is not promotion.
MIMIC up to 800 subjects is supplementary reused validation, NOT a lockbox.
Its available transition cache lacks actual future event timestamps: nominal-time
transfer scores are explicitly not strict event-time external validation.

## What this does not prove

Freezing the backbone isolates transition structure; a negative result cannot
reject end-to-end organ-state learning. No drug or intent inputs are added here.
Masked-recovery generalization, physical-store conservation, full multi-horizon
future supervision, new-database validation and calibrated hidden-state meaning
remain separate requirements. Structural intervention tests are not causal effects.
No model/checkpoint/registry in models/ is modified; promotion remains false.

## Implementation clarifications (not new selection criteria)

The seven anatomical observation nodes are constructed from owner-restricted
context encodings, NOT the checkpoint's older functional-system token names.
The fixed source support-discovery patients are always assigned to training.
Every future query reads an interpolated fixed one-hour trajectory; another
target's future measurement time cannot alter its integration step boundaries.
Post-run diagnostics of residual saturation and the full joint nominal-time
baseline are descriptive only and do not select arms or modify any thresholds.

## Support-only expansion, declared after pilot and before expanded outcomes

The 1800-development pilot produced only 74-86 valid BUN/creatinine patients
per source test partition, below the fixed 100-patient screen. Expand to 6000
development, 2000 hospital and 2400 supplementary MIMIC patients. Keep all four
arms, seven split seeds, initialization, eight epochs, loss and thresholds.
This is a follow-up on an inspected dataset, not a new independent confirmation.
Publish both runs; do not choose favorable targets or tune graph settings.
