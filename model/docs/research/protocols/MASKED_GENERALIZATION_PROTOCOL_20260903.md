# Four Generalization Diagnostics: Frozen Protocol

Status when written: implementation tested; main experiment not yet run.
This is exploratory masked-current reconstruction, not forecast promotion.

Completion note (added after the run): all predeclared comparisons finished.
The protocol below is unchanged; see
[the results](../findings/MASKED_GENERALIZATION_FINDINGS_20260903.md) for outcomes and limits.

## Cohort and Split

Use the existing hospital-balanced 5,000-stay eICU event extraction. Do not
change its inputs or re-extract different values between arms. Use all 15
numeric targets in the packaged 41-cell catalog; NWICU urine remains unsupported.

Reserve hospitals with `stable_hash("masked_generalization_hospital", id) % 5 == 0`
before any new fitting or model selection. Remaining hospitals use the existing
patient split seed 73 for fit/tune. This yields 1,736 fit patients, 1,737 tuning
patients and 1,288 hospital-test patients in 55 held-out hospitals. Exact fresh
target support is smaller and must be reported, not inferred from cohort size.
The frozen encoder's historical pretraining may have included these people;
this is hospital-heldout for the new intervention, not proof of unseen-hospital
pretraining. Source care units only influence weighting and audit, not inputs.

One outcome-independent anchor per patient, fresh truth <=1h, inputs <=6h.
Remove target, predefined siblings and their entire histories before encoding.
No new treatment, future-label or hospital features. Keep neutral masks/ages.

## Frozen-Encoder Arms

1. Raw normalized history plus current observations -> standardized ridge.
2. Same hidden observations -> frozen shared JEPA latent -> standardized ridge.
3. Frozen latent -> original reconstruction-head architecture, ordinary L1.
4. Same head -> half patient-uniform / half care-unit-balanced sampling.
5. Same head -> half ordinary L1 / half group-DRO risk.

Raw and latent ridge probes are not parameter-count matched to one another:
their input widths differ. A difference localizes a useful diagnostic direction,
not uniquely proven information loss. Each arm has its own exactly matching
placebo capacity and preprocessing; placebo contexts are deranged at fit and
independently at tuning/evaluation. Ridge alpha choices are fixed at
0.1/1/10/100/1000, selected only on eICU tuning patients.

Neural arms use seven optimizer seeds (7/19/31/43/59/71/83), initial release heads,
AdamW lr=0.003, weight_decay=0.001, batch=128, saved steps 32/128/512. Equal seed
ensemble, no external seed selection. Both real and placebo ensembles select
their own checkpoint using the same tuning objective: half mean normalized MAE
plus half worst supported care-unit normalized MAE. Report all learning curves;
selection at 512 is not proof that convergence has been reached. The fixed-32
ERM snapshot is an additional matched-budget reference.

Balanced/DRO groups need 20 fit patients. Unsupported groups remain in the
ordinary-risk component instead of being discarded. Tuning worst-unit scores
need 30 patients. Group DRO uses log-space exponentiated updates (step 0.01),
plus regularization/early model selection. This is a bounded mixed-risk test,
not a claim to reproduce published NLP/image improvements on physiology.
References: [Group DRO](https://arxiv.org/abs/1911.08731),
[scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

## Limited Shared-State Update

Keep the existing model graph; update only `state_encoder` (the final shared-body
projection) and reconstruction heads. Keep history encoder, organ mixers,
future/EMA/value heads frozen. Cached pre-projection vectors are equivalent to
the existing frozen upstream computation; there is no second body encoder.

Compare two otherwise identical joint multi-target fits:

- no preservation objective;
- weight-10 replay distillation of unmasked direct future values, context and
  target-future latents from the incumbent at 1/3/6/12/24/48h.

Replay uses fit patients and incumbent predictions, not future observations.
Both use balanced sampling, seed 73, head lr=0.003, projection lr=0.0001,
weight_decay=0.001; global snapshots after 8/32/64 updates per target. One global
snapshot per body state selected using average target tuning objective. A paired
full-capacity placebo body projection/head fit uses shuffled masked contexts.
Unlike the seven-seed frozen head ensembles, these shared-update arms have ONE
optimizer seed; no seven-training-seed robustness claim is allowed. The paired
protected/unprotected contrast has identical capacity and data. This is not
end-to-end future JEPA retraining or a test of all possible encoder changes.

## Evaluation and Boundaries

Freeze every model and selection before inspecting hospital-test/MIMIC/NWICU
scores. Same fixed patient sampling and existing diagnostic gates: population,
three wrong-patient controls, separately trained placebo, seven overlapping
patient split CIs and all supported care-unit CIs. CI thresholds unchanged.
These repeated nominal screens are not multiplicity-adjusted confirmations.

Audit the limited shared-state candidates against the existing ordinary forecast
selector across all 41 catalog cells and both databases, excluding known MIMIC
head-fit patients. Explicitly mark absent or incompatible labels, binary event
tasks and insufficient patient support. Frozen head probes cannot alter serving
forecasts. Changed representations cannot reuse incumbent conformal intervals.

The MIMIC and NWICU data have been inspected before. They remain exploratory
external validation, not untouched lockboxes. No automatic source selection,
runtime change, deployment promotion, clinical or causal claim. Save only
aggregate reports and model parameters in the project; no patient rows.
