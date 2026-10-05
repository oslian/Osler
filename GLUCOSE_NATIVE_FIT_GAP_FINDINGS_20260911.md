# Native glucose: earliest incremental failure

Question: does whole-context failure start only at hospital transport, or
already between fitting and new patients in the source database?

Reused all 21 completed native glucose fits. No training, tuning, threshold
selection or new data was performed. Verified all 45 original bindings and
reproduced all tune/evaluation/hospital MAEs within 1e-7. Current four patient
roles are disjoint within every seed. Applied existing patient-equal MAE and
2,000-draw patient paired bootstrap to fixed predictions.

Aggregate: `artifacts/glucose_native_fit_gap_20260911.json`.
Reproduce: `python -m scripts.audit_glucose_native_fit_gap` (fresh output only).

| Role | Own MAE | Whole + real events | Whole + event support | Whole vs own significant gain/harm |
|---|---:|---:|---:|---|
| Fit | 34.037809 | 32.288892 | 32.295784 | 6/7, 0/7 |
| Tune | 33.535815 | 33.731561 | 33.763309 | 0/7, 1/7 |
| New source evaluation | 33.976223 | 34.840807 | 34.961175 | 0/7, 2/7 |
| Held-out hospitals | 41.402542 | 42.400697 | 42.478966 | 0/7, 0/7 |

Units mg/dL; medians across overlapping seeds, not independent trials or
paired differences of medians. Whole-minus-own numerical error is positive
in all seven source evaluation splits, but only two CIs exclude zero.
Fit CIs are descriptive, not unbiased evidence of generalization.

Real event numbers versus the identical whole model retaining event masks
but neutralizing numbers: fit gain1/7 harm2/7; tune0/7; evaluation0/7;
hospital gain1/7 harm0/7. Thus neither the training advantage nor new-patient
accuracy establishes stable benefit from the recorded treatment numbers.
The support arm retains other lab values, so this does NOT prove workflow
alone explains the advantage or that other physiological values are useless.

## Located failure, not a unique biological cause

The earliest measured incremental failure is source fit-to-new-patient
generalization. It precedes database transport and persists after native-event
history construction. Calling it solely a MIMIC harmonization problem, lack of
all predictive ability, or an over-strict promotion gate is incorrect. Both
own and whole still beat persistence in the original source/hospital audit.

This repeats the earlier glucose fit reversal on a distinct rate-query task.
It supports an overfitting/unstable-context association diagnosis, but cannot
distinguish finite support, unobserved interventions, workflow confounding and
insufficient representation from these results alone.

Separate controlled evidence must not be conflated:
- Identity decoding fixed a proven glucose amplitude cap in earlier trials.
- Source-support masking fixed a proven out-of-support numerical channel.
- Higher projection learning rate recovered a constructed event signal, but
  did not improve real potassium. Learnability is not real conditional signal.
- Unclipping native clocks corrected collisions but did not stably improve MAE.

Limits: potassium-selected cohort; original clipped clock; inherited encoder
pretraining overlap not excluded; treatment record time not independently
verified administration availability. No untouched external validation,
calibration, retention, causal claim or runtime/promotion change.
