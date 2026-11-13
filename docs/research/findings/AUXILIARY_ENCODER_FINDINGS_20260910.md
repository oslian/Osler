# Availability-Safe Auxiliary Supervision

## Data Support

Reconstructed auxiliary pairs at49,823 unchanged scored glucose queries from
2,386 development patients. Each original seed uses2,250 fit and136 tune
patients. Glucose current, truth and actual elapsed time exactly reproduce the
existing dataset within1e-8; no new glucose task or patient selection was used.

Fourteen non-glucose variables meet the prespecified250-patient/1000-query
minimum in every fitting split: anion_gap, bicarbonate, bun, calcium, chloride,
creatinine, hematocrit, hemoglobin, magnesium, phosphate, platelets, potassium,
sodium and wbc. Support is not evidence that these targets are predictable.
Labels require both a known fresh current measurement and a future draw within
5-7h. Missing labels are not interpolated. Actual delta_t remains recorded.

## Matched Source Pilot

Nine fits:3seeds x glucose-only / real auxiliary / shuffled-patient auxiliary;
8epochs, unchanged model and glucose queries. Add0.25 times mean auxiliary
normalized value+JEPA loss. No hospital or MIMIC evaluations were performed.

Patient-equal source-tune glucose MAE (mg/dL):

| Seed | Glucose only | Real auxiliary | Shuffled auxiliary |
|---|---:|---:|---:|
|7|38.4100|38.4818|38.5752|
|19|36.4305|36.3570|36.4109|
|31|38.2308|38.3051|38.3203|

Real auxiliary vs glucose-only paired95% CIs for error delta:
seed7[-0.0336,0.1799], seed19[-0.2010,0.0442], seed31[-0.0280,0.1910].
Significant improvement0/3 and harm0/3. All three real-vs-shuffled CIs also
cross0. Shuffled is numerically worse than real3/3, but no significant finding.

Glucose-only source-tune logs exactly equal the first8 logs of the existing
2250/64epoch whole-unbounded run for each seed (maxdifference0). Thus the new
query-array construction and pilot controls reproduce the existing training.

## Interpretation

The missing multi-target supervision can now be supplied from real raw events;
it is not universally unavailable. Adding it in this short trial did not fix
source generalization. This rejects an immediate demonstrated improvement,
not the entire supervision hypothesis: all primary/real-auxiliary checkpoints
selected epoch8, the budget boundary. No convergence, information-ceiling,
organ-mechanism or runtime promotion claim is warranted. The model remains the
same base world-model, not a newly enabled persistent organ transition.

Tune patients also select checkpoints, so these bootstrap results are
diagnostic rather than unbiased external tests. A longer source-only matched
trial is still distinct from this pilot; support for the auxiliary targets'
own predictions, negative transfer, full seven seeds and compatible intervals
has not been evaluated. Pretraining patient overlap remains unexcluded.

## Execution Integrity

An initial execution stopped while preparing seed19 because placebo processing
included sparse unselected columns. Restricted placebo to selected auxiliaries;
also corrected the result aggregation call's argument order. The incomplete
run/report was archived privately and is not evidence. The final run completed
all nine fits. No early interruption of a running training process occurred.

309 unittest tests pass. No model weights, runtime registry or calibrator
changed. Raw labels/checkpoints remain private; aggregate artifacts:
`artifacts/auxiliary_supervision_20260910.json` and
`artifacts/encoder_auxiliary_pilot_20260910.json`.
