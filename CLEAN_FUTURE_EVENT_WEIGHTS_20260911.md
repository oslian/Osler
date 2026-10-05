# Outcome-event equalized sensitivity

Saved clean-input native glucose predictions, same patient roles. First average
absolute errors for each stay/patient/future source event, then average events
within patient. Repeated event truth consistency asserted. No predictions,
training or model selection changed. This sensitivity analysis does not replace
the prespecified query-level patient-equal promotion gate.

Hospital501queries map to490unique future events; source repeats are similarly
limited. Clean-full versus clean-own significant gain0/7both, sourceharm2/7,
hospitalharm0/7. Clean-full versus old-full gain0/7both, sourceharm1/hospital2.
Clean-full versus persistence gain7/7both.

Event-equal medianMAE hospital cleanfull41.9871,cleanown42.6268,persistence77.5851;
source34.8549,34.1829,57.1152. No stable incremental benefit was hidden solely
by this repeated-outcome weighting. This is not proof every scoring choice is
correct or all possible clinical endpoints have been tested.
No runtime, calibration or promotion.
Artifact: `artifacts/clean_future_event_weights_20260911.json`.
