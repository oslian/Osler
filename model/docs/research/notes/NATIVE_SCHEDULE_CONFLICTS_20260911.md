# Actual native conflict exposure

The original event table, schedule and token cache hashes reproduce their
construction report. Schedule pointers match the token history pointers.
At each arrival, source versions are resolved using only already available
reports. The audit then checks the actual latest-draw selection and positive
observation mask of the saved native token, not merely raw conflict counts.

Of 5526 scheduled queries, 514 (238 patients) contain at least one unmasked
observation whose same-draw known sources disagree. Glucose affects265queries/
104patients, bicarbonate187/94, creatinine23/8, potassium25/14, magnesium1/1.
Variable counts overlap; they must not be summed into patient totals.

This establishes actual cached input exposure, not prediction-error attribution.
It does not test subsequent source-support/own-target masking or prove how much
the neural state changed. A conflicting earlier report might precede the48h
query history; the conflict is known to the raw as-of record nonetheless.
The tiny direct magnesium exposure cannot alone establish an explanation of
its broad generalization gap; indirect cross-variable effects remain untested.

No native repair or training was performed. A lawful repair must retain the
first report until disagreement becomes known, then invalidate/replay its
contribution without deleting it retroactively from earlier predictions.
Simply skipping the later conflicting value leaves earlier state contamination
possible. Resolving source revisions and truly new same-valued measurements
must remain usable. Test this behavior before matched native retraining.

No release weights, registry, intervals or promotion changed.
Artifact: `artifacts/native_schedule_conflicts_20260911.json`.
