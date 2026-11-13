# Matched bedside expansion training

Fourteen fits completed: seven original patient partitions, lab-only versus
lab plus timestamp/quality-qualified HR, MAP, O2 saturation, respiratory rate,
temperature. Same clean glucose labels, eligibility, patient sampler, source
selection, initialization and 32-epoch budget. The source support whitelist
is recomputed from each arm's fitting patients. Both use the same existing
native filter/EMA future-target JEPA and identity value readout. No truncation.

| Median patient-equal MAE | Lab | Bedside | Persistence |
|---|---:|---:|---:|
| New source patients |34.884272|34.250550|56.779125|
| Held-out hospitals |42.079848|42.881423|77.167846|

Patient-paired bootstrap 95% CI gain/harm counts out of seven overlapping
partitions (not seven independent trials):

| Contrast | New source patients | Held-out hospitals |
|---|---:|---:|
| Bedside vs lab |0 / 0|1 / 3|
| Bedside vs persistence |7 / 0|7 / 0|
| Lab vs persistence |7 / 0|7 / 0|

All seven lab-control prediction arrays are bit-exact to the preceding
clean_full run, including NaNs. All44 artifact bindings verified. Hospital
evaluation has189patients. Thus this is a reproducible failed incremental
repair, not an uncontrolled change in the reference model or future labels.

Adding these five observed channels is not sufficient to repair the extra-
context generalization gap. It does not prove bedside information is useless:
this intervention changes numeric context, masks and event cadence together.
The source patient result is numerically better without significant gain;
hospital heterogeneity remains. No new external database, full multi-target
retention, interval calibration or full persistent-organ architecture gate.
No promotion, release weights, registry or serving changes.

Next distinguish numeric support/scale from event-frequency effects on this
same expanded cache before another architecture change. In particular, the
new channels still use the inherited release numeric transform and the longer
event sequence. Source-supported does not imply representation-invariant.
Do not frame this result as proof that all original input information is now
complete: comprehensive treatments/bedside assessments remain outside scope.

Artifact: `artifacts/native_bedside_training_20260911.json`.
Private predictions/checkpoints remain outside Git.
