# Frozen event path use

Seven trained real-event checkpoints reproduce reported evaluation/hospital
MAEs within1e-7. Their initially zero event projection weight norms become
.02843-.13275. Thus learned event weights are not frozen or exactly zero.

Median across seeds of mean absolute forecast shift when numerical event
channels are zeroed (known masks retained): source .000825, hospital .001000
potassium units. All six event channels zeroed: .001849/.002204. Original
patient MAE is .384201/.387170. Hospital median MAE with numbers off .387601,
with all off .387520; these small descriptive changes are not paired efficacy
claims. Whole-model forecast dependence on event channels is weak, not absent.

This localizes weak end-to-end utilization despite trained weights, not why it
is weak. A fit may correctly ignore weak incremental context, or intermediate
representations may attenuate useful information. Fixed-input interventions
are off-manifold and cannot distinguish those explanations or establish
treatment causality. Do not amplify weights without a matched source training
and outcome test. Only insulin record context at lab-supported steps was tested.

Aggregate `artifacts/rate_jepa_event_path_20260911.json`, runner
`scripts.audit_rate_jepa_event_path`. No new fitting, runtime, intervals or
promotion. No new external lockbox or self-learning efficacy claim.
