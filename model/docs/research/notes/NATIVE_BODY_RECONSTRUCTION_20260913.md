# Persistent Body Masked Supervision Integration

Added NativeBodyReconstruction around the SAME native anatomical trajectory.
It reads anchor body state (zero future horizon), not another patient model.
Target and existing related_siblings are erased from the entire event history;
target-only events removed, organ/module presence rebuilt. No independent
remaining measurements means rejection, not fabricated reconstruction. Forecast
heads retain owner-only graph isolation and are not used to generate nowcasts.

Four tests pass: whole-history target perturbation cannot alter output; body
update/dynamics gradients positive; empty independent history rejected; brother
variables and target-only events removed. Real source MAP pair smoke-step with
45%-of-hour measurement age (.45h) yields bodyupdategrad.038219 and bodydynamics
grad.077158. Truth is fresh anchor measurement, never future label. No weights
saved, no efficacy evaluation or promotion. This repairs the missing supervision
path structurally, not yet proof of useful learned body state.

Next training must compare real versus capacity-matched shuffled patient input,
fit-only population baseline, source-disjoint selection, and heldout masked
reconstruction. Do not treat current smoke-gradient as whole-body validation.
The single body representation and forecast safety behavior remain unchanged
in serving; only the candidate integration is new.

Evidence: `artifacts/native_body_reconstruction_smoke_20260913.json`.
