# MAP Predict-Only Event Control

## Verified Mechanism

The original recurrent update makes an innovation on every active event,
including events whose target-only diagnostic projection contains no target
measurement. A separate candidate-only filter now distinguishes elapsed-time
prediction from measurement correction. Non-target events advance exponential
decay but do not invent an observation correction in this diagnostic.

This is not a declaration that non-target real observations should be ignored
by a full-body model. They deliberately contain no target content here. A
full-body repair still needs appropriately scoped corrections with preserved
cross-organ information, trained and evaluated against the complete gate.

## Real-History Replay

Seven existing own-history/control-query factorial checkpoints; 96 distinct
source-fit patients per split, one input-selected query each. Original saved
predictions reproduced. Same parameters and full query; added full-schedule
events with target-only projection and predict-only treatment of absent target.

671/672 sampled histories have matching last active and target endpoints.
On those, maximum MAP forecast difference versus own-history reference is
0.0000076294, within float32 tolerance. Mean additional events range46.32-53.98
across the seven probes. One history with a later non-target event was excluded
from endpoint-equivalence assertion because further decay changes the endpoint;
this exclusion uses input times only, never outcomes. Splits overlap and672
is not a count of unique patients across splits.

Three tests establish preservation with inserted no-content events, explicit
decay after the last observation, and zero state for padded empty history.
The exponential composition test is bounded to monotone intervals below the
inherited delta clamp. It does not solve all time representation contracts.

## Decision

This establishes a working prediction/correction separation and structural
equivalence on real sampled histories. It is not new training, a measured MAE
improvement over full weights, proof of a sole failure cause, or promotion.
The earlier projection-content training remains a negative result. Do not
describe the seven replays as seven new trained whole-body models. Runtime,
weights, calibrators and registry remain unchanged.

Evidence: `artifacts/map_predict_only_control_20260913.json`.
