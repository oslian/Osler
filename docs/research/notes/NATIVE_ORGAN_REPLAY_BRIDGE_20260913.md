# Native JEPA Anatomical Replay Bridge

Availability-native measurement tokens now feed the same JEPA token/context
weights, then anatomical observation proposals, then the persistent chronological
state layer. All53 release variables have an existing direct canonical mapping.
No functional-system token is renamed as an anatomical organ.

Local organ proposals use observed pre-mixer variable tokens; other-organ values
cannot contaminate a local proposal via full-body attention. Shared body proposals
still use the existing JEPA context path on each event. Its distribution differs
from the trained historical context and requires supervised retraining, not
assumed checkpoint compatibility. There is one persistent body with14organ slots;
identity graph and body feedback disabled keep unvalidated organ edges closed.

Eight source-fit patient streams,112real grouped events,3-8supported organs per
stream, elapsed0.0167-45.57h, successfully replayed. Every smoke objective and
gradient was finite. JEPA token-projection gradient norms0.0300-0.0884 and organ
update-gate gradients0.1035-0.2233 demonstrate a connected trainable path.
The objective was only final state squared norm, with no optimizer step: this is
NOT future supervision, clinical model training, predictive value or promotion.

Original recovered clocks and report dependencies were checked. Times are
relative to each stream start. Source-fit numeric transforms were reused.
Treatment fields are deliberately excluded because this bridge is measurement
only; uncontracted treatment input is rejected, not treated as administration.
Missing observations cannot masquerade as observed zero; direct owner mapping
and pre-mixer isolation have focused tests. No raw rows or patient IDs exported.

Remaining: connect an explicit future state-change readout and real-label JEPA
training, matched unscoped/scoped controls, then patient/hospital and calibration
gates. No release weights, registry or interval changes.

Evidence: `artifacts/native_organ_replay_bridge_20260913.json`.
