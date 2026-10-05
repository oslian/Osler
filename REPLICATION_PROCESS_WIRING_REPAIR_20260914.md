# Replication Process-Control Wiring Repair

The initial seed19 epoch1 replication is INVALID as a trained process-control
comparison. Its fit call used raw numbers while its tune call neutralized them.
Its real/process training loss was exactly identical, exposing the mismatch.
This defect is confined to the new replication driver; the earlier seed7
process runner neutralized samples before both paths and is unaffected.

Preserve the initial report and weights for audit. V2 uses one input_tokens
function for BOTH fit and tune calls, with numeric neutralization only in process.
Regression tests verify views and both forward call sites. V2 writes distinct
reports/checkpoints and reruns all arms from initialization, so there is no
claim that invalid process optimizer state can be resumed.

The original eight-epoch/partition/support protocol is otherwise unchanged.
No runtime models were affected by this candidate-only defect.
