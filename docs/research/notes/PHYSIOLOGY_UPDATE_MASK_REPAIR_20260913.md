# Persistent Observation Update Mask Repair

## Reproduced Defect

`PersistentPhysiologyStateLayer.update` masked the organ gate after projection
but never masked the body correction. With all organ-presence values zero,
an arbitrary absent observed-body value still changed the body. A fixed seed7
counterexample moved body by98 despite zero elapsed time and no observations.
Absent NaN organ values also propagated through projection and multiplication
by zero, producing nonfinite states.

The correction now validates shapes/presence/time, ignores missing values before
projection, rejects nonfinite present observations and applies body correction
only when at least one organ observation is present. Unobserved organ states
and patients are unchanged, including gradients through missing inputs.
No artificial zero measurement is inserted. Existing elapsed-time behavior is
unchanged; this patch does not implement event-stream prediction/assimilation.

## Scope

This is a real component-contract defect, not the demonstrated cause of recent
MAP failure. Those trials use NativeEventFilter, not this layer's update.
PersistentWholeBodyJEPAWorldModel and OrganExchangeResidual use this layer to
initialize and roll out future states; their inspected forward paths do not
call observation update. Hence fixing this method alone cannot claim to have
repaired those trained models' accuracy or integrated full-body online learning.

21 tests pass across physiology update masks, physiology state and organ
exchange. New tests include all-missing finite/NaN inputs, partially observed
organs, patient isolation, masked gradients and invalid present inputs.
No checkpoints, registry or calibrators were changed. Historical reports bound
to the old source must remain historical; this source patch is not retroactive
validation of a new state-update behavior.

Next integration must combine a real chronological observation stream with
organ-scoped correction and separate time evolution. Do not describe existing
rollout-only experiments as if they already exercised this update contract.
