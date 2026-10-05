# Conditional Current-State Readout Result

Three matched eight-epoch arms completed on64fit/32tune patients with the same
larger head shapes: current+change, change-only, anchor-only. Four invariant
tests pass (zero-start, future exclusion, clock precision, state gradients).

Conditional source tune selects epoch1 MAE8.34492 versusbaseline8.34714.
Controls selectepoch0. Epoch8 scores8.39667/8.41001/8.41138 respectively:
conditional input partly changes learning but later tuning still worsens.

Selected conditional versus baseline on64heldout patients each:
- New source MAE8.46406 vs8.46155; delta+.002514 CI[-.027509,.032998].
- Hospital MAE10.72901 vs10.72190; delta+.007106 CI[-.023850,.037217].

Neither gain nor harm established. Current-state access does not provide a
verified repair in this small pilot. It weakens the claim that omission of
current state alone caused failure. Small support, training dynamics and useful
state supervision remain unresolved, not proof no useful information exists.
Previously inspected roles are not independent lockboxes. No multiple-target,
seven-seed, conformal or runtime promotion. Original bound candidate retained
only as experimental comparator; release unchanged.

Evidence: `artifacts/native_organ_conditional_pilot_20260913.json`.
