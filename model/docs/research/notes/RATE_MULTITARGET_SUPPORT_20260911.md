# Additional outcomes on identical rate queries

Built all53schema labels using existing raw body labs at exactly the same5,526
queries. Shared6h +/-1h measurement policy is a diagnostic contract, not an
assertion every variable has the same clinical timescale. Only observed current
and future pairs qualify; missing labels stay missing. Original potassium
current/future/age/actual_delta_t reproduce. Patient roles unchanged. Private
labels retain future IDs; no future IDs become inputs. Aggregate:
`artifacts/rate_multitarget_labels_20260911.json`.

Glucose support5,438pairs/1,740patients, hospital190patients; bicarbonate
4,539/1,483, hospital147; creatinine4,159/1,291, hospital142. Thus existing
native event paths can be tested against additional outcomes without another
cohort or architecture. Auxiliary supervision remains untrained here. Sampling
is still conditioned on original potassium pair eligibility and must be stated.

Separately, the original native histories have known rates in all5,526queries;
5,525contain a numerical change, median13records/6changes/range5.5. This does
not establish independent exposures or effect identifiability, but rules out
the blanket claim that real rates are constant throughout these histories.

Next compare own/whole/event support on a supported additional target using
unchanged source roles and source selection. Do not extrapolate the potassium
negative result to every target or use label availability as a forecast feature.
No training, runtime, calibration or promotion change.
