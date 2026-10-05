# Native event JEPA matched accuracy result

All14 fits completed normally: seven fixed splits, native_real/native_support,
same5,526 rate-record query pairs and original source laboratory eligibility.
All32 report bindings verified after completion. Aggregate is
`artifacts/rate_anchor_jepa_native_20260911.json`; protocol is
`NATIVE_RATE_JEPA_TRAIN_PROTOCOL_20260911.md`.

| Median patient-equal MAE | Snapshot whole | Native real | Native support | Own | Persistence |
|---|---:|---:|---:|---:|---:|
| New source patients | .384183 | .387409 | .387344 | .417130 | .496675 |
| Source bracket-after | .393479 | .399871 | .399824 | .426164 | .508829 |
| Held-out hospitals | .386973 | .389022 | .389024 | .402915 | .482238 |
| Hospital bracket-after | .413968 | .426119 | .426118 | .437972 | .523661 |

Native real versus snapshot whole: significant improvement0/7 every group;
harm0/7 except hospital bracket-after1/7. Native real versus support improves
0/7 every group; harms1/7 source both groups and hospital all,0/7 hospital
bracket. Native real versus own improves6/7 source both groups,0/7 hospital
both. Native real versus persistence improves7/7 every group, no harm.
These are overlapping split comparisons, not independent replications.
Differences of medians above are not median paired effects.

Both arms selected epochs29,23,21,24,27,20,7. Selection did not hit32, but this
does not prove optimization convergence. The source gain beyond own remains;
native sequence is not a stable additional improvement over snapshot whole.

## What this localizes

Previously untested native-event updating is now trained, not merely wired.
Keeping recorded intermediate events and activating known-rate-only updates
does not by itself repair the prediction gap under this matched warm-start
contract. The treatment-number increment remains tiny versus matched support.
Do not keep proposing this exact native switch as an untested solution.

It does not prove that true treatment dynamics lack information. Treatment
record time is still not independently verified administration/availability;
only the captured insulin-rate stream is included. Existing clipped168h clock,
fixed48h history, warm-start encoder, and single-target supervision remain.
Cadence and sparse-event encoding change together, so the result does not
identify cadence alone as harmful. Nor is it a test of complete persistent
organ-state dynamics, physical fluxes, or all41 forecast cells.

The next diagnostic should inspect native-state learning and time/support
semantics rather than enlarge decoders or assume missing events were the sole
cause. Any further fit must retain this negative control and source selection.
Pretraining patient independence remains uncertified; no new external lockbox,
conformal, retention, runtime, or promotion validation occurred. Existing
release weights, source registry and intervals are unchanged.
