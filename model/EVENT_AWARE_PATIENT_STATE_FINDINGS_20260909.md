# Event-Aware Patient-State Audit

## Decision

`candidate_only`. The release remains unchanged.

The same frozen `glucose@6h` JEPA source was used in every arm. Forecast
outcomes updated patient state only after their measurement time resolved.
The event-aware arm could read only anchor-known measurement freshness,
physiological change and treatment history; future treatment was excluded.

| Role | Frozen | Error-only | Event-aware | State-only gate | Shuffled-event gate | Wrong-patient |
|---|---:|---:|---:|---:|---:|---:|
| eicu_hospital | 36.265 | 35.879 | 35.588 | 35.589 | 35.487 | 37.721 |
| mimic | 28.944 | 29.218 | 29.090 | 28.930 | 28.950 | 30.901 |

Values are seven-seed means of patient-equal MAE. Event-aware versus
frozen passed only 1/7 eICU hospital comparisons and 0/7 MIMIC comparisons.
It passed 2/7 and 0/7 against unconditional error-only state, and 1/7
and 0/7 against the shuffled-event equal-capacity control. It never
significantly beat the state-only gate in either external role.

## Treatment-transition behavior

On eICU anchors with an observed treatment presence transition, event-aware
direction accuracy was 0.687, versus 0.667 for frozen and 0.653 for error-only state. That improvement did not transfer: on MIMIC the corresponding values were
0.775, 0.798, and 0.755.
The shuffled-event arm was often as good as or better than the real-event arm.
Therefore this contract does not show that treatment events identify the
correct signed update before the future glucose result is known.

Wrong-patient state was clearly harmful on average and event-aware beat it
in 6/7 eICU and 5/7 MIMIC comparisons. Patient identity contains useful
information, but the current event representation does not make its signed
correction stable enough to improve the frozen source across databases.

Promotion required the event-aware arm to beat the frozen source, unconditional
resolved-error state, a state-only equal-capacity gate, shuffled-event context and
wrong-patient state in all seven source-defined seeds on both external roles.
Exact aggregate estimates and subset behavior are in the JSON artifact.

This is factual prediction only; no treatment-effect or causal claim is allowed.
