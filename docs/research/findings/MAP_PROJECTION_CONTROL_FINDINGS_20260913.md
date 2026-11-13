# MAP Projection Content Control

## Seven-Split Completion

All seven fixed splits completed32epochs; 84 direct report dependencies
hash-verified. No settings changed after seed7. This supersedes the six-pending
status in the historical pilot below.

| Seven-split median MAE | New source patients | Held-out hospitals |
|---|---:|---:|
| Own history | 8.84140 | 10.21067 |
| Full history | 9.22298 | 10.57510 |
| Numeric support control | 9.23789 | 10.61140 |
| Target-only projection, full cadence | 9.22322 | 10.60673 |

Projection versus support control: zero significant gains or harms in either
role; median paired differences -0.00110 source/-0.00233 hospital. Versus full:
also zero gains/harms. Versus own: significant harm7/7 in both roles, median
paired +0.37953/+0.35811. Versus persistence: gain7/7 in both roles.

Removing non-target projection content is not a sufficient repair under this
matched training policy. The harmful expanded-history gap does not require
that content to enter observation_projection. Remaining activation cadence,
clocks, current-query support and optimization interactions are not individually
identified. No-significance is not equivalence proof. Do not repeat this as an
untested metadata-removal fix. A subsequent scheduling intervention must be
training-matched and must not be represented as a final single-organ model.

These are reused source/hospital partitions with overlapping seed cohorts,
not seven independent external datasets. No runtime, interval or promotion
changed. Evidence: `artifacts/map_projection_control_{seed}_20260913.json`,
seeds7,19,31,43,59,71,83.

## First Fixed Split Complete

Seed7 ran all32 epochs with original initialization, patient draws, source-only
transforms, loss and tune selection. No architecture capacity or query change.
The observation-projection hook strips non-target support/content, but the
filter still consumes every original active event and its original clock.
This separates projection content from scheduling more narrowly than the
previous history/query factorial. It is a diagnostic, not a deployable repair.

| MAE | New source patients | Held-out hospitals |
|---|---:|---:|
| Own history | 8.80806 | 10.21067 |
| Full history | 9.16712 | 10.56384 |
| Numeric support control | 9.10062 | 10.54625 |
| Target-only projection, full cadence | 9.09952 | 10.55518 |

Projection versus support control: -0.00110 CI[-0.02432,+0.02134] source;
+0.00892 CI[-0.03264,+0.05005] hospital. Neither significant. Against own,
harm remains: +0.29146 CI[+0.14028,+0.43911] source and +0.34450
CI[+0.06255,+0.62648] hospital. Persistence is beaten in both roles.

This first split does not support removal of non-target projection metadata as
a sufficient repair. It does not prove equivalence, a unique cadence cause,
or that biological cross-organ values lack predictive information. The control
retains extra updates, complete current-query support, original clipped clocks
and training interactions. Six prespecified seeds remain unrun; no settings
will be selected using this pilot's held-out results.

Evidence: `artifacts/map_projection_control_7_20260913.json`. Checkpoint and
row-level predictions remain private. Three focused tests verify schema,
gradient flow and exact recurrence with unchanged activation after projection
content removal. No runtime, calibration or promotion changed.
