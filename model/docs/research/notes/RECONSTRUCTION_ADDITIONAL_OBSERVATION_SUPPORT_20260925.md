# Exact Masked-History Observation Support

Extracted all1,800,214cached patient-target histories,73,670people47targets.
Payload SHA validated for every row; exact one-to-one identity join against all
five outer prediction files verified, no missing or duplicate rows. No numeric
truth/label used as a feature. No sampling or changed observation contracts.

Metrics: available event steps, observed variable events, distinct variables,
variables measured within1h/6h, median/max latest measurement age, retained
history span and latest availability age. Measurement age = anchor minus
event availability plus token age*24; availability age alone is not freshness.
Target hidden assertions and finite/time/mask checks run for every history.
Sibling masking is inherited from the verified immutable cache, not newly
re-derived by this summary extractor. Observedzero remains present via masks.

## Illustrative Unmatched Component-Equal Means

These are posthoc descriptive examples, NOT matched input states or statistical
tests. Median-age column is the mean of each history's median latest age.

| Target/fold | Role | Variables | Fresh1h | Median age h | History span h |
|---|---|---:|---:|---:|---:|
| O2sat3 | calibration |18.025|3.513|10.703|26.846|
| O2sat3 | outer |18.397|3.887|10.584|29.561|
| Albumin2 | calibration |22.526|18.721|0.802|16.699|
| Albumin2 | outer |22.388|18.897|0.811|21.030|
| MAP4 | calibration |17.516|3.346|9.201|24.065|
| MAP4 | outer |17.581|4.152|6.996|17.447|

Albumin's prior residual direction flip does not coincide with a large loss of
average input count or freshness in this comparison. MAP has a different history
span but fresher observations in outer. These facts do not prove source/outer
physiology matched; variable identity, distributions within bins and treatment
context remain unresolved. No universal "older/fewer inputs" diagnosis follows.

Two tests cover zero-vs-missing, measurement/availability timing, rejection of
future/target-visible histories and invariance to values/targettruth. Next use
source-defined bins with these pre-anchor metrics to compare support, retaining
all unmatched cases and not fitting corrections to outer labels.

No training, runtime, promotion, interval or forecast change. Private output is
not committed. Aggregate metadata:
`artifacts/native_reconstruction_additional_observation_support_20260925.json`.
