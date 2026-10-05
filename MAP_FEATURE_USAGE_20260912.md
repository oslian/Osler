# MAP Fitted Feature Usage

All seven native-missing full models and parent artifacts verified. Counted
actual internal split nodes and summed nonnegative recorded training split
gain by feature block and variable. Fitting-patient support is the number of
distinct fitting patients with any observed history of a variable.

Variables supported by fewer than 30 fitting patients account for zero split
gain in all seven models. Thirty is a descriptive audit boundary, not a new
training rule. Therefore deleting those rare channels cannot be justified as
the primary repair from these models' direct split usage. This does not rule
out thin support for combinations of common variables.

Median normalized split gains: numbers82.01%, explicit masks0.05%, ages17.90%.
MAP45.85%, anion gap7.00%, bicarbonate2.97%, glucose2.59%. These are training
usage summaries, NOT causal importance or held-out efficacy. Numeric NaN
routing also carries missingness; numeric gain cannot be called pure physiology.
Correlated features and separately fitted models preclude causal fractions.

The prior seed7 density audit reports anion-gap current support5.3% in fit
versus25.3% hospital, despite actual split use here. This identifies a supported
candidate for further investigation, not evidence it causes the error. Do not
choose a removal/routing policy based on inspected hospital outcomes.

This audit excludes a simplistic "very rare variables dominate tree splits"
explanation, not all support shift, combinations, or sampling problems. The
numerical increment still lacks held-out validation. No training, runtime,
interval or promotion changes. Evidence: `artifacts/map_feature_usage_20260912.json`.
