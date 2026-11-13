# Where bedside numeric gain disappears

Scored existing selected checkpoints/predictions, no refit. Same legal queries,
patient partitions and labels across real bedside numbers, support-only and
lab models. Historical prediction hashes checked and lab controls exact.

| Median patient-equal MAE | Real bedside | Support-only | Lab |
|---|---:|---:|---:|
| Fit |31.728776|32.299238|32.600047|
| Tune |33.746010|33.923375|33.882441|
| New source patients |34.618132|34.875907|34.884272|
| Held-out hospitals |42.328418|41.658324|42.079848|

Real-support paired CI gains/harms: fit3/1, tune0/0, newsource1/0, hospital0/1.
Thus some fit advantage exists but is inconsistent even there; no consistent
gain on new source patients. It is not solely an external hospital issue or
complete failure to fit any numeric association. Selection used tune outcomes,
so tune is not an independent holdout. Fitting CIs are descriptive, not validation.

Bootstrap caution: this diagnostic uses a different fixed resampling salt at
1000draws. The unchanged support-lab hospital count is1gain instead of the
previous2; point errors are identical. A near-zero CI crosses its boundary
under Monte Carlo variation. Do not reinterpret this as changed model ability
or select favorable resampling. None of these results meets a stable7split
increment requirement; global conclusions unchanged. Future formal comparisons
should bind resampling to the same patient cohort and inspect numerical CI
stability near decision boundaries.

The remaining failure is incremental numerical generalization, not a proven
single mechanism. This does not establish that more data, stronger encoders
or a specific repair is guaranteed to help. No training, calibration, runtime
or promotion changed. Artifact: native_bedside_fit_gap_20260912.json in artifacts.
