# BUN Readout Bias Stationarity

Freeze all predictions and solve one scalar additive BUN bias for the actual
SmoothL1(beta1) data loss. Weights reproduce expected bounded-site sampling
within each source-fit group, with equal group mass. Source-fit529/9664pairs;
all hashes bound to preceding signed-prediction audit. Existing nn.Linear
readout has independent output biases, so this direction need not change
other target outputs or shared state. This audit computes rather than installs it.

Weighted derivative at original bias=.19394515, after optimum approximately0.
Optimal shift=-.259876469(normalized units).

| Expected source-fit metric | Original | Bias optimum |
|---|---:|---:|
| MAE |.889196523|.833171053|
| SmoothL1 |.517783507|.492410300|

Both earlier median-offset candidates also improve BOTH data losses relative
to original, although SmoothL1 optimum differs from MAE optimum. Therefore an
MAE-versus-SmoothL1 tradeoff alone cannot explain this checkpoint's BUN offset.
It is not stationary along this available readout-bias direction under expected
data risk. This does not prove global optimizer failure: finite stochastic
updates, early checkpoint selection, weight decay and changing shared features
remain possible contributors. Historical draw losses are not recomputed here.

Selection broad3086: MAE .893611644->.828587826 (7.28%gain), deltaCI
[-.0735699,-.0562241]; SmoothL1 delta-.0357992CInegative.
Selection selected90: MAE1.105465044->1.120230268, deltaCI
[-.0368409,.0689275]; SmoothL1delta+.02485CIcrosszero. Reused source-selection,
not independent new external evidence. The training repair does not establish
uniform generalization. Do not install correction or relax gates.

## Next Legitimate Repair

Rather than more outcome-tuned BUN calibrators, assess a fixed final readout-bias
optimization stage across all trained targets on source-fit alone, preserving
the single body state. Each target still requires retention and independent
evaluation. Separate whether the selected checkpoint/readout combination is
poorly optimized from whether the resulting correction transfers. No broad
claim about future forecasting follows from masked current reconstruction.

Two root/invalid-weight tests pass; audit exit0. No runtime, model weights,
registry, intervals or promotion modified. Evidence:
artifacts/native_reconstruction_bun_bias_stationarity_20260920.json.
