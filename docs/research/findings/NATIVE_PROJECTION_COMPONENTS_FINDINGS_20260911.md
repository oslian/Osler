# Native observation projection scale

Source-fit unique native tokens only, seven frozen glucose-real checkpoints.
All original bindings verified. Decompose contributions to the FIRST LINEAR
layer, before LayerNorm/SiLU. This is a coordinate diagnostic, not forecast
error attribution, an explained-variance fraction, or a promotion gate.
Aggregate: `artifacts/native_projection_components_20260911.json`.

| Component | RMS | RMS after subtracting component mean |
|---|---:|---:|
| Lab values | 1.549295 | 1.547315 |
| Missing ages | 2.833008 | .326389 |
| Observed ages | .001348 | .001333 |
| Observation masks | .085766 | .077816 |

Medians across seven source fits; unique-event rather than patient weighting.
Components are correlated; do not sum these RMS values or call them independent
information shares. The actual input adds clock, presence, events and bias.

The raw scalar encoder normalizes each lab with a fixed variable-specific
median/scale, independent of panel size. No panel-dependent raw numeric
rescaling is present in encode_group. Later context dependence in a learned
Linear/LayerNorm is not in itself a preprocessing bug.

The missing-age sentinel has a large mostly constant projection. It is NOT
valid to conclude it drowns out varying labs from its uncentered norm: after
centering, the observed lab term varies more. Observed-age contribution is
small at this layer, but input range and scaling also determine its norm;
this is not proof the model ignores time or that multiplying age improves MAE.

Indeed the sentinel alone is algebraically removable without changing a
linear output: age_old = age_zero + 7*(1-mask), so bias_new = bias_old +
7*sum(W_age), and W_mask_new = W_mask_old - 7*W_age. Thus it does not by itself
destroy information. Retraining dynamics/regularization can still differ.
No numerical reparameterization, runtime change or accuracy repair was made.

No evidence here warrants another arbitrary rescaling intervention. Keep the
known native glucose fit-to-new-patient reversal distinct from coordinate
magnitudes. Treatment availability and inherited pretraining limits persist.
