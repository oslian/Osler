# Whole-lab inputs restored at rate-record query times

Built shared encoder's existing53-slot history/mask/age schema for the same5526
paired queries/1763patients. Forty variables have at least one supported current
value. Examples: glucose5493,bicarbonate5090,creatinine4719,BUN4716,magnesium3389.
Sparse cortisol20/ferritin13 do not establish learnable support. No imputed truth.
Potassium current values reproduce prior labels within numerical float tolerance.

Raw lab extraction uses existing canonical unit/availability policy. Query and
patient splits unchanged. Arrays include preceding measurement history, not
future outcomes as features. Current support is availability, not validation.
Training must apply per-fit source support and preserve patient roles; unused
bedside slots remain missing. This is not complete body/treatment observation.

No training or promotion. Treatment remains separate, not stuffed into lab
slots. Infusion availability still unverified. Full neural own-vs-whole contrast
on these queries is now feasible but unfinished. Private cache body_arrays.npz,
aggregate artifact `artifacts/rate_anchor_body_inputs_20260911.json`.
Runtime weights, intervals and serving unchanged.
