# Target-private update allocation

Fixed native glucose cohort, labels, patient roles and seven seeds, same32epochs,
patient-uniform16x32 updates/epoch, AdamW1e-4, same value plus JEPA loss and
source-tune selection including initialization. Compare own/whole input views.
Train only glucose target_future_predictors, target_temporal_adapters and
value_heads. Freeze every other parameter and EMA teacher; no shared update,
new architecture, retention penalty or tuning on hospital results.

This differs from the old context-frozen pilot, which still trained shared
future modules. It is an allocation diagnostic on the base world-model class,
not a claim to implement the full persistent-organ architecture. Native new
event projection stays at zero initialization, so this does not test learned
rate conditioning. Existing observed lab history still drives the frozen state.
Teacher EMA is deliberately fixed with its encoder, not accidentally omitted.

Compare source evaluation/hospital patient-equal MAE and paired95%CI against
matched private-own, original trainable-own/full, persistence. Check all other
52 outputs exactly equal initial and every nonprivate tensor unchanged.
Preservation alone is not success: initial other forecasts can be inaccurate.
Same prior reused data, no new lockbox, no interval calibration or promotion.
This diagnostic does not substitute for eventually learning a joint body state.
