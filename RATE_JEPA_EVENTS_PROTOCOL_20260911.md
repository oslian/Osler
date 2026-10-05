# Shared history encoder plus recorded treatment

Same rate-query potassium data, seven source patient roles, initialization,
32x16x32 sampler, AdamW and EMA teacher as original. Two end-to-end fits per
seed: six separate rate/count/age plus known-mask channels; versus identical
capacity with numeric channels zeroed and masks retained. Source-fit unique
history rows alone define median/std scales. Existing53lab slots unchanged.
Expand existing observation projection with zero event weights; retain clock
as last coordinate. Verify initial predictions within1e-5. No parallel state.

Preserve laboratory-supported update-step masks to isolate event information
from update cadence. Treatment-only steps not tested. Existing retrospective
infusion timestamp caveat remains. No future fields or integrated dose.

Frozen original whole/own predictions are hash-bound comparisons; tune chooses
epoch including zero; hospital/evaluation never fit. Patient MAE and paired
bootstrap1000 versus whole, own, persistence and event support control, all and
bracket-after groups. No calibration, new lockbox, causal or promotion claim.
