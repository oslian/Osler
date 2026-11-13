# Fixed order-issuance diagnostic

Same6174pairs and source seven patient partitions, same own/full lab event-window
baselines and28fits as POTASSIUM_TREATMENT_CONTEXT_PROTOCOL_20260911.md.
Only difference: medication order events instead of infusion snapshots.
No combined source sweep or external evaluation; no model/threshold selection.

Use drugorderoffset, medicationID and classified drugname only. Do not filter
on drugordercancelled, nor read dosage, start/stop, route or inferred delivery.
Features: log count and most recent order age in strict previous6h; dimensional
rate fields remain unknown. Include eventually canceled orders since their
cancellation time is unknown. Their issuance is not evidence of administration.
No pharmacy-verification/storetime exists here: retrospective event-time only.

Same capacity, leaf selection, patient weights, local-donor placebo and paired
bootstrap comparisons as the frozen infusion protocol. Source role only;
require all seven for stable diagnostic improvement over baseline and placebo.
No runtime or calibrator changes. Missing donor support remains a placebo
limitation; multiple exploratory comparisons do not qualify as promotion.

Source contract: https://eicu.mit.edu/eicutables/medication/
