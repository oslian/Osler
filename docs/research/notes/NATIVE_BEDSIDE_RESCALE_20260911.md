# Bedside source-coordinate retraining

The input-scale inventory found a relatively broad temperature tail in inherited
coordinates (seed7 fit about9.2% abs(z)>5; hospital7.1%). Other added channels
were less extreme. This is not proof of a conversion bug or impossible values.

Ran14matched fits using --rescale-bedside: same labels, roles, source selection
and32epochs. Only five newly added numerical coordinates are centered/scaled
from unique source-fit history observations; their masks, ages and event cadence
are unchanged. No target rescaling or external tuning. All seven lab controls
are bit-exact to the previous trial and44bindings verified. Numeric transform
metadata is saved with each candidate checkpoint.

| Median patient-equal glucose MAE | Lab | Bedside inherited | Bedside source-rescaled |
|---|---:|---:|---:|
| New source patients |34.884272|34.250550|34.618132|
| Held-out hospitals |42.079848|42.881423|42.328418|

Rescaled-bedside versus lab paired95% CI gains/harms out of7: source1/0,
hospital0/1. Versus persistence:7/0 in both roles. Hospital median is numerically
closer to lab than the inherited-bedside arm, but no stable incremental benefit
is established. Counts across overlapping seeds are not independent trials.

Conclusion: source rescaling alone does not repair added-context generalization.
Do not equate a wide z tail with the sole cause. Event-frequency versus actual
numeric-information effects remain unseparated; fixed-cadence numeric ablation
is a diagnostic, not yet demonstrated repair. Full treatment/assessment input
and untouched external validation are also not supplied by this experiment.

No runtime, interval, registry or promotion changed. Artifacts:
native_bedside_scale_20260911.json and native_bedside_rescale_training_20260911.json
under artifacts. Historical unscaled script hash remains historical.
