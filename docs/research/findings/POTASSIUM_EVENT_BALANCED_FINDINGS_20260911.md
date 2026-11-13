# Event-balanced potassium result

Completed all14 fits (seven seeds x own/whole,64 epochs). Source labels reproduce
exactly. All legitimate queries retained; only patient/event/query training
sampling differs. No future event IDs enter the model. No external data read.

| Source-tune comparison | Significant improvement |
|---|---:|
| Event-balanced own vs original own | 0/7 |
| Event-balanced whole vs original whole | 0/7 |
| Event-balanced whole vs own | 0/7 |
| Event-balanced whole vs persistence | 7/7 |
| Event-balanced own vs persistence | 6/7 |

No significant harm versus originals in either arm. Median patient-equal MAE
own0.364404, whole0.355709 versus original0.366092/0.348620. Whole numerically
beats own in all seven splits but all paired confidence intervals cross zero.
The persistence comparison crossing7/7 is not evidence of significant benefit
over the original6/7 method: direct intervention comparisons all fail.

Thus repeated-outcome weighting is a real sampling property, not an established
principal cause or an accuracy repair. Keep candidate-only. Source-tune patients
also select checkpoints; seven splits overlap, and warm-start pretraining
overlap is not excluded. No external confirmation, calibration or promotion.
Weights remain private. Runtime weights, registry and serving unchanged.

Artifact: `artifacts/potassium_event_balanced_20260911.json`.
Protocol: `POTASSIUM_EVENT_BALANCED_PROTOCOL_20260911.md`.
Command: `python -m scripts.train_potassium_matched_encoder --event-balanced`.

## Failure location, not an invented single cause

The earliest measured glucose extra-context failure remains fitting-patient
benefit that reverses on new source patients, before database transfer. The
potassium matched control is different: weak positive source differences fail
to achieve stable incremental benefit, including frozen external evaluation.
Already demonstrated amplitude and observation-contract defects explain part,
not all, of the failure. Neither MAE loss alignment nor event balancing repairs
the remaining incremental-context gap. Do not substitute a new architecture
guess for evidence or assert that all extra physiological information is absent.
