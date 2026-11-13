# Broad Encoder Exposure Audit

Read actual training code, epoch logs and checkpoint `seen` sets. No training
or model changes. Existing source sampling is bounded0.5-2x, unlike the later
uncapped component-equal correction weights. Do NOT transfer that diagnosis
to the encoder.

Each nominal epoch is1072replacement draws per47targets,50384updates total.
Eight epochs give403072updates, not eight passes over the fit dataset.

| Independent fit fold | Eligible pairs | Unique seen | Fraction |
|---|---:|---:|---:|
|0|698766|236509|0.33847|
|1|698510|236632|0.33877|
|3|682000|230194|0.33753|
|4|743818|243475|0.32733|

Fold2uses certified identical fold0 training; it is not another independentfit.
Common-target exposure is lower still: glucose0.253-0.260, heart_rate0.249-0.257,
temperature0.249-0.254. Only2-4targets/foldhave all eligible pairs seen.
All these are unique patient-target exposure, NOT fraction of raw measurement
events; a seen pair carries an entire history. Never claim unseen pairs imply
those patients never contributed to other target training.

The script uses fixedepoch8, not a selection-best checkpoint. Per-epoch losses
are on changing replacement samples and therefore cannot establish convergence.
Epoch8loss versus epoch7 increases in allfourindependentreal fits, but that is
not evidence of overfitting without fixed-query measurements. Earlier small
cohort training/exposure experiments remain negative/partial; this is a larger
cohort exposure audit, not proof a previously rejected idea is now successful.

Next discriminating test: on frozen source-only probes compare already-seen
and not-yet-seen pairs with target/component support reported; then compare
the original replacement schedule against a coverage-aware schedule with the
SAME update budget and matched process-only control. Keep target balance,
split identities and outer data fixed. Do not interpret a longer-budget run
as an isolated sampling repair. Separate population exposure from convergence.

No promotion, runtime, intervals, forecast or checkpoint changes.
