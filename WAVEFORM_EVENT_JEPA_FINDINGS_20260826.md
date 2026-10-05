# Waveform JEPA Event-Risk Findings

## Status

Candidate only. The runtime event probability remains closed. This work does
not change the validated table event heads and grants no causal, clinical, or
treatment authority.

## Question

Can high-frequency pre-event waveforms, bedside assessments, and observed
treatment intent add stable signal for a new sustained-hypoxemia event, rather
than forcing unpredictable deterioration into a continuous-value forecast?

## Data and Event Contract

- MIMIC-IV Waveform Database v0.1.0 linked to MIMIC-IV v3.1.
- 1,578 anchors, 161 patients, 162 ICU stays, and 8 care units.
- Zero waveform read failures.
- Inputs are strictly at or before the anchor.
- A stable anchor has no SpO2 at or below 92% in the prior ten minutes.
- An event requires two SpO2 readings at or below 92%, separated by 30 seconds
  to five minutes, inside the future horizon. A one-sample monitor artifact is
  not an event.
- Row-level cohort data remain outside the repository.

## New Observability

The candidate path now includes:

- 60 numeric history features;
- 1, 5, and 15 minute ECG, pleth, respiration, and arterial-pressure features;
- GCS, oxygen device, FiO2, PEEP, ventilator mode, and respiratory effort;
- pre-anchor POE intent for oxygen therapy, mechanical ventilation,
  extubation, BiPAP, and CPAP;
- observed invasive and non-invasive ventilation context.

The POE mapping initially used `order_type` for the intervention name. The
actual MIMIC schema stores `Respiratory` in `order_type` and the intervention
name in `order_subtype`. This was corrected before the formal audit. In the
final cohort, recent intent was present for oxygen (92 anchors), mechanical
ventilation (260), extubation (47), and non-invasive support (30).

## Architecture

```text
pre-anchor numeric history -> context encoder -----------+
                                                         |
1m/5m/15m waveform -> measurement-pure waveform encoder  +-> future latent predictor
                                                         |            |
future SpO2 trajectory -> EMA target encoder ------------+            +-> event head

waveform quality + model disagreement + external lockbox -> abstention gate
```

The physiology encoder excludes arbitrary ECG/pleth/respiration amplitudes,
lead counts, missingness, and flatline features. Those acquisition-process
features are used only by the quality/abstention gate. Arterial-pressure levels
remain because they have physical units.

## Formal Results

| Horizon | Events | Patient splits significantly better than table | Better than placebo | Internal promotion |
|---|---:|---:|---:|---|
| 15 minutes | 241 / 1,578 | 0 / 7 | 2 / 7 | no |
| 30 minutes | 455 / 1,578 | 1 / 7 | 2 / 7 | no |
| 60 minutes | 743 / 1,578 | 0 / 7 | 0 / 7 | no |

The original all-waveform feature set showed a forward-time waveform-only
gain, but failed care-unit transfer. Removing acquisition-process features
removed the obvious care-unit degradation but did not create a stable
incremental gain. This means the remaining signal is weak and heterogeneous,
not simply hidden by one bad feature family.

## Interpretation

- High-frequency waveform information is real enough to produce isolated
  positive splits, but it does not reliably improve the already strong recent
  SpO2/numeric baseline in this 161-patient waveform subset.
- The 15-minute task is dominated by the recent SpO2 baseline. At 60 minutes,
  unobserved future care changes and acute events increasingly limit
  identifiability. Thirty minutes did not provide a stable middle ground.
- Bedside assessment and treatment intent improve observability, but their
  relation to future events varies across patients and care units.
- MIMIC waveform is single-center. A portable event definition is implemented,
  but cross-hospital waveform validation is not complete.

## Runtime Boundary

The candidate must abstain when any of the following holds:

- pleth quality is insufficient;
- neither ECG nor respiration provides corroborating waveform;
- ensemble disagreement is above the fixed limit;
- probability calibration has not passed;
- an independent external-hospital waveform lockbox has not passed.

Because the external waveform lockbox is currently missing, the new waveform
JEPA always abstains at runtime. It does not guess a precise number or replace
the validated table event-risk head.

## Reproduction

```bash
python scripts/build_waveform_event_cohort.py \
  --waveform-root /path/to/mimic4wdb/0.1.0 \
  --mimic-root /path/to/mimic-iv-3.1 \
  --output /tmp/mimiciv_multimodal_hypoxemia_cohort.parquet \
  --report artifacts/multimodal_hypoxemia_cohort_report.json

python scripts/relabel_waveform_event_horizons.py \
  --cohort /tmp/mimiciv_multimodal_hypoxemia_cohort.parquet \
  --output-dir /tmp/waveform_event_horizons \
  --report artifacts/waveform_event_horizon_support.json \
  --horizon-minutes 15 30

python scripts/train_waveform_event_jepa.py \
  --cohort /tmp/waveform_event_horizons/mimiciv_multimodal_hypoxemia_30m.parquet \
  --event-name incident_sustained_hypoxemia_30m \
  --output models/waveform_event_jepa_30m_candidate.pt \
  --report artifacts/waveform_event_jepa_30m_audit.json
```
