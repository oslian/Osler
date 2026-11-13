# Raw Waveform JEPA Findings (2026-08-26)

## Question

Can pre-anchor ECG, plethysmography, respiration, and arterial-pressure
morphology predict incident sustained hypoxemia more reliably than table data
or the existing 60-second waveform summaries?

## Data and safeguards

- Source: local MIMIC-IV Waveform Database matched cohort.
- Eligible tensors: 1,573 anchors from 161 patients; 5 of 1,578 anchors failed
  raw sequence extraction.
- Context: the final 1, 5, and 15 minutes before each anchor.
- JEPA target: the first post-anchor minute of raw waveform.
- Event horizons: 15 and 30 minutes.
- Event definition: no SpO2 <= 92 in the preceding 10 minutes followed by two
  SpO2 <= 92 measurements separated by 30 seconds to 5 minutes.
- The self-supervised encoder is fitted inside each training split only.
- Patient splits, care-unit holdout, and forward-time holdout are patient
  disjoint.
- Comparators: measurement-pure table model, existing waveform-summary model,
  and a patient-shuffled raw-waveform placebo with identical capacity.

This remains factual event-risk estimation. It does not estimate treatment
effects and has no clinical decision authority.

## Results

| Horizon | Raw beats table (7 splits) | Raw beats summary | Raw beats shuffled placebo | Internal gate |
| --- | ---: | ---: | ---: | --- |
| 15 min | 2/7 | 3/7 | 4/7 | Fail |
| 30 min | 1/7 | 0/7 | 1/7 | Fail |

### Patient-disjoint care-unit holdout

At 15 minutes, raw JEPA Brier score was 0.1464 versus 0.1498 for table,
0.1511 for summary, and 0.1468 for shuffled placebo. Raw JEPA significantly
beat the summary model, but did not significantly beat table or placebo. Its
ECE was 0.0562, above the 0.05 gate.

At 30 minutes, raw JEPA Brier score was 0.2069 versus 0.2109 for table,
0.2122 for summary, and 0.2125 for placebo. It significantly beat summary and
placebo, but not table. Its ECE was 0.0581, above the gate.

### Patient-disjoint forward-time holdout

At 15 minutes, raw JEPA did not significantly beat any comparator. At 30
minutes it significantly beat the shuffled placebo, but did not beat table or
summary. Calibration passed at both horizons, so discrimination stability,
not only calibration, is the limiting issue.

## Interpretation

Raw morphology contains information that the 60-second summaries discard:
the 15-minute model beat summary and placebo in several patient splits, and
the 30-minute model beat both on care-unit holdout. The signal is nevertheless
not stable enough across patients, care units, and time to replace the table
model.

This rejects two overly simple explanations:

1. The earlier negative waveform result was not solely caused by summary
   feature extraction; raw morphology recovers some signal.
2. Raw waveform access alone is not sufficient. With only 161 patients, the
   learned morphology is still heterogeneous and split-sensitive.

The candidate checkpoints must not be promoted. Runtime eligibility remains
false until the internal gate passes and an independent external-hospital
waveform lockbox is available.

## Artifacts

- `artifacts/raw_waveform_cache_report.json`
- `artifacts/raw_waveform_event_jepa_15m_audit.json`
- `artifacts/raw_waveform_event_jepa_30m_audit.json`
- `models/raw_waveform_event_jepa_15m_candidate.pt`
- `models/raw_waveform_event_jepa_30m_candidate.pt`

The raw sequence cache remains outside Git because it contains row-level
derived patient data.

## Next evidence needed

Do not tune this cohort until it passes. The next valid test is a larger,
independent waveform cohort with the same timestamped event definition and
the same frozen preprocessing contract. Bedside assessment and treatment
intent may be added only when they are available before the anchor; they must
be evaluated as separate ablations so workflow signals are not mislabeled as
physiology.
