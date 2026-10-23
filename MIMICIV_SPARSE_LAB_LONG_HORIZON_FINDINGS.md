# MIMIC-IV Sparse Laboratory Long-Horizon Findings

Date: 2026-07-12  
Scope: MIMIC-IV v3.1 ICU labevents, factual observation only

## Question

Can sparse hormones, tumor/cardiac markers, coagulation tests, and hepatic
markers be forecast at 6, 12, 24, or 48 hours from measurements available at
the anchor?

## Method

`mimiciv_sparse_lab_long_horizon_audit.py` scanned the raw MIMIC-IV
`hosp/labevents.csv.gz`, joined events to ICU stays, and formed anchor-to-first-
future-observation pairs. The candidate was a ridge residual model using only
anchor-time values and measurement ages. It was compared with persistence using
subject-level bootstrap deltas.

Promotion required a significant improvement over persistence on all seven
patient-heldout seeds and on an internal care-unit heldout split. MIMIC-IV is a
single-center database, so the care-unit split is not an external hospital
validation.

## Data boundary

- 7,026,104 selected lab rows
- 64,001 subjects
- 91,643 ICU stays
- 18 sparse target families
- Horizons: 6 / 12 / 24 / 48 hours
- Validated cells: **0**

## What the data actually supports

| Target | Raw observations | Longitudinal pairs | Interpretation |
|---|---:|---:|---|
| cortisol | 7,184 | 82 | Too sparse and unstable for promotion |
| PTH | 968 | 12 | Insufficient longitudinal support |
| TPO antibody | 57 | 0 | No future pair; cannot audit |
| TSH | 12,413 | 58 | Too sparse for promotion |
| total/free T4 | 1,859 / 2,762 | 18 / 18 | Too sparse for promotion |
| troponin-I | 0 | 0 | Not present in this MIMIC extract under mapped itemids |
| troponin-T | 49,646 | 335 | Has pairs, but no robust promotion |
| BNP | 4,591 | 7 | Insufficient pairs |
| CK-MB / CPK | 54,858 / 57,170 | 319 / 807 | Variable signal, no full gate pass |
| INR / PTT | 356,421 / 391,648 | 19,287 / 22,227 | Enough observations, but no robust gate pass |
| fibrinogen | 64,672 | 2,293 | No robust gate pass |
| CRP / ferritin | 6,619 / 10,429 | 209 / 334 | Too sparse/unstable |
| bilirubin / direct bilirubin | 159,616 / 12,172 | 9,263 / 247 | No robust gate pass |

The repeated pair count across horizons is expected for many targets: when the
next lab is already within six hours, the same next observation is valid for
the longer horizon. This does not mean the model has a 48-hour observation for
every row; it means the audit uses the first future observation within each
window, as specified.

## Strict endpoint re-audit

The first implementation used the next observation **within** each horizon.
That is useful for support accounting, but it is not a true 24-hour or
48-hour forecast when the same next lab is reused across windows. A second
audit therefore required the future observation to fall in an endpoint window:

- 6h: 4.5--7.5 hours after anchor
- 24h: 18--30 hours after anchor
- 48h: 36--60 hours after anchor

The event-history ridge added the previous value, previous slope, observation
interval, observation count, and a log-space delta model for heavy-tailed
markers. Under the strict endpoint definition, **0 cells were validated**.
INR had 25 six-hour endpoint pairs and PTT had 26; no target had enough stable
24/48-hour endpoint pairs to pass the full gate. The earlier apparent INR/PTT
passes are therefore not promoted and must not be described as long-horizon
forecast validation.

## Interpretation

This is not evidence that these physiological variables are impossible to
model. It is evidence that sparse measurement scheduling and low repeated-pair
support dominate this cohort. The correct runtime behavior is therefore:

1. Show a timestamped observed value when available.
2. Use nowcast only where its own gate passed.
3. Use persistence/fallback for unsupported future cells.
4. Never manufacture a hormone concentration, tumor-marker trajectory, or
   coagulation response from a hidden-state label.

No causal, treatment-effect, clinical, or external-hospital claim is allowed.
