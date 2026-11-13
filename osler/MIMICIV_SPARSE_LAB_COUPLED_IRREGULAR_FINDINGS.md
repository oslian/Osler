# MIMIC-IV Sparse-Lab Coupled Irregular-Time Findings

## Question

Can future sparse hormones, cardiac markers, hepatic markers, and coagulation
labs become more predictable when the model uses the pre-anchor histories of
physiologically related laboratory streams, rather than only the target's own
event history?

## Design

- Cohort: MIMIC-IV v3.1 ICU laboratory events.
- Targets: INR, PTT, fibrinogen, cortisol, troponin-T, bilirubin, and TSH.
- Query windows: `(0, 6]`, `(6, 24]`, and `(24, 72]` hours, matching the
  irregular-time audit's right-closed bins.
- Candidate: ridge models with target history plus mechanism-limited
  cross-laboratory history features: previous value/age, delta, slope, and
  observation count.
- Comparators: persistence and the target-history-only irregular event model.
- Gate: seven patient-heldout seeds, internal care-unit holdout, and a
  candidate increment over both comparators.
- Leakage boundary: all source histories are strictly before the anchor;
  future laboratory values are never features.

## Result

The coupled candidate promoted **0 of 21 target-window cells**. Every cell
remained `candidate_or_fallback` for both raw and log-scale candidates.

| Target | 0-6h pairs | 6-24h pairs | 24-72h pairs | Promoted |
|---|---:|---:|---:|---:|
| INR | 62,896 | 297,278 | 546,472 | 0/3 |
| PTT | 81,993 | 423,204 | 756,472 | 0/3 |
| Fibrinogen | 20,238 | 65,201 | 91,711 | 0/3 |
| Cortisol | 1,778 | 856 | 763 | 0/3 |
| Troponin-T | 8,509 | 35,086 | 16,974 | 0/3 |
| Bilirubin | 12,887 | 94,156 | 186,618 | 0/3 |
| TSH | 111 | 256 | 247 | 0/3 |

## Interpretation

Adding physiologically related laboratory trends did not produce a robust
increment beyond the existing target-history model. This does **not** prove
that these variables are biologically independent. It means that, under this
observational cohort, measurement timing, missingness, clinical selection,
and heterogeneous care leave no stable signal that passes the promotion gate.

The result also does not validate a hidden hormone concentration, tumor burden,
myocardial injury state, or coagulation reserve. These remain observation
targets or candidate-only forecasts.

## Boundary

This is single-center factual observation research. It supports no causal,
counterfactual, clinical, treatment, or hidden-state accuracy claim. The
report is aggregate-only and contains no row-level patient identifiers.
