# Interval Personalization Findings (2026-08-12)

## Question

Can the formal adjacent three-level JEPA emit materially narrower 90% factual
forecast intervals by combining:

1. better point forecasts from measurement-pure history and observed treatment
   context;
2. intervals shown only for low predicted-risk rows;
3. online patient calibration using only forecast errors whose target event has
   already occurred?

## Implementation

- Added `osler_jepa/interval_personalization.py`.
- Low-risk selection uses only the detached distributional scale available at
  forecast time. High-risk rows abstain and receive no interval.
- Online adaptation starts from the population interval. A patient's interval
  is updated only after the corresponding `future_event_time_<target>` is no
  later than the next anchor.
- Forecasts resolved by the same measurement event are de-duplicated.
- No interval loss or historical forecast outcome can update the physiology
  latent or point model.
- Added projected Parquet loading so unused provenance and non-target future
  timestamps do not inflate memory.

The formal patient-heldout audit used 1,000 stays and seven seeds. External
hospital, care-unit and forward-time audits used three independently trained
seeds for the only strategy that passed all seven patient splits.

## Patient-heldout result

| Cell | Point forecast vs teacher | Existing interval pass | Selective low-risk | Online patient-adaptive | Selective + online |
|---|---:|---:|---:|---:|---:|
| creatinine@12h | 7/7 | 5/7 | 3/7 | 7/7 | 7/7 |
| glucose@1h | 7/7 | 5/7 | 4/7 | 6/7 | 2/7 |
| heart_rate@3h | 6/7 | 6/7 | 6/7 | 7/7 | 6/7 |

Median half-width change relative to the same patient-cluster distributional
reference:

| Cell | Selective low-risk | Online patient-adaptive | Selective + online |
|---|---:|---:|---:|
| creatinine@12h | -9.65% | approximately 0.00% | -0.29% |
| glucose@1h | +23.95% | +0.35% | -1.15% |
| heart_rate@3h | +8.98% | +0.58% | -0.75% |

## External result for creatinine@12h

`selective + online` was the only policy that both narrowed and passed all
seven patient splits. It failed every external calibration set:

| External split | Pass | Median patient-equalized coverage | Median width change |
|---|---:|---:|---:|
| hospital | 0/3 | 0.968 | -0.08% |
| care-unit | 0/3 | 0.979 | -0.17% |
| forward time | 0/3 | 0.985 | approximately 0.00% |

The failure is over-coverage, not unsafe under-coverage. The selected subgroup
changes across domains and remains too conservatively calibrated.

Plain online adaptation passed hospital 3/3, care-unit 2/3 and time 1/3, but it
did not materially narrow the interval. It is therefore not a width-improvement
candidate.

## Interpretation

### Point accuracy

The formal model already receives causal history and pre-anchor treatment
evidence. Glucose point prediction passed 7/7, while its interval passed only
5/7. Therefore glucose interval instability is not explained only by point
forecast accuracy. Residual distribution shift between patient splits remains
important.

The eICU canonical ledger has broad treatment evidence counts, but dose/rate
fields are largely unavailable. More precise observed treatment context may
still improve point prediction, but current eICU evidence cannot supply it.

### Selective intervals

Selection sometimes makes creatinine intervals narrower, but coverage is not
stable by itself. Glucose and heart-rate selections generally make intervals
wider. The scale head therefore cannot yet be used as a formal abstention gate.

### Online patient calibration

Online updates improve coverage stability: creatinine reaches 7/7 and heart
rate reaches 7/7. The width does not materially shrink, however, because only a
fraction of forecasts have at least three previously resolved outcomes and the
population residual distribution remains dominant.

## Decision

- Creatinine selective-online passed 7/7 patient-heldout but failed hospital,
  care-unit and time calibration gates.
- No policy both produced a material width reduction and passed every required
  domain gate.
- No runtime interval artifact was promoted.
- Runtime must continue to emit `lower=null`, `upper=null` for these cells
  unless an existing exact target-horizon artifact has independently passed all
  required gates.
- Online and selective policies remain bounded research candidates.

The cleanest positive result is methodological: online calibration is now
implemented without future-label leakage and improves coverage stability. It
does not yet provide a sufficiently reliable or material interval-width gain.
