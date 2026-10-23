# Adjacent Three-Level Distributional Interval Findings (2026-08-11)

## Question

Can the whole-body factual model replace broad target-wide uncertainty bands
with narrower patient-specific intervals without losing nominal 90% coverage?

## Implementation

- Added one target-specific positive scale head per target to
  `AdjacentThreeLevelResidualAdapter`.
- Scale input includes the final system token, current target value, and actual
  horizon after system-organ-body message passing.
- Scale input is detached. Uncertainty training cannot alter the physiology
  latent or point prediction.
- Compared patient-cluster distributional conformal intervals with the old
  global patient-cluster conformal interval on identical predictions.
- Acceptance coverage remains 0.87-0.93. No test labels are used to calibrate
  intervals.

## Patient-heldout result

| Cell | Distributional pass | Median half-width | Global half-width | Median reduction |
|---|---:|---:|---:|---:|
| creatinine@12h | 7/7 | 0.264 mg/dL | 0.610 mg/dL | 56.8% |
| glucose@1h, adaptive | 5/7 | 44.97 mg/dL | 68.00 mg/dL | 33.9% |
| glucose@1h, fixed | 6/7 | 61.21 mg/dL | 71.94 mg/dL | 14.9% |
| heart_rate@3h, fixed | 7/7 | 14.58 bpm | 14.84 bpm | 1.8% |

The renal cell contains strong conditional uncertainty information. Glucose
can be narrowed in median width, but coverage is not stable across all seven
patient splits. Heart-rate uncertainty is mostly irreducible under the current
inputs; the validated reduction is only 1.8%.

## External result for creatinine@12h

| Split | Point beats teacher / persistence | Conformal pass | Median reduction | Coverage range |
|---|---:|---:|---:|---:|
| hospital | 1/3 / 1/3 | 2/3 | 38.8% | 0.876-0.947 |
| care unit | 3/3 / 3/3 | 2/3 | 29.1% | 0.855-0.910 |
| forward time | 0/3 / 0/3 | 1/3 | 28.5% | 0.930-0.959 |

The narrower renal interval does not pass the full external promotion gate.
The failure is domain and time instability, not inability to learn conditional
scale. It remains candidate-only.

## Rejected follow-up (2026-08-12)

An observation-quality-aware scale head was tested with direct measurement
age, missingness, expert availability, predicted move magnitude, and
equal-probability patient sampling. It was rejected after an otherwise
identical seven-seed audit:

- creatinine@12h fell from 7/7 to 6/7 conformal passes;
- its median half-width increased from 0.264 to 0.289 mg/dL (+9.7%);
- glucose@1h fell from 5/7 to 3/7 and widened by 4.6%;
- no cell achieved both a new 7/7 result and a meaningful width reduction.

The additional quality variables were already represented inside the final
system token. Reintroducing them explicitly and reweighting patients added
variance rather than new information. The failed implementation was removed
from the formal model; the audit artifact remains as negative evidence.

Conformalized Quantile Regression was also tested for glucose@1h using
patient-disjoint quantile fitting and conformal correction. Its median total
width was 139.3 mg/dL versus 144.6 mg/dL for the global conformal baseline
(3.7% narrower), but it passed only 5/7 patient splits while the global method
passed 7/7; coverage ranged from 0.896 to 0.951. The implementation was removed
from the formal audit because the small width reduction did not compensate for
unstable coverage.

Current-state Mondrian conformal was then evaluated for glucose@1h. It passed
3/7 splits and widened the median half-width from 72.28 to 81.75 mg/dL
(+13.1%). Splitting calibration patients into glucose strata made tail
estimation more conservative without producing stable conditional coverage.

## Runtime decision

- Removed all manually hard-coded demo interval widths.
- Runtime now returns `lower: null` and `upper: null` unless an exact serialized
  patient-specific conformal artifact has passed patient, hospital, care-unit,
  and time gates.
- The point forecast remains available. The UI explicitly reports that interval
  calibration is pending instead of displaying a broad invented band.

## Boundary

No interval is promoted by this audit. Narrower intervals are technically
possible, especially for renal physiology, but current external stability is
insufficient for formal runtime use. The next uncertainty improvement must
target domain/time-robust scale calibration rather than lowering nominal
coverage or manually shrinking widths.
