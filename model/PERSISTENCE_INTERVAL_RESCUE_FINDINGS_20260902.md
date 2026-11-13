# Persistence interval rescue

All 14 fallback cells were inspected. Only the four whose frozen development-selected
source already passed both databases' point gates are eligible for this interval-only test.
The other ten are not relabelled as successful or tested against external runner-ups.

One fixed log-residual ridge scale model (penalty 10) uses only current target value and
absolute predicted change. One anchor per selection patient fits the scale; separate
calibration patients determine the exact nominal 90% conformal quantile. Point weights,
source choices, nominal level and coverage bounds [0.87,0.93] stay fixed. External residuals
are never used for fitting. These are reused research validation sets, not a fresh lockbox.

Attempted: 4; passed: 1.

| Cell | MIMIC row coverage | NWICU row coverage | Pass | Failures |
|---|---:|---:|---|---|
| creatinine@6h | 0.8764070238631247 | 0.871662566748665 | False | mimic_test:late_coverage |
| creatinine@12h | 0.8743945026472908 | 0.921903123875045 | False | mimic_test:late_coverage, nwicu:careunit_coverage, nwicu:late_coverage |
| bicarbonate@24h | 0.8925520262869661 | 0.8935089324408918 | True | - |
| bun@24h | 0.8767349636483807 | 0.8666160060792705 | False | mimic_test:late_coverage, nwicu:coverage, nwicu:careunit_coverage |

## Other fallbacks

- `platelets@1h`: insufficient_unused_development_patients
- `heart_rate@3h`: nwicu:point
- `hematocrit@3h`: nwicu:point
- `hemoglobin@3h`: nwicu:point, nwicu:coverage, nwicu:careunit_coverage, nwicu:late_coverage
- `platelets@3h`: mimic_test:point, nwicu:point, nwicu:coverage, nwicu:careunit_coverage, nwicu:late_coverage
- `respiratory_rate@3h`: mimic_test:point
- `hematocrit@6h`: nwicu:point
- `hemoglobin@6h`: nwicu:point, nwicu:coverage, nwicu:careunit_coverage
- `bicarbonate@12h`: nwicu:point
- `glucose@48h`: mimic_test:point, nwicu:coverage, nwicu:careunit_coverage, nwicu:late_coverage

Passing this test concerns factual point/interval behavior, not latent identifiability
or clinical safety. Coverage is empirical and not a guarantee for each patient.
