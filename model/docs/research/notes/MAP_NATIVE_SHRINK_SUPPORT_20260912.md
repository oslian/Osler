# Matched MAP Shrinkage Support Control

Apply the SAME source-tune alpha grid to full-own and support-own increments.
Original full alphas, tune boards and scored MAEs reproduce. Support retains
own numerical history and other masks/ages, not other numerical values.
Support alphas: .2,.6,.7,.5,.8,0,.6. No test-selected coefficient.

| Method | Source evaluation MAE | Hospital MAE |
|---|---:|---:|
| Own | 8.78353 | 10.24211 |
| Shrunk full | 8.72261 | 10.28760 |
| Shrunk support | 8.71739 | 10.28528 |

Shrunk full versus shrunk support: gain1/7source,0/7hospital, no significant
harm. Shrunk support versus own: gain1/7source,0/7hospital, no harm. The prior
full-versus-own4/7source result remains valid, but cannot be attributed to
stable extra numerical physiology without this matched control. Difference
between significance counts does not itself prove a significant difference.
Nonsignificance is not equivalence. Both hospital medians remain worse than own.

No neural fitting, new lockbox, calibration or promotion. No runtime change.
Evidence: `artifacts/map_native_shrink_support_20260912.json`.
