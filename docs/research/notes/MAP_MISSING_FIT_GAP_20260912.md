# Native-Missing MAP Fitting Gap

Replayed completed native-missing own/full/support predictions for all patient
roles, exactly reproducing existing evaluation/hospital MAEs and comparisons.

| Role | Full-own paired MAE difference | Full-support difference |
|---|---:|---:|
| Fit | -0.28974 | -0.05764 |
| Tune | -0.00780 | -0.00729 |
| New source | +0.02005 | +0.02938 |
| Held-out hospital | +0.07873 | -0.04568 |

Values are medians of seven within-split paired differences, not differences
between median MAEs. Fit full-own gain7/7, full-support4/7. Both comparisons
have0/7 gains in all other roles. Fit CIs are resubstitution diagnostics, not
independent efficacy. Hospital numerical full-support ordering varies by seed;
nonsignificance does not establish equality.

Native missing routing retains fitting numerical advantage but does not make
it generalize. It therefore does not support "all numerical capacity vanished"
as the explanation here. Failure is visible before changing hospitals. The
remaining cause is not uniquely identified by this learning gap: representation,
finite support, fitting bias, conditional noise and cohort selection remain
possible. Repeating missing handling or the gap-convention repair is not a new
untested direction. No trained change, runtime update, calibration or promotion.

Evidence: `artifacts/map_missing_fit_gap_20260912.json`.
