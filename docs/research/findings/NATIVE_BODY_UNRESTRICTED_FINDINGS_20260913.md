# Baseline-Independent Masked Raw History

Completed the prespecified same-seed/current-only query experiment without
nested-baseline support filtering or numeric recentering. Original cached
measurement histories feed the existing summary probe directly. Target and
siblings remain fully hidden; original clocks are checked before masking.

Source patient-heldout raw MAE12.26777 versus population12.00322; paired
delta+.26455, CI[-.46692,1.09957]. Hospital raw14.27455 versus14.30928;
delta-.03473, CI[-1.13409,1.09990]. Neither raw/population nor raw/support or
raw/placebo difference is significant. Support-only scores exactly reproduce
the previous probe. Raw selects ridge alpha1000 using source tune only.

The preceding support/affine transform was a dependency worth testing, not
an established bug. Removing it did not establish MAP reconstruction benefit.
Numerical differences are not proof that useful variables were previously
removed: coordinate centering, zero-filled missing summaries and regularized
fitting can interact. This experiment does not isolate these mechanisms.
It is not proof that all raw physiological information is absent; summaries,
one-anchor sampling and single-target scope remain limitations.

Report: `artifacts/native_body_unrestricted_raw_20260913.json`. Original
cache files and current reconstruction module are hash-bound. Prior reports
are preserved. No neural retraining, saved model, calibrator, runtime or
promotion change. The whole-body objective remains unfinished.
