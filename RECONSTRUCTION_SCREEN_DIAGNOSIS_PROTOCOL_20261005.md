# E41: distinguish weak predictions from inconclusive assessment

## Fixed diagnostic, not a new acceptance policy

E40 is complete and verified. Its selection role has 53 inventory targets,
46 with observations and 27 meeting the declared 30-patient/3-component support.
Those aggregate counts and the creatinine entry were inspected before this
diagnostic. This is retrospective diagnosis, not a preregistered confirmation.

Keep all ten comparisons, predictions, patients, roles, alpha .05/(53*10),
95% bootstrap CI and one-sided paired t-test from the original implementation.
The original p_less is NOT a bootstrap exceedance count: do not invent a
Monte Carlo resolution obstruction or replace the test. No new training,
selection, calibration, release, thresholds or promotion.

For every target/control list independently:
- patient and component support;
- patient and component mean error differences (candidate minus reference);
- negative mean at both levels;
- CI upper < 0 at both levels;
- nominal p < .05 at both levels (description only);
- original family-adjusted p cutoff at both levels;
- exact reproduction of the original screen.

Give disjoint FIRST-UNMET categories in this reporting order: unsupported,
insufficient_support, mean_not_better, interval_inconclusive,
nominal_test_inconclusive, multiplicity_only, passed. These are reporting
categories, not causal attribution. Also preserve ALL failed criteria so
the chosen order cannot hide overlapping problems. At target level require
all ten controls, and also list each control separately. Retain the inventory
when counting; unsupported does not mean a failed trained prediction.

Independently reconstruct normalized patient errors from all five saved
selection parquet files, using scalar math.fsum; collapse repeated patients
across folds, then collapse patient means per component exactly as E40.
Verify identities, finite values, counts, prediction-role metadata bindings,
patient/component means, paired difference means and t p-values against the
saved complete aggregate. CI draws were already independently verified by
E40: reuse their fixed values, do not regenerate different seeds. Preserve
E40's historical backbone exposure and duplicate-fold limitations.

Use existing pandas/standard library and scipy.stats.t.cdf, with numerical
comparison tolerance 1e-10 (verification roundoff, not scientific thresholds).
All inventory and comparison keys must be exact. Test malformed/missing/
nonfinite input, equality at each boundary, unequal patient multiplicities,
overlapping failure reasons and screen mismatch. A synthetic clear improvement
must pass the unchanged gate, proving it is not structurally impossible.

Output aggregate-only JSON and findings. Bind sources, code, protocol, tests
and saved selection rows; keep IDs and row-level predictions private. Stop on
contract mismatch instead of silently dropping targets. No new external claims:
MIMIC/NWICU are reused development. Even a statistical-only blockage does not
authorize relaxing the gate, and a positive sample mean is not proof of harm
or permanent unpredictability. This repairs failure reporting, not the model.
