# Fixed Conditional Residual Repair

Question: does a source-only conditional correction transfer better than the
previous single offset, and does actual physiology add beyond observation
identity/age? Frozen current reconstruction, not forecasting or a new body model.

Use all47targets/five existing component-disjoint calibration/outer roles and
the exact variable snapshots. Calibration is the correction fitting role;
outer remains evaluation. Already inspected data are reused development
validation, not untouched confirmation. No newly selected clinical subgroups.

Four predictions: unchanged real; component-equal median residual offset;
ridge residual with frozen real prediction plus observed masks/log1p age;
same ridge with latest measured values added. Missing values mean-imputed
using correction-fit observed values only, with explicit masks. Support for an
input requires>=30observed people in>=3correction-fit components; the same
input support applies to both ridge arms. No outer support/labels select inputs.
Overall correction fitting requires>=30people in>=3components else exact zero.

Ridge uses existing sklearn, weighted StandardScaler and sample weights summing
to1 with equal component mass. Fixed alpha0.1, no search/clipping. Fit residual
divided by the existing encoder-fit target scale, then restore raw units.
This minimizes penalized squared residual, NOT MAE: outlier sensitivity and
mean-versus-median mismatch are explicit risks, not hidden implementation bugs.
Both conditional arms include the same frozen point forecast. The process
correction is not an entirely process-only model because its base is numeric.
Feature dimension differs; this is not an equal-capacity placebo gate.

Source leave-one-component-out: repeat ALL preprocessing, input support and
correction fitting without that component. Report equal-component MAE and
support completeness. No activation or hyperparameter selection from these
results this run. Then fit each fixed arm on full calibration and assess all
outer patients. No outer-derived exclusion or sign-flip deletion.

Primary contrast: numeric conditional versus unchanged, offset, and process
conditional, at patient and component level; paired bootstrap5000 and existing
one-sided tests with47target correction, allsix required. This is only a
descriptive incremental screen conditional on overlapping crossfit models,
NOT runtime promotion. Report absolute MAE and fit-scale-normalized macro.
No interval, clinical/causal, future forecasting or new lockbox claim.
