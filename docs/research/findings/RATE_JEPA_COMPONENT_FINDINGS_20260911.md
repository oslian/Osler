# Frozen rate-query JEPA component diagnosis

Completed seven frozen-weight comparisons. Full predictions reproduce cached
predictions to absolute tolerance 1e-7. No training, selection, runtime change,
calibration, or promotion. Machine-readable evidence:
`artifacts/rate_jepa_components_20260911.json`.

Same 5,526 potassium pairs and inherited patient roles as the rate-query JEPA
experiment. Non-target values were replaced by checkpoint-normalized zero at
all cached history rows, preserving observation support and ages. A separate
intervention set observed non-target ages to three hours, preserving values and
support. Potassium inputs remained unchanged.

| Patient-equal MAE, median across seeds | Source evaluation | Held-out hospitals |
|---|---:|---:|
| Own-target JEPA | 0.417130 | 0.402915 |
| Whole-context JEPA | 0.384183 | 0.386973 |
| Non-target numerical values neutralized | 0.407211 | 0.407274 |
| Non-target observed ages standardized | 0.384183 | 0.386957 |

Numerical neutralization significantly harms the frozen whole model in 5/7
source comparisons and 3/7 hospital comparisons; improves it in 0/7. Age
standardization has no significant hospital effect and one small significant
source harm. It is not literally invariant. These overlapping seeds are not
independent replications.

## Failure location

The latest model uses non-target numbers beneficially under this sensitivity
test. Therefore neither a disconnected numerical input nor exact measurement
age alone is supported as the principal explanation of its remaining gap.
This does not prove pure physiology: neutralization is off-manifold and does
not replace a train-matched support-only control. Masks and value/support
interactions remain intact.

The earlier own-versus-whole trained comparison remains authoritative for
incremental utility: 5/7 source all-window gains but 1/7 hospital gains. Hospital
mean gain is negative in all seven fits, yet removing one hospital reverses
the mean in two fits; only one of 15 hospitals has at least 30 patients.
Thus partial shared numerical prediction is established in the tested source
contrast, but broad transportability is not. Small support and hospital
heterogeneity are observed limitations, not a proved unique biological cause.

Do not generalize this potassium rate-query result to every release cell,
or to continual learning: these weights have no treatment-state input or
online error update. Rate-record times define queries but are not independently
verified bedside availability times. No new external lockbox was used.
