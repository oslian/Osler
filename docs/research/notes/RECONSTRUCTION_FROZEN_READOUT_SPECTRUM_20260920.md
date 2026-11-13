# Frozen Readout SVD Localization

Use exact current bounded7body features and same-capacity refit weights. For
each of33targets compute64dimSVD on centered source-fit states weighted by
the same bounded-site/equal-group objective. Fixed bottom16directions, no rank
search or evaluation-selected subspace. Remove only the centered NEW readout
correction in those directions; original weights and mean correction retained.
All body states and treatment/mask inputs unchanged. Cached shard identities,
parent hashes and original refit aggregate errors reproduced.

Across targets median bottom16variance fraction=.0065474 (0.655%). Median
fraction of squared correction-weight norm in those directions=.7095425
(70.95%). These are separately summarized medians, not a pooled information
measure. Low variance is not proof of noise; coefficient norm depends on native
latent scaling. This is evidence of amplification of weakly varying directions,
not physiological interpretation of the singular vectors.

| Source selection | Original MAE | Refit MAE | Weak correction removed |
|---|---:|---:|---:|
| Selected90 |1.797984335|1.896873821|1.882582842|
| Broad3323 |1.559770689|1.601832405|1.601027027|

20/33targets numerically improve versus refit in each group. Aggregate
regression recovery is about14.5%selected and1.9%broad, leaving most regression
unexplained. This is ablation, not a fitted reduced-rank model and not a claim
that all20changes are significant. Do not deploy or choose a new cutoff using
these outcomes. Weak-direction removal alone fails to recover original accuracy.

## Decision

Unregularized fixed-state readout optimization partly exploits directions with
little source variation. However, this cannot explain the full cross-site
failure; retained directions/conditional shifts still matter. A broad claim of
latent collapse or an absence of all signal is unsupported. Further regularized
refit, if attempted, must select strength solely within fitting hospitals and
preserve the same-capacity comparison, not sweep ranks on this selection set.

Focused synthetic subspace test passes; audit exit0. Cache digest stored in
aggregate report. No candidate installed, runtime or promotion changes.
Current reconstruction, no future prediction or causal claim. Evidence:
artifacts/native_reconstruction_frozen_readout_spectrum_20260920.json.
