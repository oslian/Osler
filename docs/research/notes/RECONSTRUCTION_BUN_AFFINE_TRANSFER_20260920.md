# BUN Affine Calibration And Rank Diagnostic

Same cached source-fit529/9664 and disjoint source-selection90/3086pairs.
Fit unpenalized median QuantileRegressor(solver=highs) on scalar prediction,
separately for each source group and full/body-off. Both fits cross-applied;
no hyperparameter or winning fit selection. Hashes and identity separation
verified. Source-selection has been repeatedly examined, not a fresh lockbox.

Full prediction source-fit slopes1.041923selected/.922187broad, intercepts
-.325161/-.368966. No support here for a universal large amplification.

| Selection | Original full MAE | Selected-fit affine | Broad-fit affine |
|---|---:|---:|---:|
| Selected90 |1.105465|1.123474|1.157106|
| Broad3086 |.893612|.823873|.818155|

Compared with offset-only, selected-fit affine changes selected-.005888
(CIcrosszero), broad+.001240(CIpositive). Broad-fit affine changes
selected+.014083(CIpositive), broad-.002046(CInegative). Thus the extra slope
does not supply a transferable repair. All intervals descriptive/unadjusted.

Full Spearman correlation .440347selected/.384312broad; prediction SD
.649744/.540403 versus truth SD1.584789/1.452745. Body-off rank correlations
.385236/.398165. Narrower marginal prediction dispersion is real but not proof
of defective scaling: a conditional median should omit unpredictable outcome
variation. Inflating predictions to match truth variance is not justified.
All fitted slopes positive, so this affine adjustment cannot repair ranking.

## Decision

The offset component is useful in broad selection, but a simple output-scale
repair does not recover cross-group accuracy. Stop adding scalar calibration
degrees of freedom on these reused outcomes. Remaining questions concern
conditional information/ranking and population support, not a proven universal
compression bug. These correlations do not prove that information is absent
from the internal latent, nor that new heads would recover it.

Two tests pass (exact affine recovery and empty-input rejection); audit exit0.
No release weights, runtime, intervals or promotion changed. Current masked
reconstruction only; no future forecast or causal conclusion.
Evidence: artifacts/native_reconstruction_bun_affine_transfer_20260920.json.
