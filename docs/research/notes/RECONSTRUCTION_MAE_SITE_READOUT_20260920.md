# Exact MAE-Aligned Site Readout Results

Executed the fixed protocol in RECONSTRUCTION_MAE_SITE_PROTOCOL_20260920.md.
Same bounded7 encoder, cached source data and33 linear heads. Sparse scipy
HiGHS solves weighted MAE plus fixed0.1 standardized coefficient-change L1
penalty, with/without per-source-group/site MAE non-regression constraints.
Both arms use the same objective; this differs from previous SmoothL1/L2.
All coefficients saved before held-source scoring. No hyperparameter sweep.

All66 fits succeed. Constrained source-site maximum MAE increase is below
1e-7 (largest about8.7e-12), fixing the previous metric mismatch exactly up
to numeric tolerance.17/33constrained heads change by more than1e-8 in L1
coefficient norm;16are effectively original. Retention is not new accuracy.

| Held-source macro MAE | Original | MAE/L1 unconstrained | MAE/L1 constrained |
|---|---:|---:|---:|
| Selected |1.797984335|1.793822994|1.791458019|
| Broad |1.559770689|1.553107909|1.552941555|

Constrained improves original macro about0.36%/0.44%. Added gain over matched
unconstrained is about0.13%/0.011%. No target passes existing patient/site
incremental gate against both references in either group. No promotion.
Metrics are normalized, target-equal means of patient errors, not clinical
units or percentages of patients benefiting.

PTT: selected original2.090042/unconstrained2.180820/constrained2.092948;
broad1.375589/1.404156/1.376521. Constraints greatly reduce regression but
do not make patient-average performance better than original. Equal-site
means improve slightly, illustrating weighting differences rather than proof
the model is uniformly better. BUN broad improves original .893612->.837296
but loses matched unconstrained .829234; selected worsens1.105465->1.114133.

## Decision

MAE/SmoothL1 guarantee mismatch is repaired in this candidate. Persistent
held-source failures therefore cannot be assigned solely to that mismatch.
The fixed representation plus global linear correction still trades off
cohorts, even under exact observed-source hospital constraints. In-sample
retention does not imply unseen-hospital retention. This does not prove that
all more expressive models fail or that information is absent.

Stop treating another offset/penalty/site-constraint variant as an untested
complete cure. The separate encoder-independence/representation transfer
question remains; old independent-head confirmation already tested a related
but different boundary. Reused source/MIMIC/NWICU cannot be relabelled untouched.

Two tests pass: unpenalized exact affine recovery and protection of a perfectly
predicted source site. Full LP audit exits0 after all33matched pairs. No runtime
weights, registry, intervals or causal permissions change. This remains masked
current-value reconstruction, not future prediction or clinical validation.

Evidence: artifacts/native_reconstruction_mae_site_readout_20260920.json.
