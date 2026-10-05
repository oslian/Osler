# Potassium Matched Encoder Results

All14 prescribed fits completed (7 seeds x own/whole,64epochs). Same initial
JEPA, optimization, patient splits, measured potassium pairs and potassium
checkpoint selection. Source-only, no external reads or runtime change.
Fit support1231-1246patients, tune63-78patients per split. These remain
glucose-query anchors, not a general potassium cohort. Pretraining overlap
is not excluded, and tune also selects checkpoints.

| Seed | Own MAE | Whole MAE | Whole minus own | Paired CI95 |
|---|---:|---:|---:|---|
| 7 | .37611 | .35552 | -.02059 | [-.04255,.00189] |
| 19 | .36609 | .34862 | -.01747 | [-.04296,.00806] |
| 31 | .33563 | .32485 | -.01078 | [-.03306,.01494] |
| 43 | .34904 | .32988 | -.01916 | [-.04663,.01054] |
| 59 | .34730 | .33639 | -.01090 | [-.03537,.01601] |
| 71 | .40296 | .37975 | -.02322 | [-.05484,.00719] |
| 83 | .41403 | .40312 | -.01091 | [-.03565,.01654] |

Whole is numerically better7/7, significantly better0/7. Both arms beat
persistence significantly6/7, with seed71 failing. Seeds reuse overlapping
patients: seven same-sign means are not seven independent replications.

## What Changed In The Diagnosis

This corrects two limitations of the prior diagnostic: a linear own-history
control had different capacity, and auxiliary checkpoints were glucose-selected.
Target-matched neural training shows small consistently signed incremental
whole-input benefit, but no complete statistical or external validation. It
does not prove sample size is the cause; uncertainty, heterogeneity and selection
bias remain. Do not deploy or globally amplify the residual.

Unlike the earlier glucose fit-to-tune reversal, potassium's tune comparison
is not a consistent directional loss. A blanket claim that full-context
training always harms is therefore unwarranted. The unresolved bottleneck is
target-dependent, not a demonstrated universal dead pathway. No causal organ
attribution, calibration, persistent-organ trajectory or runtime promotion has
been established. Source-independent evaluation is required before interpreting
this small tune advantage as a repair.

Evidence: `artifacts/potassium_matched_encoder_20260911.json`, complete=true.
Protocol: `POTASSIUM_MATCHED_ENCODER_PROTOCOL_20260911.md`.
Private checkpoints and prediction arrays remain outside Git.
