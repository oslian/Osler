# Known-source glucose offset trial

Frozen native glucose JEPA predictions, original roles and measured labels.
Source tune fits patient-weighted median regression of residual on current
source indicator with intercept. Same treatment for own/whole; controls use
a global intercept or independently row-shuffled source indicator within each
role. All42 small regressions completed. Current source reconstructed only
from results available by anchor; future source never used as input.
Original scores and hashes reproduced.

| Median MAE | Whole original | Whole source offset | Whole constant | Whole shuffled | Own source offset |
|---|---:|---:|---:|---:|---:|
| New source | 34.84081 | 34.76522 | 34.63634 | 34.77482 | 33.91928 |
| Hospital | 42.40070 | 42.54759 | 42.50110 | 42.50205 | 41.34126 |

Whole source-offset vs original gain1/7source; hospital gain1/7,harm1/7.
Versus constant0/7both (source harm1/7), shuffled0/7both. Versus equally
corrected own0/7both (harm2/7source,1/7hospital). Thus source-conditioned
constant bias does not repair the extra-context failure. This does not reject
source-aware history encoding, matching actual assay targets, or interactions.

The shuffled comparator has equal parametric capacity but shuffles rows, not
entire patient trajectories; it is not a definitive patient-history placebo.
Negative comparison against the simpler constant and matched own controls
already prevents a promotion claim. No selection between methods using these
results. Roles are reused validation, seeds overlap. Tune already selected
checkpoints; this is diagnostic post-calibration, not untouched validation.

Aggregate: `artifacts/glucose_source_offset_20260911.json`.
Command: `python -m scripts.audit_glucose_source_offset` (fresh output only).
Raw metadata stays private. No encoder weights, intervals, serving or registry
changes; no clinical/causal claim. Preserving source provenance remains useful
for audit, but is not itself a demonstrated predictive repair.
