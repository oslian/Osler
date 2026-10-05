# Continued Reconstruction Site Diagnosis

Frozen continuation predictions, unchanged patients and labels, no fitting or
selection. All stored patient/site gate statistics reproduced within 1e-10;
parent dependency hashes and unique patient-target pairs verified. Three tests
cover weighting reversal, uniform improvement and invalid/single-site inputs.

Counts below overlap, denominator 33 trained targets per cohort. Six untrained
targets stay explicit. Evidence failure uses the existing .05/39 threshold.

| Cohort / reference | Pooled mean not better | Equal-site mean not better | Patient evidence fails | Site evidence fails | Weighting flips gain sign |
|---|---:|---:|---:|---:|---:|
| Broad / old | 1 | 1 | 2 | 22 | 0 |
| Broad / previous expanded | 10 | 12 | 24 | 33 | 8 |
| Broad / matched placebo | 1 | 3 | 10 | 22 | 2 |
| Selected / old | 8 | 11 | 31 | 33 | 11 |
| Selected / previous expanded | 8 | 13 | 30 | 33 | 7 |
| Selected / matched placebo | 2 | 8 | 20 | 30 | 8 |

## What This Locates

The latest candidate is not failing the same way against every reference.
Against the old model on broad patients, favorable means are common (32/33),
but site evidence is often insufficient. Against the previous expanded model,
there are actual numerical mean regressions as well as weak evidence. Zero
full incremental passes cannot be attributed solely to a stringent gate.

Patient-weighted versus equal-site signs differ on 8/33 broad incremental
comparisons and 11/33 selected comparisons against old. Training samples
patients within targets/groups, whereas validation additionally requires an
equal-site benefit. This is a concrete objective-weighting difference, not
proof that site-balanced training will repair generalization. Sampling noise,
site case mix and measurement support remain possible contributors. Site
identifiers must not become physiology inputs.

The artifact includes minimum per-site support, effective site count,
improving/worsening site counts and leave-one-site-out equal-site ranges for
every comparison. Removing a site is a sensitivity diagnostic, not permission
to exclude difficult sites or a new independent validation. No per-site model
or target selection was performed.

## Next Discriminating Check

Before a site-balanced training intervention, check the SOURCE fit/selection
site distribution and whether the same target loss discrepancy exists there.
Only source data may determine sampling weights or selection rules. Preserve
patient support, shared-state architecture, the prior model as a control and
all evaluation sites. Do not treat reused evaluation signs as weights.

Evidence: artifacts/native_reconstruction_continued_site_transfer_20260920.json.
No runtime weights, registry, intervals, promotion or future-forecast claims.
