# E30: E29 role identifiability audit

## Question and prior evidence

E29 verified a fit gain followed by selection reversal for 22/46 targets
relative to unbridged. E13 and the older conditional-offset audit already
examined source error and composition for different candidates. This audit
does not repeat those models or claim a new general domain-shift finding.

Hypothesis: E29's two saved roles cannot isolate new-patient error from
new-source error, because their source components are disjoint within each
frozen model fold. A preliminary read of all five folds found zero overlap;
this protocol records that known observation, not a prospective preregistration.

## Fixed diagnostic

- Require the complete E29 verifier, its bound assessment, all five bound fold
  reports and both prediction files per fold. Do not refit or modify them.
- Validate nonempty patient/component/target identity and unique patient-fold
  rows using the existing contract. Reject patient overlap across fit and
  selection within a fold, and inconsistent patient/component mappings.
- Count source overlap separately for each fold, and each target within it.
  All 53 inventory targets remain represented, including missing role support.
- Report same-source unused-patient comparisons as unavailable when overlap
  is zero. Do not call zero overlap a numerical or training error.
- Also count components shared after pooling folds, but never use that pooled
  overlap to assert within-model overlap: different folds mean different
  learned weights and data roles.
- Report existing paired MAE means only through the original E29 assessment;
  this audit has no new performance gate, p-value or promotion decision.

## Interpretations and stopping rule

If same-source support exists, a later fixed diagnostic can inspect that
support, retaining exclusions and counts. If it does not, stop composition
decomposition rather than inventing missing fit/source counterfactuals.

The next valid comparison would need three roles under the SAME fitted model:
fitting patients, adaptation-unseen patients from fitting sources, and patients
from unseen sources. Audit exposure to the original encoder separately:
unused in E29 adaptation does not mean never seen by the frozen predecessor.
No third-role patient inference or new training is authorized by this protocol.

No-overlap results cannot prove source shift is the cause. In-sample optimism,
different patient mixtures, observation support and representation remain
confounded. Do not infer future forecasting failure from current reconstruction.
Existing runtime, weights, intervals and gates remain unchanged. Store only
aggregate counts and hashes; no patient identifiers in the public artifact.
