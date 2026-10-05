# Corrected Creatinine Timing and Attribution Findings

## Decision

Both predeclared runs completed. No source qualifies for promotion. The original
claims of information absence and anchor-mismatch elimination are withdrawn.
There is partial patient-associated readout evidence, but not reliable predictive
improvement over persistence. This is not a general result for all renal targets.

The correction protocol is `ANCHOR_ALIGNED_SIGNAL_CORRECTION_PROTOCOL_20260908.md`.
Original aggregate results remain unchanged; their interpretation is corrected in
`ANCHOR_ALIGNED_SIGNAL_FINDINGS_20260908.md`.

## What was corrected

- Original raw placebo swapped only lagged history, not current cross-state.
  Current-only, temporal-only and full cross-block controls are now separate.
- Original nominal 6h labels accepted actual future events 2-10h later. Primary
  labels now require 5-7h; original broad timing is a prespecified sensitivity run.
  Recorded delta must match the actual timestamp. Future time is not a feature.
- The 311-feature release bundle contains 256 learned latent features, a base
  point, 53 current measurements and repeated target current value. A new pure
  latent readout uses only the four named learned blocks, plus the same seven
  own-history features used by every arm. This is not end-to-end JEPA retraining.
- Every learned head has the same HGB family, update budget and patient weights.
  Each fitted bundle is saved and hashed before MIMIC outcomes are evaluated.
- Hospital/patient role definitions stay fixed across windows. Narrowing changes
  available patients and anchors, so cross-window contrasts are not one-factor
  causal estimates. This is still general ICU, not matched AKI-onset sampling.

## Support

| Actual future window | eICU fit patients / rows | eICU patient test | eICU hospital test | MIMIC test |
|---|---:|---:|---:|---:|
| 5-7h primary | 622 / 1022 | 275 | 237 | 2000 |
| 2-10h sensitivity | 1230 / 2651 | 510 | 426 | 2000 |

Evaluation uses one deterministic anchor per patient. Pretrained-encoder patient
exclusion was not independently verified; heldout refers to newly fitted heads.

## Primary point error

Creatinine MAE, mg/dL. Every learned arm includes own history.

| Predictor | eICU patient | eICU hospital | MIMIC |
|---|---:|---:|---:|
| Persistence | 0.26665 | 0.23452 | **0.17755** |
| Own history | 0.26501 | 0.23127 | 0.19280 |
| Add current cross-state | 0.25673 | 0.22732 | 0.18388 |
| Add current + temporal cross-state | 0.26445 | 0.22700 | 0.18187 |
| Add pure latent | 0.25991 | 0.22781 | 0.18373 |
| Add full release bundle | 0.25854 | 0.22851 | 0.18214 |

## What the controls say

Candidate-minus-reference MAE; negative favors candidate. Intervals are
exploratory, unadjusted paired-patient 95% bootstrap CIs.

| Primary comparison | Mean difference | 95% CI |
|---|---:|---|
| Pure latent vs swapped latent, eICU patient | -0.00812 | [-0.01642, -0.00007] |
| Pure latent vs swapped latent, eICU hospital | -0.00834 | [-0.01502, -0.00176] |
| Pure latent vs swapped latent, MIMIC | -0.00425 | [-0.00691, -0.00153] |
| Pure latent vs persistence, MIMIC | +0.00618 | [+0.00285, +0.00966] |
| Current cross-state vs current placebo, MIMIC | -0.00251 | [-0.00492, -0.00027] |
| Full cross-history vs full-block placebo, MIMIC | -0.00451 | [-0.00714, -0.00178] |
| Full cross-history vs temporal placebo, MIMIC | -0.00106 | [-0.00266, +0.00053] |
| Full cross-history vs current-only, eICU patient | +0.00773 | [+0.00215, +0.01316] |

The pure-latent placebo comparison also has favorable means in all seven patient
resamples for all three datasets/roles. These are seven resamples of one fitted
model, not seven independent training runs. Pure latent does not significantly
beat own history in either source holdout. None of the four main candidates
passes the complete diagnostic; eight supported MIMIC care units show mixed
effects, and none provides a significant persistence win for these candidates.

Swapping a latent also removes encoded OWN-target information. Thus this result
does not specifically prove cross-organ coupling; it needs an own-target-only
encoder ablation. Raw current/temporal comparisons are the more specific cross-
variable tests, and they are not consistently positive across roles.

## Sensitivity and remaining failure

In the broad-window MIMIC evaluation, MAE is persistence 0.16035, own 0.17667,
current cross-state 0.16747, full cross-history 0.16715, pure latent 0.16785 and
release bundle 0.16733. All MIMIC added-block placebo CIs cross zero. Original
arms reproduce their previous MAEs exactly; this confirms the correction did
not silently change the baseline feature pipeline.

The first demonstrated obstacle is source-side incremental performance: even
with partial latent placebo evidence, narrow eICU own-history comparisons remain
uncertain, and extra temporal features worsen patient-test error. At transfer,
all main candidates have significantly worse pooled MIMIC MAE than persistence.
This is not just an overly strict routing switch: these are direct head outputs.

Timing/freshness remains a possible contributor, not a proven root cause. Median
anchor creatinine age is 3.83h (source patient), 3.58h (source hospital), 3.58h
(MIMIC) even in the narrow run. A 6h future label can therefore describe change
from a much earlier last observation. Narrowing the future window also halves
source training-patient support; neither the effect of freshness nor the effect
of cohort support has been isolated. Label availability itself is conditional
on measurement practice. Readout regularization, representation and domain
differences remain competing explanations, not diagnosed certainties.

A defensible next diagnostic would freeze patient support and use actual current
measurement anchors to isolate freshness, while testing own-target-only versus
whole-body latent with the same head. Do not tune on these MIMIC results and call
the same data a new external lockbox. No new calibrator or intervals were fitted.

## Reproduction and release boundary

Run from the repository root with the existing medical_jepa virtual environment:

```bash
python -m scripts.audit_anchor_aligned_signal --window narrow --output artifacts/anchor_aligned_signal_corrected_narrow_20260908.json --fitted-output /private/tmp/anchor_aligned_signal_corrected_narrow_20260908.joblib
python -m scripts.audit_anchor_aligned_signal --window broad --output artifacts/anchor_aligned_signal_corrected_broad_20260908.json --fitted-output /private/tmp/anchor_aligned_signal_corrected_broad_20260908.joblib
```

Use new output paths for any rerun; existing artifacts cannot be overwritten.
Default local inputs are the two immutable `*_anchor_aligned_examples_6h_5000`
parquets. Reports bind protocol, script, checkpoint, feature contract, input and
fitted bundle hashes. Fitted diagnostic heads remain local in `/private/tmp`;
they are not release models. No patient rows or identifiers are committed.

Verification: 243 unittest tests passed; `git diff --check` clean; release
validator remains 41 catalog cells, 27 example-patient move, 14 fallback,
0 missing, 2 derived. The frozen global selector's 28/13 is a different scope.
No runtime weights, registry, calibrator or serving behavior changed.
`promotion_allowed: false`, `causal_claim_allowed: false`.
