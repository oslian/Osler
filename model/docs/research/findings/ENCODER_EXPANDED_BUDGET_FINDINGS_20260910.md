# Expanded-Cohort Exposure and Optimization

## Source-Only Diagnostic

The script audit_encoder_exposure.py replays the actual NumPy sampling order,
not an independent exposure estimate. Each32epoch run contains512 optimizer
steps and16,384 patient-row draws. No external scores enter this diagnostic.

Median across7seeds:

| Fitting patients | Unique supervised row coverage | Future event coverage | Draws per patient median |
|---|---:|---:|---:|
|250|74.58%|76.79%|65.5|
|1000|38.44%|41.61%|16|
|2250|21.60%|24.07%|7|

Coverage refers to supervised anchors/future event identities. A sampled
anchor also contains historical measurements; this does NOT mean the model
has seen only22% of all input measurements. Future keys distinguish stay,
measurement timestamp and result availability. No unique-source-ID claim.

Whole32epoch late source-tune MAE versus epochs9-16 improves0/7 at250fit,
7/7 at1000 and2250fit. This motivated the prespecified64epoch comparison,
not a proof of convergence or undertraining as the sole cause.

## Matched 32 -> 64 Epoch Trial

All28 fits completed on the identical2250fit/136tune roles. Every first32
epoch training log reproduces exactly. Only epoch budget changed; all
checkpoints selected on source tuning MAE including epoch0.

Median patient-equal glucose@6h MAE in mg/dL:

| Model | Hospital32 | Hospital64 | MIMIC32 | MIMIC64 |
|---|---:|---:|---:|---:|
|Own identity|36.998|36.310|31.214|31.209|
|Whole identity|37.108|36.796|32.223|32.037|

Patient-paired significant improvement over32epochs: own6/7hospital,2/7MIMIC;
whole2/7both. No significant harm in these identity comparisons. Whole vs
own remains0/7both. Whole selected source-tune score remains numerically
worse6/7seeds (one seed better); medians across different seed difficulty
must not be interpreted as a paired whole advantage.

MIMIC TSICU vs persistence now own2/7,whole0/7.2014-2016 subgroup own7/7,
whole3/7. These are descriptive subgroups, not a fresh temporal lockbox.

## What This Resolves

Increasing optimization on the expanded source gives some improvement, but
does not repair incremental whole-context generalization. Thus neither
"just add patients" nor "just train longer" is a demonstrated complete fix.
The earliest residual failure remains generalization of non-target context,
including source-tune patients, not merely external database mismatch.

Selected epochs range48-64own and45-64whole; no convergence ceiling claim.
Sample count/composition, pretraining overlap, and observation support remain
limits. Learned associations are not established organ mechanisms.

No release weights, registry, calibrator or runtime changed. Candidate_only;
no all-target, conformal or causal promotion.303 tests pass.

Evidence: artifacts/encoder_exposure_20260910.json,
encoder_source2250_64_20260910.json,
encoder_source2250_64_comparison_20260910.json,
encoder_source2250_64_strata_20260910.json (all under artifacts).
