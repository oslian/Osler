# Numeric Content Versus Exact Age

Seven fixed32-epoch whole identity checkpoints,12 groups x two independent
interventions across two domains. Full outputs reproduce. Glucose and its
history unchanged. Values neutralized to checkpoint normalized center0 keep
mask/age/presence; age neutralization changes only observed ages to3h.
Both are off-manifold diagnostic interventions, not runtime observations.
`artifacts/encoder_components_20260910.json` is complete.

## Main Result

Median patient-equal glucose6h MAE:

| Input | Hospital | MIMIC |
|---|---:|---:|
| Full | 38.202 | 33.368 |
| Non-glucose numeric content neutralized | 38.485 | 32.437 |
| Non-glucose exact age neutralized | 38.201 | 33.366 |

Numeric neutralization improves MIMIC7/7 paired CIs, median paired delta-0.858
mg/dL; hospital0/7 improvement or harm. Exact-age effect is practically tiny
(MIMIC median paired delta-0.003), despite some significant individual groups.
Median differences of medians need not equal median paired differences.

## Group-Level Clarification

Neutralizing acid-base numbers harms hospital3/7, helps MIMIC1/7; prior whole
block removal harmed4/7 hospital and helped3/7 MIMIC. Neutralizing heme numbers
helps MIMIC4/7 (median delta-0.255), whereas prior heme block removal harmed5/7.
Immune numeric neutralization helps MIMIC4/7; musculoskeletal5/7. No individual
numeric group improves7/7 in both roles. All groups, including no-ops, reported.

The earlier block-removal result must NOT be read as proof blood NUMBERS
improve transfer. Values, masks, missing-token embeddings, age and presence
interact. Keeping support while removing numeric variation yields a different
result. This localizes a numeric-content/representation dependency, not a causal
organ effect or proof the model only learns hospital workflow. Exact age is
not the main sensitivity in this test; missingness and history still encode
timing and have not been independently neutralized.

## Decision

Do not replace measured values by artificial centers in serving. This screen
does not establish no information in non-target physiology or justify dropping
all context. The separately trained own identity still achieves lower MIMIC
MAE31.238. Source-scale/support and learned numeric response remain candidate
causes; fixed constant intervention cannot uniquely separate them. A repair
must be trained and selected on source data with actual masks retained, then
tested without changing observed-value semantics at inference.

No new weights, calibration, runtime or promotion. Reused validation patients,
unadjusted multiple comparisons, no new lockbox.297 tests pass; one regression
verifies each intervention changes only the declared component, including
history input layout, and preserves missing ages and the target channel.
