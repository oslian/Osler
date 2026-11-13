# Local Organ Exchange: Results and Limits

Date: 2026-09-06. Research-only; no runtime promotion or causal claim.

## What was actually tested

This tests a structural part of the proposed organ-state learning approach,
NOT a fully retrained anatomical body model. The same released JEPA encodes
seven owner-restricted observation histories (heart, vessels, lungs, kidneys,
pancreas, marrow, brain). Non-owner values, masks and ages are removed before
history encoding. The checkpoint's functional-system tokens are NOT relabelled
as anatomical organs: each new node receives the owner-restricted context latent.

One recurrent organ/body state evolves on a fixed hourly grid. Each target
reads its owner-organ change at its own actual event time by interpolation.
Zero-start value and target-JEPA residual heads augment frozen owner forecasts.
The shared backbone is NOT retrained here. No physical hidden-store meanings,
mass conservation or treatment effects are asserted.

The four arms have identical 154,956-parameter candidates, initialization,
optimizer, training batches and eight epochs: no edges, specified sparse
edges, permuted edges, dense edges. The permutation preserves the total edge
count and degree multiset, not each organ's degree. Consequently this is a
topology control, not a conclusive test of individual biological edges.

## Two implementation hazards isolated

1. Identity adjacency did not by itself isolate organs in the existing layer:
   pooled body information could still feed every organ. The experimental
   `body_feedback=False` path and owner-only readout close that bypass. Default
   existing behavior is unchanged; shared feedback is not inherently a bug,
   but it invalidates interpreting identity adjacency as full isolation.
2. Advancing through the sorted union of different targets' event times can
   change integration boundaries for an earlier target. A fixed hourly path
   with target-local interpolation prevents one target's query time changing
   another target's prediction. Future labels never enter the context encoder.

Both properties have direct regression tests, along with zero-start parity,
equal parameter counts, non-owner history removal, strict label validity,
and nonzero transition gradients after the zero-start heads begin learning.

## Cohorts and training

- Pilot: 1,800 source development + 600 held-out-hospital patients; 800 MIMIC.
- Expanded: 6,000 source development + 2,000 held-out-hospital patients;
  159,970 source history rows; 2,400 MIMIC patients with 25,871 history rows.
- One deterministically selected eligible anchor per patient. Its selection
  does not inspect future values or errors; pair availability affects eligibility.
- Patients/stays named in the pretrained checkpoint's event-example file are
  excluded from eICU before sampling. This does not erase prior research exposure
  or establish complete provenance of every historical encoder experiment.
- Source support is fixed to 14 observed variables. Source-only support-discovery
  patients are always training-only. All four arms have the same input support.
- Seven genuinely separate patient train/calibration/test partitions; same
  initialization seed throughout. The held-out-hospital cohort is separate.
- The pilot had only 74-86 valid BUN/creatinine test patients per split. The
  support-only expansion was declared before its results; no model/threshold
  was chosen using MIMIC outcomes. Expanded source test support is 243-691
  patients per target/split, above the 100-patient screen.
- MIMIC is reused supplementary validation, not an untouched lockbox. This
  MIMIC cache has nominal 6h labels but lacks actual future event timestamps;
  its results are NOT strict irregular-event-time external validation.

## Expanded source hospital results

MAE in the existing transition target units; median over seven fitted models.
Hospital patients are the same in each split, not seven independent datasets.

| Target near 6h | Patients | Sparse graph | No edges | Permuted | Owner JEPA | Persistence |
|---|---:|---:|---:|---:|---:|---:|
| Creatinine | 633 | 0.384837 | 0.385266 | 0.384069 | 0.407710 | 0.360553 |
| BUN | 630 | 5.880088 | 5.879950 | 5.873211 | 7.093992 | 4.877778 |
| Urine output | 1190 | 155.603210 | 155.647034 | 155.607071 | 158.550598 | 157.736130 |
| Potassium | 854 | 0.386534 | 0.386676 | 0.386374 | 0.395840 | 0.456733 |
| Bicarbonate | 775 | 2.402831 | 2.404387 | 2.402726 | 2.617949 | 2.473936 |

Sparse beats owner JEPA by pooled paired-bootstrap CI in all 7 fits for all
five targets at held-out hospitals. That is NOT evidence of cross-organ value:
the no-edge and permuted arms make similar improvements. Sparse wins against
no-edge in only 0/7 creatinine, 0/7 BUN, 4/7 urine, 1/7 potassium and 4/7
bicarbonate fits; against permuted, only urine has 1/7 and the others 0/7.
No target passes all control comparisons across all patient partitions/roles.

## Supplementary MIMIC results

| Target near nominal 6h | Patients | Sparse graph | No edges | Persistence |
|---|---:|---:|---:|---:|
| Creatinine | 376 | 0.282129 | 0.282577 | 0.248404 |
| BUN | 369 | 4.732682 | 4.712913 | 3.753387 |
| Urine output | 2274 | 101.687454 | 101.653542 | 105.208885 |
| Potassium | 549 | 0.379605 | 0.379858 | 0.422404 |
| Bicarbonate | 382 | 2.020877 | 2.021946 | 2.103141 |

Only creatinine@6h is an actual cell in the current 41-cell registry. The other
four nominal 6h tasks have no matching shipped cell. Their report's `router`
reference is a persistence policy, not an evaluated shipped expert. This is
explicit in `router_reference_status`; do not call these five promoted cells.

## Uncertainty and diagnosis

Source-only split conformal intervals were computed for all four arms. Median
pooled sparse coverage at held-out eICU hospitals is 0.9131 / 0.9270 / 0.8748 /
0.9180 / 0.8645 (creatinine/BUN/urine/K/bicarbonate). MIMIC is 0.9282 / 0.9458 /
0.9477 / 0.9180 / 0.9241. Pooled coverage inside the band is not a claim that
every care unit passes; per-unit results are in the JSON. Intervals were not
attached to the release.

What the data support:

- Local residual fitting helps several owner-only forecasts.
- The specified topology has not shown reliable incremental value over matched
  no-edge/permuted controls. More edges are not a demonstrated solution here.
- The source sample-size deficiency was addressed; the topology result remained.

What remains uncertain:

- Owner masking changes what the frozen encoder sees. The full joint nominal
  baseline is better than the owner baseline for several targets, particularly
  BUN/creatinine. This is consistent with initialization/domain mismatch but is
  not proof of its cause. A frozen-encoder experiment cannot decide whether
  end-to-end anatomical observation/state training would succeed.
- The inherited 0.25-normalized-unit residual cap is restrictive. At held-out
  hospitals the median fractions near that cap are 24.6% creatinine, 31.3% BUN,
  61.3% urine, 2.2% K and 12.4% bicarbonate. Do not interpret a bounded-residual
  failure as rejection of all mechanistic/state-space models. The cap was not
  changed using external results.
- No physical stores or fluxes are supervised. Anatomical names plus local
  connectivity do not automatically teach organ function.
- Full target/sibling-masked recovery, end-to-end history assimilation,
  multi-horizon truth-supervised trajectories, precise MIMIC event labels and
  genuinely independent validation remain uncompleted by this experiment.

## Verdict and reproducibility

**No new promotion.** Existing model weights and runtime selection remain intact.
This is a negative structural pilot with explicit limitations, not evidence
that the body has no organ interactions or that the entire proposed approach
has been tested and rejected.

```bash
python -m scripts.audit_organ_exchange
python -m scripts.audit_organ_exchange \
  --output artifacts/organ_exchange_expanded_20260906 \
  --source-patients 6000 --hospital-patients 2000 --external-patients 2400
python -m unittest discover -s tests
python -m scripts.validate_release
```

Output directories must not already exist. All candidate weights are stored
outside `models/`; reports contain aggregate metrics, not patient identifiers.
The deterministic verified pilot reproduced every original MAE exactly.
Final unit/regression suite: 204 tests passed. `git diff --check` passed.
Release validation returned `status: ok`, 41 catalog cells, 27 movable cells
and 14 fallback cells. No candidate was attached to that release.

Artifacts:
- `artifacts/organ_exchange_20260906/report.json`: initial pilot.
- `artifacts/organ_exchange_verified_20260906/report.json`: exact reproduction
  plus diagnostics and explicit missing-registry reference status.
- `artifacts/organ_exchange_expanded_20260906/report.json`: support expansion.
- `ORGAN_EXCHANGE_PROTOCOL_20260906.md`: method, boundaries and expansion record.
