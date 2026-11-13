# Frozen nonlinear transfer result

All21 source-selected trees refitted using only their original fitting
patients. Every own/full/support source evaluation MAE reproduced within1e-10.
Leaf counts were not retuned, external patients never entered fitting, and
the external current-value encoding matched reconstructed potassium labels.

Patient-equal potassium MAE, median across seven source seeds:

| Role | Patients / pairs | Persistence | Own | Full | Support |
|---|---:|---:|---:|---:|---:|
| eICU held-out hospitals | 207 / 819 | .521309 | .444111 | .436375 | .447481 |
| MIMIC | 859 / 5674 | .428901 | .381259 | .379362 | .380675 |

Full vs own: significant gain0/7 and harm0/7 in both roles. Both full and own
beat persistence7/7 in both roles. Full vs support gains2/7 hospital and1/7
MIMIC, no significant harm. Median paired full-minus-own difference is
-.009151 hospital and+.000528 MIMIC. Difference of marginal medians is not
the median paired difference; do not infer a consistent MIMIC gain from the
table alone.

MIMIC care-unit full-vs-own gain: CVICU1/7; remaining five units0/7. CCU has
significant harm2/7. All units have at least30 patients. Full vs persistence
passes7/7 except MICU6/7. No posthoc care-unit routing or threshold changes.

## What This Localizes

The source nonlinear partial gain (full-own3/7) does not establish stable
external incremental gain (0/7). Failure is therefore not isolated to JEPA
latent compression: an encoder-free nonlinear readout also loses robust
incremental evidence. The usable own-history forecast remains strong.

This narrows the unresolved failure to learning transferable incremental
context beyond own history under this observation/cohort contract. It does
not prove biological information is absent, identify one causal mechanism,
or show trees and JEPA are equivalent. The earlier neural external trial used
a different fitting partition; do not claim a paired tree-vs-neural advantage.

Scope: glucose-result-available eligible anchors, with real potassium current
age <=6h and future5-7h. Not a representative all-potassium cohort. History is
lab-only; unavailable bedside and treatment streams remain missing. Seeds
share external patients and CIs are unadjusted descriptive comparisons.
MIMIC is reused validation, not an untouched lockbox. No interval calibration,
runtime weights, registry, serving change or promotion.

Protocol: `POTASSIUM_TREE_TRANSFER_PROTOCOL_20260911.md`.
Artifact: `artifacts/potassium_tree_transfer_20260911.json` (complete=true).
Command: `python -m scripts.evaluate_potassium_tree_transfer`.
