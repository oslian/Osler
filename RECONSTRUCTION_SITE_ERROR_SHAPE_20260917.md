# Reconstruction Error Shape Localization

## Scope

Frozen selected strict-site predictions, all33 supported targets and all6
unsupported catalog entries retained. No fitting, trimming, label removal or
selection. All original MAEs reproduce to1e-10. Each row is one patient-target;
patient identities, evaluation role and saved artifact hashes are checked.

Central/tail boundaries use the FIT median and90th percentile absolute distance
from it, independently per target. Evaluation membership uses true current
values and is therefore RETROSPECTIVE: it must not become a deployment gate or
evidence that a model can identify such patients in advance. Values below are
normalized reconstruction errors, not physical-unit forecast errors.

## Findings

| Target | N | Real beats process fraction | Median paired error difference | Central mean difference | Tail mean difference |
|---|---:|---:|---:|---:|---:|
| Lipase |62|37.1%|+0.1093|+0.1771 (57)|+0.4975 (5)|
| Triglycerides |30|40.0%|+0.2521|+0.4198 (26)|-0.0598 (4)|
| Bilirubin |137|56.9%|-0.0707|+0.0354 (122)|-0.8930 (15)|
| Glucose |176|53.4%|-0.0365|+0.0238 (140)|-0.3592 (36)|
| Creatinine |177|58.8%|-0.0491|-0.0276 (151)|-0.7734 (26)|

Negative differences favor real numerical input over the trained process model.
These descriptive conditional means and fractions do not have new significance
claims. Small tail counts cannot establish subtype generalization.

Lipase and triglycerides harm is not confined to a few tail patients: their
median paired error differences are positive and most patients do worse.
Triglyceride central errors contribute+0.363788 to the total mean difference,
while tail errors contribute-0.007975. Tail suppression alone is not supported
as the repair for this particular failure.

Triglyceride real prediction Spearman correlation is-0.4971 on30patients versus
-0.0663 process. An additive bias adjustment preserves ranking, and positive
linear shrinkage does too (zero shrinkage gives a constant). Thus such fixes
cannot repair the observed ranking reversal, though they could still reduce
MAE. Do not reverse the prediction sign based on evaluation outcomes. No
confidence claim or cross-cohort generalization is attached to this correlation.

Glucose and bilirubin show a different pattern: favorable overall means coexist
with central mean harm and larger tail gains. This does not demonstrate a usable
stability selector, because group membership here uses the withheld truth.
Creatinine mean gains appear in both groups, with a much larger tail effect.

## Implications

The failure cannot be reduced to one common score, a few extreme labels, or a
fixed output offset. The next source-only repair should test whether patient
ranking is retained across source sites, before fitting an output calibration.
For any regression candidate, report ranking, mean/median paired errors and
all-target retention together; do not optimize evaluation subgroups or discard
tail values. Existing source generalization evidence remains relevant.

This is additional failure localization, not a proven mechanistic cause or
accuracy repair. No weights, runtime registry, intervals or promotion changed.

Artifact: `artifacts/native_reconstruction_site_error_shape_20260917.json`.
Runner: `scripts/audit_site_reconstruction_error_shape.py`.
