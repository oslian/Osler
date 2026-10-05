# Exact missingness control: numerical treatment increment remains unproved

Seven fixed-setting fits. Original and transition real checkpoints reproduce
all scores exactly. Support control keeps every missing position unchanged,
replaces every finite treatment entry with zero, and refits from source only.
No donor patient or availability shift. Own history is unchanged.

| Role/group | Numerical treatment | Support only | Own |
|---|---:|---:|---:|
| Source/all | .422908 | .422819 | .424584 |
| Source/sequence | .394722 | .398194 | .402279 |
| Hospital/all | .366660 | .365074 | .365562 |
| Hospital/sequence | .343361 | .337547 | .336886 |

Numerical vs support gain0/7 in every group. Significant harm0/7source,
1/7hospital all and sequence. CIs do not prove equivalence or absence of
information. They fail to establish incremental numerical treatment value
under this readout, even when placebo availability mismatch is removed.
Support includes whether a last-change exists, not merely hospital identity.
Do not claim workflow-only physiology or that every alternative encoder fails.

Artifact `artifacts/expanded_insulin_support_control_20260911.json` uses
historical key shuffled for the explicitly declared support-only control;
it is NOT the earlier donor placebo. All weights local, no runtime change.
No calibration, external untouched validation or promotion.
