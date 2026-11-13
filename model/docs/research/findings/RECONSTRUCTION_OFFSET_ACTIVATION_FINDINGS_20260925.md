# Source-CV Offset Activation: No Full Incremental Pass

Fixed protocol used complete leave-component-out support, at least4groups,
negative mean and one-sided p<.05 to activate. Same rule for all47targets,
5folds and3arms, no threshold sweep. Outer reused validation only.

Physiology8/235target-folds active, process5/235, population0. Of8activated
physiology target-folds,5improve component-equal MAE and3worsen. These are
target-fold counts, not counts of individual components or independent trials.
No target passes the full incremental comparison against unchanged physiology
and identically gated controls under patient/component .05/47screen.

Normalized target macro patient MAE .925567->.925203; component .901157->.900760.
Improvement is tiny; unconditional offset achieved .917067/.897123 but also
caused harm. Conservative activation does not recover its helpful corrections.

Albumin disabled5/5 (avoids previously observed calibration harm), but MAP
disabled5/5 despite all five unconditional outer corrections helping. The three
enabled component harms are potassium fold0(+.00599), ASTfold1(+.16404),
O2satfold3(+.22649). O2sat also harms patient MAE(+.21595). Thus low activation
alone does not explain all failures; selected corrections still sometimes fail
transport. Do not relax the threshold using this outcome or cherry-pick cells.

Two tests verify complete-support/sign/fixed-threshold logic and identical
all-arm behavior with exactzero when disabled. Original dependencies hash-
checked, source decisions computed independently of outer outcomes. Existing
predictors/weights/registry/intervals unchanged. Candidate-only. No forecast,
clinical, causal or untouched external validation claim.

This disproves neither conditional calibration nor shared physiology. It shows
that this simple source-component utility heuristic is insufficient. Before
another selector, distinguish component composition/conditional relation shift
from estimate uncertainty in the enabled failures, without fitting on their
outer labels. Original13descriptive reconstruction screens remain separate.

Artifact: `artifacts/native_reconstruction_offset_activation_20260925.json`.
