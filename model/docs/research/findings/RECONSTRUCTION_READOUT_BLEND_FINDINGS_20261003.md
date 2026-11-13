# Same-State Readout Blending Does Not Resolve the Shared Error

## Scope

Completed `RECONSTRUCTION_READOUT_BLEND_PROTOCOL_20261003.md`: all47 targets,
all5 source folds,1,797,884 query occurrences. Native and body HGB read the SAME
frozen shared state. No alternative encoder or forecast model was introduced.
The blend coefficient was fitted on other source components, never the scored
component, using the exact equal-component MAE solution. Process and trained-
shuffle blends received the same fitting procedure. The fixed50:50 blend is
also retained; previous learned prefix predictions were not fed into another
fitting layer because that could leak held-component outcomes indirectly.

This replicates an existing ensemble idea on the new reconstruction readouts,
not a claim that blending was previously untested. Earlier negative comparisons
are `docs/research/notes/CLEAN_CONTEXT_COMPLEMENTARITY_20260911.md` and
`docs/research/findings/NEAR_MISS_PORTFOLIO_FINDINGS_20260908.md`.

Forty-four targets have complete selection support. Unsupported components of
anion_gap/myoglobin/pH remain explicit; native-only bookkeeping cannot pass.
All original source means and identities reproduce. Thirteen aggregate report
bindings verify, and26 focused tests pass. Private row predictions stay local.

## Main Result

Zero targets pass all six corrected comparisons. Zero establish incremental
gain over body100 at both patient and component levels under the fixed screen.
Nine pass the native comparison, but none passes all required comparisons;
these are not nine new capabilities.

| All47 target-macro normalized MAE | Patient equal | Component equal |
|---|---:|---:|
| Native | 0.882389 | 0.831593 |
| Body100 | 0.857139 | 0.811241 |
| Fixed50:50 | 0.860615 | 0.812544 |
| Component-selected body blend | 0.856785 | 0.810552 |
| Matched process blend | 0.881788 | 0.830657 |
| Matched shuffled blend | 0.882646 | 0.831222 |

For44 fully supported targets, body100->selected blend is0.875882->0.874097
patient and0.831573->0.829499 component. Small average improvement is not proof
of stable target-specific benefit, and equal blending is worse than body100.

Nine targets numerically improve over BOTH individual heads at BOTH levels:
cortisol,CPK,hematocrit,LDH,lipase,magnesium,MAP,O2sat,serum osmolality. None
passes the complete screen. No target was removed based on its result.

## Errors Are Mostly Shared, Not Independent Opinions

Across47 targets, median patient-level signed-error Pearson correlation is
0.98937; median Spearman correlation is0.96640. Median patient-equal fraction
of opposite-sign errors is9.369%. Large Pearson correlation can partly reflect
a shared true outcome or extreme values; it alone does not prove shared
physiological ignorance. Rank correlation and actual blending performance are
therefore also reported.

The exact50:50 cancellation is measured relative to AVERAGE member error, not
the better member. Arithmetic cancellation is guaranteed when errors oppose;
beating the best individual predictor is not. There is no outcome-aware row
oracle or claim that the observed winner can be known before seeing truth.

## What Happened to the Main Failure Targets?

Patient-equal normalized MAE (not physical units):

| Target | Native | Body100 | Selected blend | Mean body weight |
|---|---:|---:|---:|---:|
| Ferritin | 0.827030 | 0.845038 | 0.827473 | 0.1216 |
| Troponin I | 5.184358 | 5.226455 | 5.185463 | 0.1566 |
| O2 saturation | 0.600807 | 0.602190 | 0.600062 | 0.5730 |
| BUN | 0.635639 | 0.614053 | 0.614686 | 0.9036 |
| INR | 0.893784 | 0.860433 | 0.861140 | 0.9488 |

Ferritin and troponin improve relative to the harmful body100 readout primarily
by moving back toward native, not by surpassing native. Their component body-
vs-blend reductions have nominal conditional95%CIs below0, but the corrected
screen fails and the native reference still wins numerically at both levels.
For ferritin the body coefficient is exactly0 on74.31% of patient-weighted
choices; for troponin51.38%. Harm reduction is not newly identified physiology.

O2sat has small mean complementarity, but its body100 comparison's component CI
[-0.00509597,0.00184817] crosses0. Fixed50:50 also has lower means than the
learned blend, so weight estimation is not an established improvement there.

BUN/INR weights mostly retain body100, yet blending slightly increases both
mean errors. ALT chooses body100 exactly everywhere; AST almost does. Good
individual readouts should not be weakened merely to obtain a combined model.

## Decision

This intervention does not solve the generalization problem. The newly tested
readouts contain some complementary errors, but simple legal averaging and a
one-parameter source-component fit do not establish the required incremental
benefit. A support-only selector, earlier stopping and blending now each have
direct complete-family evidence on this SAME new candidate set; do not present
them as untried generic repairs.

This does not prove that all further learning is impossible or that no better
state exists. It limits the value of manipulating existing correlated outputs.
Further work should isolate missing information in the representation or
conditional supervision, with source-only matched controls, rather than adding
another unrestricted combination rule and selecting it on reused outer data.

Machine-readable evidence:
`artifacts/native_reconstruction_nonlinear_blend_20261003.json`.
Source data and overlapping fits have been reused. Bootstrap is conditional and
does not include refitting mixture weights, so these are diagnostic screens,
not new external validation. Current masked reconstruction only; no forecasting,
causal/clinical claim, runtime weights, intervals or promotion changed.
