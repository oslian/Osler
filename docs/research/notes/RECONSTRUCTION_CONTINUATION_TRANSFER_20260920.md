# Continuation Source-to-Evaluation Localization

Compare the frozen selected checkpoints7 and16 for all33trained targets, with
the exact same preexisting source selection and evaluation cohorts. Original
eight-epoch training logs reproduce exactly, report bindings verified, two sign/
nonfinite tests pass. No new fitting, routing or epoch selection.

| Source mean change / evaluation mean change | Broad | Selected |
|---|---:|---:|
| Improve / improve | 13 | 16 |
| Improve / not improve | 5 | 3 |
| Not improve / improve | 10 | 9 |
| Not improve / not improve | 5 | 5 |

These are numerical sign counts, not significant harm/gain claims. Equal errors
are classified as not improved, not as harms. Six untrained targets remain
explicitly untrained.

## Creatinine

Source broad MAE1.572594->1.569180 and source selected1.635209->1.631777 both
improve slightly. Evaluation broad1.435759->1.445204 and selected1.420725->
1.447244 both worsen. Thus the selected checkpoint's creatinine regression
cannot be explained simply as sacrificing a visibly worse source creatinine
score to improve aggregate selection. Source signs did not transfer. This
does not prove irreducible shift: source gains are very small and the original
selection sets remain90patients per group. Sampling noise, case composition
and learned dependence are not isolated by this calculation.

Lipase likewise improves source but worsens evaluation in both groups. In the
other direction, troponin improves both evaluations despite worse source scores.
Therefore a rule that protects all tiny source MAE gains would also risk
rejecting useful changes. No per-target oracle should be built from this table.

## Next Discriminating Evidence

The all3323 broad-source selection replay is complete only through epoch8.
Replaying new savedepochs9-19 on that same source set can test whether the
creatinine sign reversal already disappears with better source support, before
another training intervention. It is not a fresh external test and cannot
guarantee generalization. Keep evaluation labels out of selection and preserve
paired controls and full39-target scope.

Evidence: artifacts/native_reconstruction_continuation_transfer_20260920.json.
No runtime, interval, registry, weights or promotion changes.
