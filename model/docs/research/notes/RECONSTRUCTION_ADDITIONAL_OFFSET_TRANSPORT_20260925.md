# Source-Only Calibration Stability Diagnostic

All235target-folds examined. For each calibration component, fit the fixed
weighted median correction on OTHER calibration components, then evaluate its
change in MAE in the omitted component. No outer labels enter these fits.
Require the same support on each refit; do not treat unsupported zero offsets
as successful validation. Do not change the previously fixed offset experiment.

163target-folds have full source CV support and an outer result;72incomplete.
Descriptive sign concordance (component-equal delta, negative is helpful):

| Source calibration CV | Outer helpful | Outer nonhelpful |
|---|---:|---:|
| Helpful |42|21|
| Nonhelpful |47|53|

This is not a trained selector or an independent success probability estimate.
Related targets/folds are correlated. No outer-selected threshold applied.

## Distinguishable Failures

MAP source CV helps in5/5folds and outer correction helps5/5. Albumin source CV
harms5/5; outer correction harms4/5. Thus blindly applying supported offsets
ignored available source-only negative evidence for albumin. Support alone
does not establish correction utility. This explains part of the latest repair
failure, not the original encoder's clinical root cause.

Heart rate source CV helps4/5; outer helps3/5. Fold3 source CV+0.07814and outer
+0.24416(component MAE) correctly warn against applying the offset. Fold4
source CV-.04130but outer+.04994 shows that this diagnostic is not sufficient.

Bicarbonate source CV harms5/5; outer component correction harms3/5. Sodium
source CV harms4/5, but there is still source/outer disagreement. The21false
transfer signs and47missed useful corrections rule out perfect routing from
this simple signal. No blanket "all instability fixed" interpretation.

## Next Distinct Repair

A source-only activation decision can be tested as a NEW experiment: validate
correction utility across calibration components before allowing an offset,
otherwise preserve the uncorrected model. Apply the same rule to every arm,
fix its statistical threshold before assessment, report retained coverage and
missed gains. Do not choose the rule using which outer targets improve.
This is not a new encoder, clinical subgroup model or change to the core gate.
Outer data remains reused validation, so any apparent gain still needs new
confirmation. No existing candidate/runtime/promotion changed by this audit.

Two tests verify excluded-component labels cannot affect its fitted correction
and that insufficient refit support stays unassessed. All original dependency
hashes and235source offsets verified before analysis.
Artifact: `artifacts/native_reconstruction_additional_offset_transport_20260925.json`.
