# BUN Residuals Within Fit-Defined Prediction Bins

Fixed pooled source-fit prediction quartiles: .0446113,.3180589,.7320343.
No truth-defined strata, learned routing, correction or new inference.
Use hash-bound cached full predictions, fitting/selection identities disjoint.
Residual is truth minus prediction; negative means overprediction.

| Prediction bin | Selected fit median | Selected selection median (N) | Broad fit median | Broad selection median (N) |
|---|---:|---:|---:|---:|
| 0 |-.324|-.163 (11)|-.282|-.326 (1140)|
| 1 |-.358|+.069 (12)|-.432|-.396 (679)|
| 2 |-.260|+.157 (19)|-.497|-.509 (643)|
| 3 |-.361|+.006 (48)|-.445|-.578 (624)|

All four broad selection residual median bootstrap intervals are negative,
consistent with a broadly persistent overprediction offset, not just one
prediction-range subgroup. Selected selection first three bins have fewer
than30patients: no interval reported. Last bin95%interval[-.488,.464] is wide
and includes both zero and the fitting median. This does NOT establish a
statistically reliable conditional shift in selected selection.

The90selected sample has53.3%in the highest prediction bin versus28.2%of529
selected fitting samples. Both composition and uncertain within-bin residuals
must be considered. Coarse scalar matching is not full patient-state matching;
these results cannot prove that identical body states need opposite answers.
Do not infer missing physiology or an unavoidable information ceiling from it.

## Decision

Broad offset replication is supported over the observed prediction range.
The smaller selected cohort cannot reliably adjudicate per-bin correction;
prior claims of a definitive opposite physiological regime would be too strong.
Do not build a four-bin correction using selection residuals. Further work
should improve independent support for the selected cohort or inspect its
predefined sampling/anchor contract, not fit another correction on90people.

Two median/boundary/support tests pass, audit exit0, no runtime/promotion.
Intervals are unadjusted descriptive patient bootstrap, not a new lockbox.
Current reconstruction only. Evidence:
artifacts/native_reconstruction_bun_conditional_residual_20260920.json.
