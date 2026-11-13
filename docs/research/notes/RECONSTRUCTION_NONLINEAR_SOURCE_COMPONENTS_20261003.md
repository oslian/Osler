# Different Reasons for the Remaining Nonlinear Readout Failures

## Scope and Verification

All47 targets and four fixed references (native, process summary, population,
matched trained shuffle), SOURCE-SELECTION rows only. No new training, no
outer outcome-based selection, no threshold changes.1,797,884 source query
occurrences are reduced to one averaged paired loss per patient/target over
eligible folds. Every original source prediction, patient count, component
count and aggregate normalized MAE reproduces the completed reports.

Reuse the existing component influence, conditional variance decomposition,
paired bootstrap and fixed patient-hash halves. Within-component95% intervals
use1000 paired resamples and are NOMINAL descriptive intervals, not multiple-
testing corrected discoveries. Support>=30 labels a group; it never excludes
it. Hash-half agreement requires>=15people per half. Shared fits/development
reuse prevent treating halves or folds as independent replications.

## The Previous Generalization Needs Qualification

The blanket wording "benefit is unstable across groups" mixed two situations:
benefit DIRECTION can reverse, or its MAGNITUDE can vary without any mean harm.
This distinction changes which repair is justified.

**BUN and INR: all12 source components have lower mean error against ALL FOUR
references.** Removing any one component preserves the negative mean difference.
No supported nominal harm or supported two-half harm occurs in any comparison.
They still fail parts of the prespecified component-level family screen.
This is not evidence that these two probes generally pick the wrong direction
of improvement across source components.

For BUN versus process-summary, component mean differences range from
-0.46047 to -0.01526. The mean remains at most-0.08524 after removing any one
component, but the previous component p=.002699 misses0.05/(8*47). Magnitudes
vary greatly; all-negative signs do not guarantee a tightly estimated mean under
this prespecified test. The conditional estimated sampling contribution is
about13.1% of observed between-component spread here, so simple finite-patient
sampling noise does not explain all spread under the estimator's assumptions.

INR versus native ranges-0.20069 to-0.01434. A singleton component makes its
complete variance decomposition unidentifiable. It is retained with unknown
sampling variance, not silently assigned zero variance or removed. Neither
the all-negative means nor the missing variance justify declaring promotion.

## Other Promising Targets

Numbers are source component mean harms (not statistically proven harms).

| Target | Native | Process summary | Population | Matched shuffle |
| --- | ---: | ---: | ---: | ---: |
| BUN |0/12|0/12|0/12|0/12|
| INR |0/12|0/12|0/12|0/12|
| Calcium |1/12|0/12|1/12|0/12|
| Heart rate |2/12|1/12|2/12|2/12|
| Potassium |0/12|2/12|1/12|1/12|

None of these five has a supported nominal harm interval against these
references. However, heart rate has one supported two-half adverse sign versus
process, and potassium has one versus population and matched shuffle. Thus
they cannot all be dismissed as identical small-sample near misses.

For several comparisons the estimated sampling variance exceeds observed
between-component spread (for example BUN/native ratio4.54, K/native5.39).
The signed excess is retained, not clipped or interpreted as negative biological
heterogeneity. This rough decomposition is conditional on fixed fitted models
and independent patient errors; it is not a power calculation, a causal
decomposition or proof that more people must fix the gate.

## Genuine Performance Counterexamples

| Target/reference | Mean harms | Supported nominal harm intervals | Supported two-half harms |
| --- | ---: | ---: | ---: |
| Ferritin versus native |10/12|1|2|
| O2sat versus native |6/12|2|4|
| Troponin I versus native |8/10|4|5|
| Troponin I versus process summary |9/10|3|5|
| Troponin I versus population |9/10|3|4|

Troponin's adverse pattern is already present in source data, not merely a
surprise in the outer cohort. Repeated signs and nominal intervals are warning
signals, not corrected subgroup discoveries. Nevertheless it would be wrong to
describe every failed target as "works, only lacks significance."

Across the entire family, targets with at least one supported nominal harm are
18 versus native,9 versus process,9 versus population and5 versus matched
shuffle. Counts overlap and are not independent tests. All-component beneficial
mean signs occur in7/15/13/18targets respectively. Negative mean differences
survive any single-component removal in29/42/41/44targets. Full lists and every
component remain in the aggregate artifact; no favorable-only reporting.

## Concrete Diagnosis and Repair Implications

1. **Readout utilization is partly repaired, not universally repaired.** The
   prior fixed-body nonlinear improvement plus matched shuffle result supports
   useful patient-specific state information. NativeBodyReconstruction.readout
   is a linear64-to-variable map; the new learner changes readout capacity,
   fitting and loss. It does not isolate a single optimizer or encoder defect.
2. **For BUN/INR, repeatedly changing architecture is not justified by this
   failure alone.** Their present source blocker is the certainty of incremental
   group-level benefit, with only12selection components and unequal effect sizes.
   The safe next evidence collection is a prespecified additional DEVELOPMENT
   component evaluation of the already frozen heads, where those components
   never fitted that encoder/head. It must not consume the same data for tuning
   and later claim untouched calibration/external validation. Changing the
   threshold after seeing these signs is not a repair.
3. **For ferritin/troponin/O2sat, prediction errors still need repair.** Their
   source adverse groups deserve examination of conditional bias, tails,
   observation support and objective mismatch before any deployment. A new
   matched readout comparison must isolate its claimed change. Earlier33-target
   SmoothL1/LAD/site-constrained studies already exist; do not claim those ideas
   were never tried or casually repeat global calibration.

This diagnostic does not complete the overall generalization goal. It prevents
using one imagined cause and one replacement model for all failed cells. The
runtime, intervals, encoder/readout weights and promotion status are unchanged.
Current-value reconstruction evidence is not future-trajectory evidence.

## Reproduction

`python -m scripts.diagnose_nonlinear_source_components`

`python -m unittest tests.test_nonlinear_source_components tests.test_coverage_variance tests.test_coverage_component_influence`

Nine focused tests pass, including per-patient loss averaging (not prediction
averaging), duplicate/outer-role rejection, retained singleton uncertainty and
stable half-split harm detection. Completed output refuses overwrite.

Artifact: `artifacts/native_reconstruction_nonlinear_source_components_20261003.json`.
Only aggregate groups and hashes are stored in Git; source patient rows remain
local. CodeGraph was unavailable (`command not found`); direct source/index
inspection was used. All work in this stage completed; no live jobs remain.
