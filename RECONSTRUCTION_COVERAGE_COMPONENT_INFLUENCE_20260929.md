# Component Influence Diagnosis

Uses existing full outer predictions without refitting, exclusions or threshold
changes. Focus:30targets failing the unchanged descriptive base screen.
Per-component errors are means over its patients. Sensitivity to minimum30
patients is descriptive only; allcomponents remain in the original gate.

| Failed-target summary | Versus process | Versus population |
| --- | ---: | ---: |
| Majority of components improve | 29/30 | 27/30 |
| Negative mean after removing any one component | 26/30 | 24/30 |
| At least one leave-one-out mean sign flip | 4/30 | 2/30 |
| Mean sign changes when considering components >=30people | 2/30 | 2/30 |

These statistics overlap. They are not significance tests or new promotions.
They weaken the blanket hypothesis that numeric information helps only one
fortunate source. Most failures instead combine generally favorable signs with
insufficient precision or heterogeneous magnitudes under the fixed family test.
This does not prove low power: true group differences and sampling variability
have not yet been separated.

## MAP

Against process:23/36 components improve; mean rawdelta -0.01653, median
-0.13148, patient-weightedmean -0.19663. Removing one favorable component can
flip the overall mean to +0.00998. Nine components have<30people and average
+0.38937; the other27 average -0.15183 and20/27 improve.
This is a specific small-group sensitivity, not a license to discard smallgroups.
Against population28/36 improve and the negative mean survives every single
removal. The same target can fail different references for different reasons.

## Next Discriminating Step

Estimate within-component uncertainty of the paired error and compare it with
between-component spread, retaining allgroups. This can distinguish unstable
small-sample estimates from reproducible adverse domains descriptively. If
domain harms persist with adequate within-group precision, source-only matched
numeric perturbations/representation repair are warranted. If not, report the
uncertainty rather than making the model appear successful by changing the gate.
No posthoc exclusion, tuning, or new untouched-external claim is allowed.

Reproduce:`python -m scripts.diagnose_coverage_component_influence`.
Artifact:`artifacts/native_reconstruction_coverage_component_influence_20260929.json`.
Three unit tests cover unequal group weights, leave-one sensitivity and invalid
support. No runtime/calibration/registry changes.
