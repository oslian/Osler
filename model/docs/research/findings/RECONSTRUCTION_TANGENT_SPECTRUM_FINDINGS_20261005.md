# E44: removing weak tangent directions does not repair transfer

Current target-masked reconstruction only. No new forecast, self-learning,
clinical claim, gate pass or release promotion. The existing E43 correction was
split, not refitted. All five folds and 20,480 queries were completed and
independently replayed. Fold0/2 are repeated models, not independent replications.

## Question And Fixed Intervention

E43 fitted its local prediction Jacobian well but failed on unseen patients.
E44 tests whether removing the saved correction's weak parameter-sensitivity
directions repairs that failure. This differs from the older 64-dimensional
readout spectrum audit: the operator is the true recurrent prediction Jacobian
with respect to the single shared 64x448 bridge.

Use the original E43 lambda as the cutoff on squared singular values, the
original accepted alpha=0.5, the original data roles and exact control outputs.
No query-based cutoff, regularization or step search. Strong-only and weak-only
corrections are each replayed through the actual nonlinear model; their linear
approximations are separate diagnostics, not deployable predictions.

| Fold | Retained directions | Strong | Weak | Weak share of parameter squared norm |
|---|---:|---:|---:|---:|
| 0 | 2048 | 2000 | 48 | 6.42% |
| 1 | 2048 | 1952 | 96 | 11.35% |
| 2 | 2048 | 2000 | 48 | 6.42% |
| 3 | 2048 | 1987 | 61 | 5.07% |
| 4 | 2048 | 2012 | 36 | 3.63% |

Strong means sensitivity above this prespecified numerical cutoff. It does NOT
mean biologically meaningful, causally correct, or statistically supported in
new patients. Spectrum depends on parameterization, scaling and fitting support.

## Actual Errors

Normalized MAE, patients equalized within target, then targets equalized.
Lower is better. These are not physical-unit errors or accuracy percentages.

| Role | Scored targets | Zero bridge | Full E43 | Strong only | Weak only |
|---|---:|---:|---:|---:|---:|
| Fit | 47 | 0.753682 | 0.745849 | 0.746151 | 0.758774 |
| Selection | 46 | 0.876270 | 0.893038 | 0.892978 | 0.878618 |
| Same source | 44 | 0.781413 | 0.784977 | 0.787198 | 0.784675 |

Strong-only improves fit by 1.00% relative to zero, but worsens selection by
1.91% and same-source query by 0.74%. Removing weak directions barely improves
the selection average over full E43 and worsens its same-source average.
Weak-only also fails to beat zero in the aggregate query roles.

Component-equalized strong-only MAE is 0.786452 / 0.939614 / 0.779151,
versus zero 0.795575 / 0.921333 / 0.770998. The query deterioration therefore
remains under both fixed weighting schemes. The selection raw control remains
0.853856 patient-equalized and 0.871928 component-equalized.

Strong-only improves 34/47 fitting target means. Of 33 such improvements also
scored in selection, 26 lose their mean advantage; of 33 scored in same-source,
25 lose it. Selection still has 10/46 individual target mean improvements and
same-source 13/44; do not erase these counterexamples or claim all targets worsen.
These are descriptive counts, NOT significant harms or new promotion screens.

## Linear Diagnostic

| Role | Zero bridge | Full E43 linear | Strong-only linear | Weak-only linear |
|---|---:|---:|---:|---:|
| Fit | 0.753682 | 0.407425 | 0.409760 | 0.752433 |
| Selection | 0.876270 | 1.332393 | 1.321648 | 0.899474 |
| Same source | 0.781413 | 1.163262 | 1.165090 | 0.798739 |

The strong subspace preserves almost all the surrogate fitting gain and almost
all the surrogate transfer failure. This rules out weak-direction amplification
as a sufficient explanation or repair of THIS fixed E43 correction. It also
means perfect realization of the strong linear surrogate would not fix this
version's query average. Nonlinear outputs must not be added together as though
they were linear contributions.

This does not establish a unique root cause. Conditional relationships,
observation support, finite sample, task weighting and state/readout limitations
remain intertwined. Same-source failure means switching source identity is not
necessary for this failure, not that all domain-shift effects have been removed.
The original encoder exposure and stronger raw reader's larger fitting support
remain inherited limitations. Reused MIMIC/NWICU are not new external lockboxes.

## Numerical Failures And Verification

- Pre-run syntax and invalid synthetic identity-fixture errors were repaired;
  the original identity guard was not weakened. They are not model root causes.
- Run19824 stopped after fold0 Jacobian, at the float32 part-sum check. Rounding
  each part is not additive. The omitted conversion residual is now explicitly
  stored, never applied to either ablation model. Original tolerance unchanged.
- Run90628 completed fold0 but stopped on fold1 Gram back-projection
  orthogonality. Fit-only diagnosis11741 showed orthogonality error 2.0741e-7
  versus 4.7499e-11 with direct SVD, with identical subspace counts and maximum
  parameter difference 2.6122e-10. No query MAE was used to choose this repair.
- The original protocol and failed sources/partial outputs are retained and
  bound. Direct economy SVD computes the same projections without Gram inversion;
  all five folds were repeated using it. See the numerical amendment for paths
  and the explicit archived-source resolution of the numerical diagnosis.
- Formal38107 and independent49011 both exited0. Full E43 replay unchanged;
  independent forward replay of both saved ablations on all 20,480 queries has
  maximum error0. Original rows, controls and roles were checked unchanged.
- Independent scalar MAE/SmoothL1 agrees within 3.552713678800501e-15. All39
  original and30 verification bindings were rechecked. 34 related unit tests pass.
- Maximum complete slope-decomposition discrepancy is 4.884981308350689e-14;
  the separately disclosed casting contribution reaches 1.253156969084034e-6.
  Final decomposition orthogonality error <=4.4890e-10. These numerical checks
  do not measure clinical efficacy.

## Decision

Close this fixed weak-direction-removal branch. Do not sweep cutoffs or choose
per-target ablations using these query outcomes. No formal new efficacy gate
was run: E44 is a prespecified diagnostic ablation, not another promoted model.

The new evidence narrows the failed repair: even the strong retained correction
improves fitting while failing to transfer. The full root cause and a validated
generalization repair remain unresolved. Runtime, release weights, intervals and
safety anchors are unchanged. No commit or upload was performed.

## Evidence

- [Protocol](../../../RECONSTRUCTION_TANGENT_SPECTRUM_PROTOCOL_20261005.md)
- [Numerical amendment](../../../RECONSTRUCTION_TANGENT_SPECTRUM_NUMERICS_20261005.md)
- [Complete aggregate](../../../artifacts/native_tangent_spectrum_20261005.json)
- [Independent verification](../../../artifacts/native_tangent_spectrum_verification_20261005.json)
- [Fit-only numerical diagnosis](../../../artifacts/native_tangent_projection_numerics_20261005.json)
- [Prior E43 findings](RECONSTRUCTION_TANGENT_BRIDGE_FINDINGS_20261005.md)
