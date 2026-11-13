# Additional-Domain Failure Localization

Posthoc descriptive analysis of the frozen all47target outer predictions.
No exclusion, fitting, calibration or promotion. Two synthetic regression
tests check equal-component aggregation, leave-one-out means and sparse support.

## Confirmed Differences

- MAP: vs process,23/36components harmed;17/27components with>=30people harmed.
  Supported-component mean delta+0.10717; all-component+0.30531. Removing any
  one component retains harm(+0.20630 to+0.34761). Against population,
  supported mean+0.23063 and leave-one-out+0.36826 to+0.55268.
  Therefore one small outlying component does not explain its failure.
- Sodium: all-component process delta-0.03339 changes to+0.01610 in the
  >=30people sensitivity subset;18/30supported components harmed. Against
  population23/30supported components harmed, mean+0.04582. Small-domain
  sensitivity is real, but excluding small groups does not repair performance.
- Glucose: supported process delta-0.34544, only3/30supported components
  harmed; leave-one-out all-component mean remains negative(-0.37578,-0.15406).
  Cross-component uncertainty/multiplicity still prevents the fixed screen.
- Heart rate: supported process delta-0.42905,4/28supported groups harmed.
  Leave-one-out means remain negative(-0.65552,-0.41264). Ordinary95%paired
  bootstrap intervals exclude0 against both controls, but47-target adjusted
  screen fails. Do not equate failure here with zero evidence of benefit.
- Bicarbonate: supported process delta-0.12715,2/28supported groups harmed;
  leave-one-out(-0.11725,-0.08842). Ordinary95%interval excludes0, but process
  component p=.002916 does not satisfy .05/47. Same gate retained.

Median bias direction is heterogeneous: MAP23positive/13negative components;
glucose22/16; bicarbonate18/20. Heart rate30/6 and sodium8/30 are more skewed
but still not universal. These are retrospective residual signs, not deployable
hospital corrections. Do not fit offsets from outer outcomes or infer that a
source-only correction will transport from these numbers.

## Remaining Discriminating Repair

Reserved multi-component source calibration can test bias transport with
frozen encoders, without changing the screening threshold. It remains unrun.
MAP's broad harm additionally requires checking whether its representation
contains usable conditional information; calibration alone is not a proven
solution. Positive mean targets and negative mean targets must remain distinct.
All47targets were analyzed, not only the examples above. Support sensitivity
does not replace the primary all-component analysis. No unique clinical cause
has been established; no forecast or runtime claim follows.

Artifact: `artifacts/native_reconstruction_additional_domain_diagnosis_20260924.json`.
