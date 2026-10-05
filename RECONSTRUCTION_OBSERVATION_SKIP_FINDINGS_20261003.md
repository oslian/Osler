# Observation Skip: Improvement Over Body Readout, Not Over Raw Observations

## Completed Scope

All 47 masked CURRENT reconstruction targets, all 5 original source folds,
29,580 unique selection patients, 720,470 patient-target pairs and 1,797,884
query occurrences. 705 new absolute-error HGB fits completed. Identical frozen
epoch-10 body states; target/sibling-masked raw summaries from the same legal
events. Original native/body/raw/population results reproduce to the assessment
tolerance. Fit/selection patients and whole-history components are disjoint
within each fold. No outer data read, new encoder or runtime change.

The new readout receives body64 plus raw265 (width329). Matched-budget controls
receive body plus process-only summary, or body plus a fit-component-permuted
raw summary. At query time all receive their own valid summary. The same tree
budget is not identical effective capacity with the 64/265-wide references;
the shuffled branch is not a full encoder null.

## Result

| Source normalized target-macro MAE | Patient equal | Component equal |
|---|---:|---:|
| Original native readout |0.882389|0.831593|
| Body-only nonlinear readout |0.857139|0.811241|
| Raw-only nonlinear readout |0.825737|0.787947|
| Body + real observations |0.828889|0.784964|
| Body + process summary |0.854767|0.811346|
| Body + shuffled observations |0.858132|0.813096|

Compared with body-only, observation skip reduces macro MAE 3.30% patient and
3.24% component. Excluding ALT/AST still improves body .837976->.812596 patient
and .798070->.774226 component. Improvement is not just two liver markers.
46 of 47 targets improve BOTH means by more than 1e-12; chloride improves
patient MAE but worsens component MAE (.603874->.653635).

All screens use the frozen protocol: >=30 patients, >=3 components, negative
paired means, bootstrap95% upper<0 and p<.05/(47*6) at BOTH levels. These are
conditional development diagnostics, not full runtime gates.

- 10 targets pass versus body-only: albumin, ALT, AST, bicarbonate, heart_rate,
  hemoglobin, magnesium, o2sat, potassium, total_protein.
- 6 of these ALSO pass versus process-skip AND shuffled-skip: albumin, ALT, AST,
  bicarbonate, heart_rate, total_protein. They also pass native and population.
- 0 targets pass versus raw-only. Consequently 0 pass the full six-comparison
  conjunction. Do not report six branch increments as six new promotions.
- 17 targets numerically improve both means over raw-only, but none passes the
  prespecified comparison. The macro slightly loses raw-only at patient level
  (0.38% worse), and slightly improves it at component level (0.38%).

## What This Localizes

The practical failure is not simply absence of usable cross-variable numeric
information. For six targets, exposing patient-matched observations beside the
frozen body state improves held-component reconstruction beyond both augmented
controls. This intervention repairs part of the body-only/readout pathway's
underuse of available observations on the reused source population.

However, the SAME experiment does not show a stable increment from adding the
body state to a raw-observation readout. Some gains could be easier feature
access rather than irreversible encoder information loss. The encoder and
readout, optimization and changed feature geometry are not separately identified.
A bigger latent or more heads is not established as the necessary remedy, and
these results cannot explain future forecasting failures by themselves.

| Target | Body-only patient MAE | Observation skip | Raw-only |
|---|---:|---:|---:|
| Albumin |0.340241|0.314491|0.318774|
| Bicarbonate |0.608804|0.591052|0.593115|
| BUN |0.614053|0.578155|0.578286|
| Creatinine |0.873193|0.821397|0.820478|
| Glucose |0.701064|0.688559|0.689013|
| Hemoglobin |0.444886|0.421804|0.425513|
| INR |0.860433|0.831776|0.833173|

Albumin's incremental advantage over raw is small: patient delta -.004283,
component delta -.004499, component95%CI [-.010396,+.000890], with 4 of 12
component means worse. BUN's raw comparison patientCI [-.001799,+.001539]
includes zero. Creatinine is numerically worse than raw at BOTH levels; this
is not merely a strict significance threshold blocking a better predictor.

Ferritin remains worse than native at both levels despite reduced body-only
harm. Troponin improves native patient mean but not component mean, and loses
raw at both levels. Explicit observations do not solve all transport errors.

## Decision And Limits

Keep diagnostic candidate only; no replacement or new interval. Retain both
results: explicit legal observation access helped, while a transferable
increment from the compressed state beyond those observations remains unproved.
Do not repeat simple blending, support-only switching or this same raw skip as
an untested universal fix. A subsequent representation test must isolate what
information the state retains, rather than infer absence or loss solely from
competing readouts. A new method would still need an uninspected final cohort.

12 tests passed: shapes, unmodified shared state, numeric/process separation,
pairing, target masking, fixed fitting budget, finite predictions, patient and
component leakage, repeated-patient loss aggregation and screen logic. Runtime
warnings concern sandbox CPU topology/cache queries; all fits and checks finish.
The combined regression run with previous blend/prefix tests passed 24 tests;
all 484 files bound by the final assessment were rehashed successfully.

Evidence: `artifacts/native_reconstruction_observation_skip_assessment_20261003.json`
and five `native_reconstruction_observation_skip_fold*_20261003.json` reports.
Private rows and candidate heads remain ignored local artifacts. No release
weights, forecasts, registry, intervals or promotion counts changed.
