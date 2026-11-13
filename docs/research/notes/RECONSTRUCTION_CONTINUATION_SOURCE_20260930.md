# Source Failure Location And Monitor Fidelity

## Scope

All 47 targets and five existing folds are retained. Read full epoch-8 source
selection predictions, fixed source monitors at epochs 8/10, and complete
outer epoch-8/10 reports. Validate hashes, source-only patient roles, unique
patient-target keys, finite values, and exact reproduction of prior full-source
patient/component mean errors. No model, selector, threshold or cohort changes.

Full source epoch-10 predictions do not yet exist. The epoch-10 source monitor
contains at most 32 patients per target/component. Its result must not be
mislabelled as a full-source assessment. Source and outer components differ;
this compares patterns, not the same people moving between data roles.

## Results

Of the 26 targets failing the completed outer base screen:

- 18 already have a source selection component with at least 30 patients whose
  mean error exceeds the fixed population predictor at epoch 8.
- 17 have such a source component versus the process control.
- BNP, O2sat and sodium have at least one supported population-harmed source
  component in every fold. This is not five independent replications.
- 17 of the 21 targets with recurring epoch-8/10 outer population mean harm
  also have the source epoch-8 population-harm pattern. Amylase, anion gap, ESR
  and ferritin have the outer pattern without this supported source pattern.

These are component mean signs, not statistically established subgroup harms.
The same target can have favorable overall source means while individual
source components lose. Thirteen failed targets have favorable equal-component
means against both controls in ALL source folds: amylase, bicarbonate, direct
bilirubin, CPK, ESR, hemoglobin, lactate, LDH, magnesium, MAP, pH, respiratory
rate and temperature. Pooled source gain therefore does not settle portability.

The small monitor is an imperfect diagnostic surrogate:

| Comparison | Patient weighting | Component weighting |
| --- | ---: | ---: |
| Full source8 vs sampled source8: sign disagreement vs process | 27/235 | 17/235 |
| Full source8 vs sampled source8: sign disagreement vs population | 18/235 | 15/235 |
| Monitor8-to10 vs outer8-to10: real error change sign disagreement | 84/233 | 103/233 |

The first two rows hold source population and epoch fixed and compare sampling
and weighting effects. The third ALSO changes source/outer population, so it
cannot attribute disagreements to monitor sampling alone. Differences can be
tiny and are not significance disagreements. Overlapping fits/queries make
these 235 nominal comparisons dependent; they are not independent experiments.
Two unsupported outer target-fold cells remain explicitly null, not mistakes,
successes, or disagreement counts.

## Failure Location And Next Repair Boundary

Supported mean harms are often visible before the outer stage. It would be
incorrect to call every failure purely an external-database mismatch or low
sample size. It is also incorrect to infer memorization, absent biological
information, or a unique optimization defect from these signs.

The monitor did NOT select the continuation epoch: epoch 10 was fixed in
advance. Monitor disagreement is therefore a limitation of diagnosis, not an
explanation for how these weights were selected or a discovered training bug.

Before another training change, obtain full source epoch-10 component results
under the unchanged query contract. Compare them with the existing epoch-8
source predictions; do not choose a new epoch or target route from the already
viewed outer outcomes. A source-only repair needs to improve new-patient and
component behavior, not merely its fitting loss or overall source mean.
Existing unsuccessful source corrections and weighting trials remain relevant;
this diagnostic does not establish that repeating them will fix the encoder.

## Artifacts

Run `python -m scripts.diagnose_continuation_source` on the bound local inputs.
Report: `artifacts/native_reconstruction_continuation_source_20260930.json`.
Five focused tests plus five margin tests pass. The first diagnostic draft
assumed every outer fold had target support; it stopped before emitting an
artifact. Explicit unsupported-fold handling and its regression test repaired
that diagnostic assumption, not a historical model failure.
No raw patient rows, weights, calibration, serving behavior or promotion changed.
