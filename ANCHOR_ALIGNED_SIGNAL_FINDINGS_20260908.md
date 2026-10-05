# Anchor-Aligned Signal: Historical Result and Correction

Interpretation corrected on 2026-09-08. Original numerical report is retained
unchanged in `artifacts/anchor_aligned_signal_20260908.json`; original code and
prose are available at commit `aef4180`. This is superseded by
`ANCHOR_ALIGNED_SIGNAL_CORRECTED_FINDINGS_20260908.md`.

## What actually ran

Common canonical extraction rules, 53 variables, regular 3h anchors, current
measurement age <=6h, episode <=96h, no treatment inputs. Nominal creatinine@6h
accepted actual future events 2-10h after anchor. This is general ICU sampling,
NOT a one-factor matched AKI-onset comparison. New readouts trained on eICU;
MIMIC used for already-inspected validation. Pretrained-encoder patient exclusion
was not independently established.

Raw cross-history placebo preserved own target history AND non-target current
values/masks/ages, replacing ONLY lagged deltas/masks. JEPA placebo instead
replaced the entire 311-feature bundle, which included raw current measurements
and base forecast. These are different attribution questions, not equivalent
tests of whether the whole body state carries patient-specific information.

## Preserved numerical result

MAE in mg/dL, one representative anchor per evaluation patient:

| Evaluation | Persistence | Own history | Raw cross-history | Temporal placebo | Release bundle | Bundle placebo |
|---|---:|---:|---:|---:|---:|---:|
| eICU patient (510) | 0.2610 | 0.2543 | 0.2463 | 0.2470 | 0.2484 | 0.2535 |
| eICU hospital (426) | 0.2666 | 0.2584 | 0.2551 | 0.2554 | 0.2553 | 0.2573 |
| MIMIC (2000) | 0.1604 | 0.1767 | 0.1671 | 0.1659 | 0.1673 | 0.1675 |

Source fitting: 1230 patients. No candidate passed the reported full diagnostic.
Seven resampled subsets were evaluated, not seven independent training seeds.

## Withdrawn claims

The earlier claims that anchor mismatch was ruled out, that all added body state
only acted as population correction, and that no re-encoding could help were too
strong and are withdrawn. Raw current-state signal was not isolated; timing and
cohort effects were not isolated; the JEPA arm was not latent-only. Failure of
these finite fitted readouts cannot establish information absence or identify a
unique root cause. A common schema does not remove domain/clinical-phase shift.

The corrected experiment separates current versus temporal evidence, isolates
the four learned latent blocks, and prespecifies primary actual 5-7h labels with
the original broad timing as sensitivity analysis. Runtime weights, registries,
calibrators and serving behavior remain unchanged. No causal claims are allowed.
