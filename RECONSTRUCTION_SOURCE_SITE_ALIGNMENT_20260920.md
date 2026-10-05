# Source Site Alignment

Source-only provenance and frozen epoch7/16 predictions audited. No evaluation
rows, model fitting or checkpoint selection. Prediction hashes, patient-target
identity, truth and encoder role checked. Ambiguous multi-site people are NOT
assigned to a representative hospital.

| Role | Patients | Multi-site excluded from site summaries | Sites among unambiguous patients | Largest site share | Effective sites |
|---|---:|---:|---:|---:|---:|
| Fit | 10201 | 35 | 11 | 27.59% | 5.73 |
| Encoder selection | 3323 | 10 | 4 | 50.44% | 2.55 |

Effective sites is inverse squared patient-share sum, not statistical degrees
of freedom. Fit counts are overall patient counts, not target-specific counts.
Current cached examples omit the originating stay/site for each target. For
45 multi-site source patients, patient identity alone cannot safely determine
sample hospital. This blocks complete site-balanced sampling until provenance
is reconstructed; it does not invalidate their physiological observations.

## Fixed Epoch Comparison

Among source encoder patients with unambiguous hospitals, changing patient
weights to equal-site weights reverses the numerical improvement sign for
only albumin, chloride and magnesium (3/33). No significance inference.

| Target | Patients | Pooled error change | Equal-site change | Improving sites |
|---|---:|---:|---:|---:|
| Creatinine | 3085 | +.001412 | +.003272 | 2/4 |
| Lipase | 452 | +.025186 | +.016376 | 2/4 |
| Glucose | 3117 | -.016662 | -.013762 | 3/4 |
| BUN | 3076 | -.010341 | -.012509 | 4/4 |

Changes are normalized MAE, negative better. Glucose and BUN retain negative
equal-site differences after removing any single source site. Creatinine and
lipase do not. This is descriptive sensitivity, not proof of true harm or
benefit; prior paired source intervals crossed zero for creatinine/lipase.

## Decision

Site concentration is real, but simple weighting does not explain the primary
creatinine/lipase tradeoff. Do not launch a broad reweighting sweep as if its
success were established. Source selection population is large in people but
limited in independent hospitals; adding people did not add site diversity.
Before a site-aware training repair, recover target-specific stay/site from
the deterministic anchor selection and raw provenance, without altering
existing examples or assigning ambiguous people arbitrarily. Retain all
39 target statuses, including six untrained heads.

Five combined source/site helper tests pass. Artifact:
artifacts/native_reconstruction_source_site_alignment_20260920.json.
No runtime, calibration, weights, registry or promotion changes.
