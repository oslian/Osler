# Native-Anchored Nonlinear Readout: Complete Source Result

## Question and Scope

Does fitting a real-label residual around the native estimate fix the failure
of scalar teacher imitation, without replacing or enlarging the shared state?

The fixed protocol ran all 47 targets and all five source folds: 940 new HGB
fits, 1,797,884 selection occurrences, 720,470 unique patient-target pairs and
29,580 people. Each of four heads used 65 columns and the same 100-iteration
absolute-error budget. The encoder remained frozen. Target and sibling inputs
were masked; fitting and query patients/components were disjoint.

This is current-value masked reconstruction on reused source development data,
not future forecasting, an independent external replication or promotion.

## Results

Normalized target-macro MAE (lower is better; not physical-unit errors):

| Readout | Patient equal | Component equal |
|---|---:|---:|
| Original native | 0.882389 | 0.831593 |
| Body-only direct HGB | 0.857139 | 0.811241 |
| Native-anchored residual | 0.855898 | 0.810218 |
| Same inputs, absolute-value head | 0.856663 | 0.812692 |
| Native scalar calibration only | 0.868519 | 0.820933 |
| Shuffled body + own native anchor | 0.870431 | 0.822961 |
| Masked raw observations only | 0.825737 | 0.787947 |
| Body + masked raw observations | 0.828889 | 0.784964 |

Full prespecified eight-comparator screen: **0/47**. Comparisons require both
patient and component evidence, bootstrap upper bound below zero, and the fixed
47*8 multiplicity correction. No thresholds were changed after seeing results.

The residual beats native on eight targets: ALT, AST, BUN, creatinine, heart
rate, hematocrit, platelets and potassium. It beats scalar-only calibration on
eight, and the conditional shuffled-body control on eleven. These do not imply
a new ability: **zero targets pass against the existing body-only HGB, the
same-input absolute-value head, raw-only HGB or observation-skip comparator**.

The mean improvement over body-only is only 0.001241 patient / 0.001024 component
(about 0.14% / 0.13%). Patient mean improves for 36/47 targets, component mean
for 32/47. Conversely, it loses to raw-only on 46/47 patient means and 44/47
component means, and to observation-skip on every patient mean. Failure is not
solely an overly stringent significance screen.

Selected residual-minus-comparator normalized MAE differences:

| Target | Native, patient | Body HGB, patient | Raw HGB, patient | Raw HGB, component |
|---|---:|---:|---:|---:|
| BUN | -0.023547 | -0.001961 | +0.033805 | +0.025536 |
| Creatinine | -0.036740 | -0.002023 | +0.050693 | +0.040443 |
| Glucose | -0.020855 | +0.000053 | +0.012104 | +0.007693 |
| Hemoglobin | -0.032080 | -0.002148 | +0.017225 | +0.016001 |
| MAP | -0.006622 | -0.001171 | +0.005592 | +0.011099 |
| Troponin I | +0.031104 | -0.010993 | +0.071730 | +0.081752 |

## What This Locates, and What It Does Not

- Confirmed: real-label residual training removes the teacher-imitation objective
  mismatch by construction, but that alone does not close the observed gap.
- Confirmed: the same state contains information beyond its scalar native output
  for a subset. However, ordinary body-only nonlinear readout already captures
  enough of it that this residual parameterization adds no complete-screen win.
- Confirmed: raw-observation access remains substantially stronger in these
  matched source comparisons. Reparameterizing the readout is not sufficient.
- Not established: irreversible information loss in the encoder. Finite decoder
  capacity, fitted representation geometry, optimization and component shift
  remain possible contributors. A readout failure cannot identify only one.
- Not established: a reliable patient-state router. All retrospective errors
  remain evaluation outcomes, never prediction-time inputs.

Residual fitting beats absolute-head fitting on only 130/235 patient-equal,
113/235 component-equal and 116/235 bounded-weight losses. Thus this is not even
a universal training-loss improvement followed only by transfer failure. Native
fit outputs are in-sample for the encoder, not OOF. The shuffled control retains
the true native anchor; it tests incremental state, not an entirely nonbiological
null. Fold overlap and conditional bootstrap do not establish new independent
replications. These limitations are retained rather than attributed to biology.

## Verification and Decision

All original patient identities and baseline predictions reproduce exactly;
all-fold comparator metrics reproduce to 1e-12. The final assessment binds 485
files, including models and private predictions. All 16 native-anchor, teacher
geometry and shuffled-control tests pass. Training and scoring exited normally.
Patient rows and fitted diagnostic heads remain in ignored private storage.

Artifacts:

- `artifacts/native_reconstruction_native_anchor_assessment_20261003.json`
- `artifacts/native_reconstruction_native_anchor_fold{0..4}_20261003.json`
- `RECONSTRUCTION_NATIVE_ANCHOR_PROTOCOL_20261003.md`
- `scripts/train_native_anchor_probe.py`
- `scripts/score_native_anchor_probe.py`
- `tests/test_native_anchor_probe.py`

Keep candidate-only. No runtime, release checkpoint, interval or promotion change.
Do not repeat residual-versus-absolute parameterization as an untested fix, or
choose an outer-data winning subset from this report. The remaining legitimate
representation question is how to preserve useful masked numeric information
inside the shared state, with same-query raw and body controls; this result does
not authorize unreviewed retraining or relaxed validation.
