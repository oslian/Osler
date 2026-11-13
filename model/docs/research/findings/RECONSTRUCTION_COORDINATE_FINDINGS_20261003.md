# Lossless Coordinate Audit: Readout Bias Confirmed, PCA Repair Rejected

## Complete Experiment

All47 targets, all five source folds, 940 new fits completed. Same frozen body64,
same masked raw265, same actual labels, same patient/component roles, same HGB
absolute-error100 budget. Queries:1,797,884 occurrences,720,470 unique
patient-target pairs,29,580 people. No outer cohort queried or used to select.

Each representation was transformed by a fixed random orthogonal rotation or
the full fit-weighted covariance eigenbasis. No dimensions were discarded,
including null-variance directions. No whitening, feature rescaling, labels in
the transform, additional input or patient permutation. Fit and query underwent
the identical saved transformation. Largest scale-relative inverse error across
all fit/query arrays was7.8911e-13. This is not an inference-only perturbation
with an unchanged head: every transformed head was trained in those coordinates.

## Results

Normalized target-macro MAE, not physical-unit errors:

| Representation/readout | Patient equal | Component equal |
|---|---:|---:|
| Raw original coordinates |0.825737|0.787947|
| Raw full PCA coordinates |0.863901|0.823810|
| Raw fixed random rotation |0.916796|0.875915|
| Body original coordinates |0.857139|0.811241|
| Body full PCA coordinates |0.858206|0.816076|
| Body fixed random rotation |0.857428|0.813094|
| Native neural readout |0.882389|0.831593|

Raw random rotation increases macro error by11.03% patient /11.16% component.
Every one of47 patient-mean errors is worse;46/47 component means are worse.
Their conditional pointwise95% bootstrap intervals are entirely positive for
47patient and42component comparisons. These harm counts are descriptive with
pointwise intervals, NOT a newly introduced multiplicity-corrected harm gate.

Raw PCA harms45/47patient means and44/47component means. Body PCA harms43/47
patient means and32/47component means. Under the fixed47*9 correction, body
PCA improves over native for ALT/AST/glucose/platelets, but **0targets beat the
existing body HGB, raw HGB or random body rotation**. Full six-reference primary
repair screen: **0/47**. No candidate promoted.

Selected patient-equal normalized MAEs:

| Target | Raw | Raw rotation | Body | Body PCA |
|---|---:|---:|---:|---:|
| BUN |0.578286|0.688338|0.614053|0.619066|
| Creatinine |0.820478|0.951776|0.873193|0.882806|
| Glucose |0.689013|0.721485|0.701064|0.703806|
| Hemoglobin |0.425513|0.506971|0.444886|0.445917|
| MAP |0.550190|0.572574|0.556953|0.557974|

## What Changed in the Diagnosis

Previously, stronger raw HGB established a performance gap but did not isolate
why the body path lagged. The lossless intervention now establishes an actual
confound: **the same available information can have materially different HGB
accuracy solely under a different invertible coordinate system**. Raw's macro
advantage over body even reverses after its random rotation, without losing any
raw input information. Information retention and finite-head readability cannot
be equated.

This is consistent with HGB's feature-wise binning and tree partitions, not a
proof of a library bug. The model and its fixed optimization/capacity budget
interact with representation geometry. Documentation for the implementation:
https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting

However, the attempted unsupervised reorientation of body did NOT repair its
accuracy. Native body coordinates were already better than full PCA on most
targets. No justified claim that "a rotation is all the encoder needs" follows.
Nor does this prove all missing signal is present in body, all problems are in
the decoder, or that arbitrary larger heads/rotation seed search will help.

A future information-loss claim needs a decoder/representation diagnostic that
does not confuse accessibility with loss; this experiment supplies a lossless
positive control for that distinction. Any subsequent learned alignment must
remain fit-only and be compared with these unchanged same-query controls. Do not
count this negative PCA repair or the older PCA64+ridge as an untested method.

## Verification and Boundaries

16coordinate/native-anchor/patient-permutation tests passed, including inverse
reconstruction for rank-deficient inputs, distance preservation, query isolation,
fixed100iteration budgets, duplicate rejection and patient equalization. All
484assessment bindings verified. Exact original query/reference rows and allfold
baseline metrics reproduce to1e-12. Both training and assessment exited0.

Results remain reused source development, conditional on fitted models and
overlapping folds, with one fixed random rotation per variable. This is current
masked reconstruction, not future dynamics, uncertainty calibration, a new
external lockbox or a clinical/causal claim. Raw contains process features too;
preserving them does not prove physiology-only signal. Four new diagnostic heads
per target are not new independent whole-body models. Runtime, released neural
weights, intervals and registry remain unchanged; private rows/heads not committed.

Reproduce:
`python -m scripts.train_coordinate_probe`
`python -m scripts.score_coordinate_probe`

Evidence: `artifacts/native_reconstruction_coordinates_assessment_20261003.json`
and five `native_reconstruction_coordinates_foldN_20261003.json` files; fixed
`RECONSTRUCTION_COORDINATE_PROTOCOL_20261003.md`.
