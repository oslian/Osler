# Frozen-State Nonlinear Probe: Complete Source and Outer Results

## Question and Scope

Does the epoch-10 shared body state contain useful information that its native
readout does not recover? Does a fixed nonlinear learner recover more from the
same masked observations before their compression into that state?

This is CURRENT-value masked reconstruction, not future forecasting. All47
eligible targets, all five original folds, original actual-time histories,
target-and-sibling masking and fit-only normalization were retained. No encoder
weights changed. All three fixed probes were fitted on each fold's fit role
and evaluated without refitting, head selection or target selection. The outer
assessment was frozen after the complete source assessment and applied to the
entire target family, not just source-positive targets.

The source roles contain 1,797,884 query occurrences, 720,470 distinct patient-
target pairs and 29,580 people. Repeated queries are averaged as paired losses
within patient, not counted as independent people or combined into an ensemble.
Outer roles contain 1,800,214 pairs and 73,670 people; each patient and provenance
component belongs to exactly one outer fold. Training fits overlap. Fold2 uses
the certified fit0 encoder but its own outer queries. Two unsupported target-
fold combinations remain explicit missing support (myoglobin/fold0, pH/fold4).
These are reused development data, NOT a new untouched external lockbox.

## Fixed Comparison

| Arm | Input | Learner |
| --- | --- | --- |
| Native | Original shared state | Unchanged trained neural readout |
| Body probe | Frozen 64-dimensional body state | Fixed histogram gradient boosting |
| Raw probe | Masked latest/mean/presence/age/count summary, 265 columns | Same boosting settings |
| Process summary | Raw summary with latest/mean numeric values zeroed | Same boosting settings |
| Population | Existing fold-fit population reference | Unchanged |

Boosting uses absolute error, 100 iterations, learning rate0.1, 15 leaves,
min20 samples/leaf, L2=1 and seed19; no early stopping or parameter sweep.
The same bounded fit-component weighting is applied to all three probes.
Tree budgets match, but feature dimensions and effective capacity do NOT match
exactly. The native learner also differs in loss and optimization. Raw summaries
omit detailed sequence ordering. This is not an isolated causal test of
nonlinearity or proof that the encoder irreversibly destroyed information.

## Results

Equal-target macro normalized MAE, lower is better. Normalization uses each
fold's fit-only scale. Components are provenance-connected groups, not guaranteed
independent hospitals. These descriptive means are not promotion gates.

| Arm | Source patient | Source component | Outer patient | Outer component |
| --- | ---: | ---: | ---: | ---: |
| Native |0.882389|0.831593|0.917355|0.894102|
| Body probe |0.857139|0.811241|0.891238|0.877110|
| Raw probe |0.825737|0.787947|0.862525|0.854088|
| Process summary |0.928558|0.885266|0.964697|0.948643|
| Population |0.941965|0.893289|0.978512|0.955007|

The frozen-body probe reduces outer patient/component macro error by
2.85%/1.90%. The raw probe reduces it by5.98%/4.48%. Both improve the native
macro in every outer fold. Folds are not independent trials.

The full eight-contrast,47-target diagnostic family is retained. Each screen
requires negative patient and component mean differences, support>=30people
and>=3components, and one-sided paired-unit t-test p<0.05/(8*47) at both levels.
The reports ALSO contain paired bootstrap95% intervals (5,000 resamples).
The p-values are not bootstrap tail probabilities. Screens are descriptive,
conditional on these fitted models and reused development data, not a substitute
for the full runtime validation contract.

| Outer comparison | Targets passing both levels under8x47 screen |
| --- | ---: |
| Body probe versus native |7|
| Raw probe versus native |16|
| Process summary versus native |0|
| Raw probe versus body probe |2|
| Body probe versus process summary |19|
| Raw probe versus process summary |27|
| Body probe versus population |23|
| Raw probe versus population |26|

All seven body-probe incremental targets also beat both new controls:
**ALT, AST, BUN, calcium, heart rate, INR and potassium**. Four also passed the
source incremental screen: ALT, AST, BUN and potassium. This overlap is not
independent replication; it must not be described as seven new forecast cells.

Fifteen raw-probe targets beat native AND both new controls: albumin,
alkaline phosphatase, ALT, AST, BUN, calcium, creatinine, heart rate, hemoglobin,
lactate, platelets, potassium, serum osmolality, total protein and WBC.
ESR beats native but not both controls. Raw beats body under the strict outer
screen only for bilirubin and lactate; neither passed that same source screen.
Thus raw's average advantage is not proof of a generally superior replacement
for shared state or a stable target-by-target encoder bottleneck.

## Counterexamples and Sensitivity

The body probe improves both outer mean errors in32/47targets, but worsens at
least one mean in15: amylase, BNP, cortisol, ferritin, lactate, lipase, magnesium,
myoglobin, O2sat, pH, serum osmolality, triglycerides, troponin I, TSH, uric acid.
Raw improves both in43/47; the other four are amylase, BNP, myoglobin and TSH.
Mean differences are not individually proven harms.

ALT and AST contribute substantially to macro gain. A post-assessment descriptive
sensitivity excluding BOTH (not a new metric or gate) still gives body improvement
of1.23%patient/0.88%component and raw improvement of4.22%/3.31%. Removing any
single target retains a negative mean delta for both probes and both weighting
levels. Benefits are not exclusively the two liver targets, but are smaller
without them.

Even incrementally screen-positive body targets have supported component mean
harms: ALT2, AST1, BUN2, calcium3, heart rate2, INR4, potassium0, counting only
components with>=30patients for this diagnostic. ALL components remain in the
actual scoring. This threshold labels support, never removes difficult groups.
Body source/outer mean-change signs disagree for7patient and9component targets;
raw for2 and5. These observations reject uniform replacement and perfect
domain transfer even where overall screens pass.

## Diagnosis and Decision

**Confirmed:** some useful predictive information is already present in the
frozen state but its original readout does not use it as effectively as this
fixed alternative. The evidence contradicts the blanket explanation that no
usable signal is available. This is a demonstrated partial readout repair for
current reconstruction, not a repair of all generalization or future dynamics.

**Supported but unresolved:** raw observations can support a stronger average
probe under the tested procedure. Information compression, representation
geometry, loss/optimization, and differing effective capacity remain competing
explanations. It is not justified to announce the encoder as the unique cause.

**Still failing:** sparse/heterogeneous targets and some provenance components
remain worse. New source/outer overlap is limited. The new body probe has not
passed a same-input shuffled-person body-probe control, its own interval
calibration, or untouched external validation. The process summary is a control
for the selected observation features, not proof of no workflow confounding.

No release weights, runtime registry, intervals, forecast decisions or promotion
were changed. The raw probe remains diagnostic; it is not a second deployed
body model or an unvalidated bypass around the shared-state architecture.

Before a deployment claim, the next discriminating step is to keep these exact
fitted readouts and test their person-specific assignment against a properly
matched shuffled-state control, using source-only design and an unchanged full
target family. Existing output-permutation diagnostics of the NATIVE readout do
not establish that the NEW nonlinear body probe uses the right person's state.
Do not select new targets, weights or hyperparameters from the outer results.
Any actual promotion then still requires the complete validation/calibration
contract and new untouched external evidence. New experiments must check the
index first; generic clustering, global-offset and affine failures are retained.

## Verification and Reproduction

All cache grids, role disjointness, parent/checkpoint/model hashes, masked-input
contracts and prediction identities were verified. Native residuals and original
outer normalized MAE reproduce to1e-10 tolerance. All five outer inference jobs
and scoring completed. Seventeen focused tests pass, including corruption,
missing identities, role/provenance changes, source-only fit, patient-equalized
losses, full-family retention and descriptive-sensitivity accounting.

Commands (completed artifacts refuse overwrite):

```bash
python -m scripts.cache_nonlinear_probe --fold N --role fit
python -m scripts.cache_nonlinear_probe --fold N --role selection
python -m scripts.train_nonlinear_probe --fold N
python -m scripts.score_nonlinear_probe
python -m scripts.infer_nonlinear_outer_probe --fold N
python -m scripts.score_nonlinear_outer_probe
python -m scripts.summarize_nonlinear_probe
python -m unittest tests.test_nonlinear_probe_cache tests.test_nonlinear_probe_training tests.test_nonlinear_probe_scoring tests.test_nonlinear_outer_probe tests.test_nonlinear_outer_scoring tests.test_nonlinear_probe_summary
```

Aggregate reports:
- `artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json`
- `artifacts/native_reconstruction_nonlinear_outer_assessment_20261002.json`
- `artifacts/native_reconstruction_nonlinear_robustness_20261003.json`
- Ten cache proofs, five training reports and five frozen outer inference proofs.

Private feature caches, patient predictions and diagnostic fitted heads stay
local and ignored. No patient rows or identifiers are committed.
