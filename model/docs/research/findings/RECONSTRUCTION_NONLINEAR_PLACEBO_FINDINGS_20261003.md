# Matched Nonlinear State-Readout Control

## Completed Experiment

All235 planned target-fold controls completed. Each uses the same64-dimensional
body input, original real-trained frozen encoder, HGB hyperparameters,100
iterations, fit population and bounded component weights as the real probe.
Only fit-state/person pairing is changed: within each provenance component,
stable patient identities define a seeded random ordering, then a cyclic shift
reassigns whole states. Non-singleton groups have no identity matches.94 of
3,521,860 fit-pair occurrences are singleton groups that cannot be reassigned;
they stay present and are explicitly reported. These counts are not independent
people, because training populations and target observations recur.

Query states remain correctly assigned for both fitted heads. Query truth
never enters fitting. Source and outer grids, all original numeric predictions,
role/component disjointness within fold, checkpoint and cache hashes reproduce.
The encoder remains real-trained: this is a matched READOUT learning control,
not a fully shuffled encoder or physiology-only null.

## Results

Same data as the previous fixed nonlinear experiment: source1,797,884 query
occurrences/29,580unique people; outer1,800,214pairs/73,670people. Source paired
losses are averaged per person across eligible folds. Outer people/components
each appear in one fold. No tuning after outer outcomes; no new external lockbox.

| Comparison | Source | Outer |
| --- | ---: | ---: |
| Real body probe patient macro normalized MAE |0.857139|0.891238|
| Matched shuffled-training probe patient macro |0.950063|0.983810|
| Real body probe component macro normalized MAE |0.811241|0.877110|
| Matched shuffled-training probe component macro |0.904068|0.965183|
| Targets passing real versus matched shuffle |29/47|26/47|
| Prior incremental+process+population screens AND new shuffle screen |4/47|7/47|

The new comparison uses the predeclared47-target screen: negative paired mean
differences and one-sided t-test p<.05/47 at BOTH patient and component levels,
at least30patients and3components. Paired bootstrap95% intervals use5000
resamples. The prior eight-contrast47-target screens remain unchanged; no
threshold is relaxed. Conjunctions are descriptive, not new runtime gates.

Twenty-four targets pass the new shuffle comparison in both source and outer:
albumin, alkaline phosphatase, ALT, AST, BUN, calcium, chloride, CPK,
creatinine, fibrinogen, glucose, heart rate, hematocrit, hemoglobin, INR,
magnesium, phosphate, platelets, potassium, PTT, respiratory rate,
serum osmolality, total protein and WBC.

All seven previous OUTER incremental/control positives survive this matched
readout control: **ALT, AST, BUN, calcium, heart rate, INR and potassium**.
Outer component paired differences versus shuffle (normalized error):

| Target | Real minus shuffle | Paired95% bootstrap interval |
| --- | ---: | --- |
| ALT |-0.46130|[-0.56439,-0.35840]|
| AST |-0.52623|[-0.65017,-0.40718]|
| BUN |-0.13513|[-0.16701,-0.11017]|
| Calcium |-0.14908|[-0.18403,-0.11528]|
| Heart rate |-0.05672|[-0.07566,-0.03735]|
| INR |-0.11766|[-0.15332,-0.08668]|
| Potassium |-0.06119|[-0.07813,-0.04747]|

These intervals describe paired ERROR DIFFERENCES, not individual patient
prediction intervals. Nominal95% intervals do not replace family-corrected
screens. The native versus real-body improvement remains the previously
reported2.85%patient/1.90%component, not the larger real-versus-shuffle margin.

## Where It Still Fails

Twenty-one outer targets do NOT pass the new paired screen: amylase, anion gap,
bicarbonate, bilirubin, direct bilirubin, BNP, cortisol, ESR, ferritin, freeT4,
lactate, lipase, MAP, myoglobin, O2sat, pH, sodium, temperature, triglycerides,
troponin I and TSH. Of these, ferritin, myoglobin and O2sat have worse mean
errors at at least one weighting level. The other18 have lower means but fail
the statistical/support criterion; this is not proof of no information.

Source full conjunction positives are ALT/AST/hematocrit/phosphate. Outer full
conjunction positives are the seven above. **Only ALT and AST pass the entire
descriptive conjunction in both roles.** No promotion is made from the more
favorable outer list. The seven-target outer result must not be presented as
seven fully validated runtime capabilities.

The source blockers for the other five outer-positive targets are already
visible BEFORE their outer result:

| Target | Source screen(s) failing, all on component comparison |
| --- | --- |
| BUN | Versus process-summary control |
| Calcium | Incremental versus native |
| Heart rate | Versus native and process-summary control |
| INR | Versus native, process-summary and population |
| Potassium | Versus process-summary and population |

All these source mean differences favor the real body probe; family-corrected
component evidence is insufficient. Examples: source BUN process p=.002699,
calcium native p=.001081, potassium population p=.001337, versus the fixed
8x47 threshold approximately.000133. Source BUN/K/INR have12distinct components,
versus36-38outer components; source and outer are not equivalently powered or
identically distributed. This suggests uncertainty/domain heterogeneity rather
than a simple absence of individual signal, but DOES NOT isolate power as the
only explanation. Groups and fits are dependent across reuse. Existing supported
component mean harms from the prior report remain, not erased by better means.

## Interpretation and Boundaries

The new test strengthens a concrete finding: correctly paired patient states
help this readout; its gain is not fully reproduced by the same-dimensional
head trained on within-component scrambled pairings. Together with the prior
fixed-body readout improvement, this supports a partial readout repair, not a
claim that the shared state learned nothing.

It does not prove the representation contains only physiology. Workflow and
severity variation within components remain possible signals. It also does not
prove an irreversible encoder bottleneck, stable future dynamics, individualized
causal mechanisms, or reliable deployment. This is one fixed shuffle realization
with a real-trained encoder, and all data are repeatedly used development data.

The first remaining blocker for many promising targets is source component-level
consistency of incremental numeric benefit, not the new matched placebo itself.
Do not solve it by selecting the seven outer targets or weakening thresholds.
A next source-only diagnosis should distinguish a few genuinely harmful
components from uncertain benefit across the12selection components, using the
already saved paired losses. Prior global offset/affine corrections did not
transfer and must not be repeated under a different name. Any model selection,
calibration or promotion requires its own fixed contract and untouched final
evidence. Current release, intervals, weights and serving are unchanged.

## Verification

All five control fits and source/outer scoring completed, without interrupting
training. Twenty-four focused tests pass, including seven new tests for stable
within-component derangement, preserved capacity, singleton accounting,
fit/query role separation, original-prediction parity and per-patient weighting.
No patient rows, IDs, feature caches or fitted diagnostic heads enter Git.

Reproduce with completed-artifact overwrite protection:

```bash
python -m scripts.train_nonlinear_placebo --fold N
python -m scripts.score_nonlinear_placebo
python -m unittest tests.test_nonlinear_placebo tests.test_nonlinear_placebo_scoring
```

Artifacts: five `native_reconstruction_nonlinear_placebo_foldN_20261003.json`
proofs and `artifacts/native_reconstruction_nonlinear_placebo_assessment_20261003.json`.
Protocol: `RECONSTRUCTION_NONLINEAR_PLACEBO_PROTOCOL_20261003.md`.
