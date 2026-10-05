# Observation Support Does Not Reliably Choose This Readout

## Complete Fixed Experiment

All47targets/five source-selection folds,1,797,884 query occurrences. Within
each original fold, a second component-held-out probe predicts body-minus-native
absolute error from pre-query observation support. It never fits on the held
component's outcomes. Existing encoder/head fitting components are disjoint
from all selector selection components; roles and provenance are checked.

Compare a training-weighted constant, four aggregate support features,106
variable-specific mask/measurement-age features, and a same-capacity shuffled
106-feature control. Features exclude numeric physiology and the masked target
and its algebraic siblings. Weighted scaling and fixed Ridgealpha100 use only
the selector's training components; choose body iff predicted difference<0.
This is not the earlier descriptive support correlation or a fitted value
correction. The earlier negative studies remain relevant, not superseded.

All runs terminated normally.1,288component-held-out fits have sufficient
training support;23 do not and remain explicitly unsupported.44targets have
complete support. Anion gap has17,376 unsupported query occurrences, myoglobin
785 and pH6,540. Unsupported decisions use native only for complete-population
bookkeeping, are labeled unavailable, and cannot pass the diagnostic screen.
Shuffled fitting preserves component feature distributions;228 singleton
occurrences remain unchanged across repeated fits. These are not228unique
independent patients. Original native/body/population metrics reproduce exactly.

## Result

Normalized target-macro MAE, all47targets including explicit unavailable rows:

| Readout/decision | Patient-equal | Component-equal |
| --- | ---: | ---: |
| Always native |0.882389|0.831593|
| Always nonlinear body |0.857139|0.811241|
| Training-mean constant choice |0.857247|0.812924|
| Aggregate support choice |0.857894|0.813644|
| Variable-specific support choice |0.863143|0.816275|
| Shuffled support choice |0.862673|0.817017|

The variable-specific choice is worse than always-body and the simple constant
choice on both means. It does not reliably beat the shuffled choice: patient
macro is worse, component macro slightly better. On the44fully supported
targets the same conclusion remains: body0.875882/0.831573 versus variable
choice0.881019/0.835840; constant0.874616/0.831972. The negative result is not
solely due to unsupported queries being assigned native.

Zero targets pass allfive fixed incremental source diagnostic contrasts.
Variable choice passes the multiplicity-adjusted patient+component screen
versus always-native for10targets, but for zero versus body, constant,
aggregate or shuffled. Beating an older reference is not proof that choosing
between readouts added value.

Variable choice has lower mean error than body on both levels for7targets;
than constant for6, aggregate for5, shuffled for22. These are descriptive mean
comparisons, not validated improvements or multiple independent discoveries.

## Previously Adverse Targets

Patient-equal normalized MAE:

| Target | Native | Body | Constant choice | Variable choice | Body selected by variable choice |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ferritin |0.827030|0.845038|0.827240|0.832870|27.80%|
| Troponin I |5.184358|5.226455|5.187334|5.194432|46.27%|
| O2sat |0.600807|0.602190|0.598562|0.600984|42.00%|
| BUN |0.635639|0.614053|0.614053|0.616903|80.90%|
| INR |0.893784|0.860433|0.860433|0.862795|84.26%|

Ferritin/troponin selection reduces some harm relative to always-body but still
loses native. BUN/INR constant choices correctly select body for every query;
the more detailed selector introduces switches that worsen their averages.
The selector is active, not held shut by a conservative positive-quantile rule.
Simply relaxing a switch cannot explain or repair this result.

The error-difference estimation task itself also fails to gain: among45targets
with any supported decisions, variable-specific regression improves held-out
expected-difference MSE over the constant on only1patient comparison(creatinine)
and2component comparisons(creatinine/ESR). This is descriptive, not a new gate
or proof that every possible conditional expectation estimator must fail.

## Diagnosis and Next Repair Boundary

Confirmed: the tested support representation and fixed decision learner do
not estimate transferable readout benefit well enough. A complete input schema
and more process features do not automatically create a useful patient-state
selector. Clinical measurement workflow may itself shift by component.

Not established: absence of physiological signal, impossibility of switching,
or a unique encoder defect. Numeric physiology was deliberately excluded, and
linear expected-error estimation is a specific restricted probe. Its failure
does not validate unrestricted numeric routing or justify trying a large search
on these already inspected outcomes. Prior conditional numeric correction and
generic selector failures remain part of the evidence.

Before another selector, the unresolved question is whether the fixed current
nonlinear body head already fits these targets poorly or instead obtains large
fit gains that disappear even on source-selection components. Comparing fit
versus selection errors for the SAME heads and controls would distinguish an
optimization/representation warning from an overfit/transport warning without
training another architecture. Training error is only a diagnostic, never a
generalization or promotion claim. Do not use outer data to tune it.

No model weights, intervals, runtime, gates or promotion were changed. No
new final lockbox was consumed; this is reused source development, current
masked reconstruction only, not future forecasting or causal inference.

## Verification and Files

Five new tests verify unknown ages/masking, held-component outcome independence,
a learnable synthetic decision, patient-fold averaging, retained unsupported
queries, and rejection of outer/duplicate/nonfinite inputs. With the prior
source-error/component tests,12tests pass. All original prediction hashes,
snapshot hash, role exclusivity and original aggregate means checked. Only
aggregate artifacts are committed; per-patient decision files remain local.

Reproduce: `python -m scripts.probe_nonlinear_support_selector`.

Protocol: `RECONSTRUCTION_SUPPORT_SELECTOR_PROTOCOL_20261003.md`.

Artifact: `artifacts/native_reconstruction_nonlinear_support_selector_20261003.json`.
