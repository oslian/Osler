# Same-State Native-Anchored Nonlinear Readout

Fixed source-only experiment, all 47 targets and all five existing folds. The
parent teacher diagnostic found both unsuitable teacher corrections and student
direction errors. This experiment removes the scalar teacher objective and tests
whether a supervised correction around the native prediction preserves useful
existing structure better than replacing the prediction outright.

Keep the frozen epoch10 encoder, target/sibling masking, fit/selection identities,
scaling, bounded component weights and HGB parameters unchanged. Native output
is a deterministic readout of the same body64, not a new patient measurement.
No future information, outer queries, outcome-derived category or new modality.

Four new 65-column heads, each with the existing 100-iteration absolute-error HGB:

1. `anchored_residual`: body64 + native prediction, target Y-native; add native
   at inference. Signed corrections unrestricted, no clipping or hand-set gate.
2. `absolute_anchor`: identical inputs and capacity, target Y directly. This
   isolates residual parameterization from merely adding the native coordinate.
3. `anchor_only`: zero body64 + native prediction, target Y-native. This tests
   whether extra state improves over nonlinear calibration of the scalar anchor.
4. `shuffled_anchor`: within-fit-component cyclic shuffled body64, own native
   prediction and own Y-native retained. Evaluate own body at query. Same width,
   HGB settings and fit budget. Singletons stay unchanged and are reported. This
   is an incremental-state null conditional on the real native output, NOT a
   wholly nonphysiological null or exactly equal effective model capacity.

All fit labels are real measurements, not raw-teacher values. The encoder saw
fitting patients; native fitting predictions are in-sample, not out-of-fold.
Selection patients and their provenance components are excluded from ALL model
fitting. This does not solve upstream in-sample residual optimism by assumption.

Compare anchored_residual with body_probe, native, raw_probe, population,
absolute_anchor, anchor_only, shuffled_anchor and the completed observation_skip.
Predeclare 47*8 comparisons. Patient-mean losses average eligible folds first;
then assess patient-equal and component-equal differences. Require >=30 people,
>=3 components, mean<0, paired bootstrap95% upper<0 and one-sided t p<.05/(47*8)
at BOTH levels, exactly the previous screen pattern with the explicit family.
All contrasts reported, no target dropping. Also report unadjusted descriptive
means, fitting losses and component mean harms. No source outcome selects model
settings, checkpoint, correction strength or a winning subset for deployment.

Prior linear Huber/Ridge calibration on separate calibration roles, September20
anchored linear readouts, and native/body output blending were negative and are
NOT untested. This is a fixed nonlinear residual-vs-absolute parameterization
contrast on the current full source cohort, not a new claim that residuals in
general are a solution. Direct body HGB on truth is already a strong comparator.

Reproduce all original reference rows and metrics; bind scripts, protocol,
source artifacts, model and prediction hashes. Save patient rows only in the
existing ignored private directory. No runtime, release, interval, causal claim
or promotion changes. Source results are reused development, not a fresh test.
