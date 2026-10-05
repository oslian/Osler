# Same-State Observation Skip: Fixed Source Protocol

Question: can one readout of the SAME frozen body state retain useful masked
numeric observations by receiving an explicit observation skip, rather than
blending two already highly correlated output predictions?

This is not the September25 calibration-role squared-loss linear residual
correction. Fit a joint absolute-error nonlinear readout on the original fit
patients, without residual targets, selection tuning, encoder updates or outer
evaluation. This is a diagnostic candidate, not a new parallel body model or
evidence that a shared physiological representation is already sufficient.

## Fixed Design

- All47 targets, all5 original source folds, original fit/selection roles,
  exact frozen epoch10 body64 and target/sibling-masked raw265 feature caches.
- Raw265 contains latest53, mean53, present53, availability lag53 and log count53.
  These summaries omit full event order; lag is NOT substituted for collection
  age. No new feature extraction, label tolerance or population is introduced.
- Candidate: concatenate body64 and raw265, width329.
- Process control: same329, own body plus raw with first106 numeric fields zero.
- Pairing control: same329, own body plus another fit patient's COMPLETE raw265
  summary, cyclically reassigned within fit component. No labels are shuffled;
  pairing seed20261003+target index. Singletons remain unchanged and are counted.
  At query time every arm receives its own legal summary. This is a trained
  pairing-null of the added branch, not a randomized full encoder or physiology-
  only proof; correlated body and mismatched raw can make the control harder.
- Original native/body-only/raw-only/population predictions remain references.
- All three new heads reuse EXACT original HGB hyperparameters, absolute-error
  objective, bounded component weights and100 iterations. No hyperparameter
  selection, prefix selection, test-driven routing or extra folds are permitted.
  Same329 inputs/tree budget are matched controls, not exact effective capacity
  equivalence with64/265-dimensional reference models.
- Check patient and component disjointness, feature/label/prediction hashes,
  target-masked zeros and baseline reproduction. Private rows/models stay local.

## Fixed Assessment

Six paired comparisons: candidate minus body-only, native, raw-only, population,
process-augmented and pairing-null-augmented heads. Average absolute normalized
loss within each patient across eligible original folds, THEN average equally
over patients or whole-history components. Do not average predictions across
folds, count repeated queries as independent people, drop unsupported targets,
select a winning fold or infer improvement from training loss.

Report all47, fold and component means,95% paired bootstrap CI (5000 replicates),
and the existing one-sided paired t-test. A descriptive incremental screen
requires >=30patients, >=3components and ALL six comparisons at BOTH levels
with mean<0, CI upper<0 and p<.05/(47*6). Also report each comparison separately
under the same correction so a failed conjunction is interpretable. Conditional
bootstrap does not refit overlapping trained models. No new lockbox or runtime
promotion is claimed, even if a descriptive screen passes.

Candidate vs body asks whether the added observations help; vs process/pairing
controls asks whether patient-matched numeric information helps beyond context;
vs raw-only asks whether shared state adds value to that observation channel.
Passing only the first contrast is NOT proof of recovered shared physiology.

No release weights, registry, intervals or future forecasts change. Results are
masked CURRENT reconstruction on repeatedly used source development. If this
fails, record the boundary; do not expand the method search using outer labels.
