# Fixed Teacher Readability of the Same Shared State

Question: can the frozen body64 reproduce the useful prediction function of
the original masked-raw265 head if its readout learns that function explicitly?
Observation skip improved body-only reconstruction but failed to establish
state value over raw-only. This test changes the readout learning target, not
the encoder, input, population, architecture or branch routing.

This is NOT the September3 unmasked latent/forecast retention distillation.
The teacher here is the frozen October1 raw HGB for the exact target-block-masked
CURRENT task, trained only within each original fit partition. It never sees
selection labels in training. Its fit predictions are in-sample teacher outputs,
not cross-fitted labels or independent physiology truth. The test therefore
cannot by itself prove optimal information retention or irrecoverable loss.

## Frozen Fit

All47 targets/all5 source folds. Original fit/selection IDs, scalers, raw265 and
body64 caches, fixed HGB absolute-error parameters100 iterations and bounded
component weights. No new tuning, ensemble, outer queries or gating changes.

Three student readouts receive the SAME fit-teacher prediction as their target:
1. body64; 2. process-only265 (numeric first106 set to zero); 3. body64 cyclically
reassigned within fit component (seed20261003+target index). At selection all
students receive their own allowed inputs. Third is same-dimensional readout
null; second is a larger process probe, not equal effective capacity. Encoder
remains real-trained. A fit-weighted median teacher prediction is an imitation
constant. Every learned inference path excludes the raw numeric branch.

Reference predictions: native, original body100, original raw100 teacher and
original population. Verify all saved reference predictions and query roles.
Fit/selection patient and component overlap must fail. All47 are retained.
Also record student fit imitation MAE so fit limitations and selection transport
can be distinguished without claiming fit/selection case mix is identical.

## Two Separate Outcomes

1. Against REAL truth: body student minus native/body100/raw teacher/population/
process student/shuffled student. Six paired comparisons; each descriptive
screen requires >=30patients,>=3components, negative means, bootstrap95% upper<0
and one-sided paired t-test p<.05/(47*6), at patient and component levels.
Full six-comparison conjunction is reported without selecting favorable cells.
2. Against TEACHER predictions: body student minus body100/process student/
shuffled student/fit teacher median. Same support and CI requirements, with
p<.05/(47*4). This is readability, not physiology accuracy or promotion.

In BOTH outcomes, average absolute normalized losses within each patient over
eligible folds before group averaging.5000 paired bootstrap replicates, no
model-refit bootstrap; overlapping training remains a limitation. No prediction
averaging, truth-based filtering, improvement claims from in-sample loss alone,
or threshold tuning after results. Constant predictions/unsupported groups
remain explicit failures, not dropped rows.

A positive imitation result shows this function can be read to that degree by
the tested decoder, not that all raw information is retained. A negative result
does not distinguish information loss from decoder/optimization limitations.
Only real-truth improvements support a repair; no new external-validation,
future-forecast, uncertainty, clinical or runtime-promotion claim is permitted.

All private patient outputs and fitted students remain ignored local artifacts.
Source development has been used repeatedly; source success would still require
uninspected final validation. Existing serving weights and registry stay intact.
