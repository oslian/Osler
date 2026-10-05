# Root-cause evidence, not a single-cause claim

## Earliest failure and scope

Latest distinct native glucose task independently localizes the same pattern:
`GLUCOSE_NATIVE_FIT_GAP_FINDINGS_20260911.md`. Whole beats own on fit6/7,
but is numerically worse on all7 new-source splits (significant2/7).
Fit whole/own MAE32.2889/34.0378 becomes evaluation34.8408/33.9762.
Treatment-number vs support increment is not stable even on fit. This narrows
where generalization first fails, not a proof of its unique physiological cause.

Train-matched qualification of the latest sensitivity test:
`RATE_ANCHOR_JEPA_SUPPORT_FINDINGS_20260911.md`. Support-only beats own5/7
source but0/7hospital; whole beats support1/7source,2/7hospital all. Thus some
source improvement does not require other lab numbers. Frozen numerical
dependence must not be conflated with stable incremental numerical utility.
Observation support can encode both severity and workflow; this contrast does
not identify which, and does not justify discarding truthful masks.

Latest qualification: native rate-query potassium JEPA now earns source
incremental whole-versus-own gain5/7, but hospital gain1/7. It is no longer
accurate to say every latest shared model fails on new source patients.
`RATE_JEPA_COMPONENT_FINDINGS_20260911.md` finds numerical neutralization
harms5/7source and3/7hospital; exact-age changes are small. This localizes
beneficial numerical dependence, not a unique cause of transport instability.
Hospital support is193patients across15sites, only one>=30; mean gain reverses
after deleting one hospital in2/7fits. Neither zero signal nor universal
transport can be inferred. Earlier glucose failure below remains a separate
task and must not be silently overwritten by this potassium result.

The failing claim is incremental whole-context future prediction, not all
forecasting or shared current-state recovery. In the original matched glucose
trial, whole beats own on fitting patients6/7 but is worse on new source-tune
patients7/7 (significantly6/7). This happens inside eICU before MIMIC transfer.
Thus database harmonization alone cannot explain the failure.

## Controlled contributors already located

| Contributor | Discriminating evidence | Remaining boundary |
|---|---|---|
| Direct decoder amplitude cap | tanh limited glucose change to about30mg/dL; identity plus retraining reduced own MAE, subsequently confirmed7/7 | Partial repair, not whole-context gain |
| Source-unseen numerical input | MIMIC ionized calcium entered a source-unseen channel around z=-15; fixed-weight source-support masking improved3/3 without changing eICU predictions | Does not establish old scaler provenance; no guessed units |
| Availability/history semantics | Results entered after old anchors; history mask omission and repeated-event sensitivity reproduced | Correctness repairs do not themselves prove accuracy gain |
| Extra-context generalization | Fitting gain reverses on new source patients; larger source cohorts and training budget help partly | Overfitting pattern is located, unique statistical/physiological cause is not |

## New discriminating result

A raw nonlinear tree comparator bypasses JEPA entirely, uses the same legal
history and source patient partitions, and recovers some potassium source
incremental gain3/7. Its selected configurations were frozen and reproduced
exactly before transfer. On207 held-out hospital and859 MIMIC patients,
full-vs-own significant gain is0/7 in each, while both beat persistence7/7.
See `POTASSIUM_TREE_TRANSFER_FINDINGS_20260911.md`.

This rejects blaming *only* JEPA compression or a disconnected neural path.
The common failure is the small, heterogeneous incremental correction from
other lab histories, not an inability to forecast potassium from its own
history. Different patient states and observation support can yield opposing
corrections; the tests establish instability, not its unique causal origin.

## What has not been proved

- Failure across a finite set of readouts does not prove no information exists.
- Lack of CI significance does not establish equivalence or zero benefit.
- Reused MIMIC and overlapping seeds are not independent confirmatory trials.
- Glucose-conditioned anchors and a lab-only history are not the entire patient
  state. The matched experiment cannot rule out properly available treatment,
  bedside, or waveform information, nor characterize all other targets.
- A warm-started encoder's original pretraining patient overlap is not excluded.

## Repair implication

The infusion comparator has an established rare-action sensitivity limit:
`TREATMENT_LEAF_SUPPORT_FINDINGS_20260911.md`. Potassium fit support8-19 is below
minleaf30 in all seeds; two evaluation roles havezeroexposed queries. Perfect
synthetic exposure signal is also blocked. Therefore no potassium splits cannot
be taken as evidence of no information. Pooled no-gain remains valid, while
potassium-specific utility is under-supported. This does not explain all
whole-context or order-model failures; deployment gates remain unchanged.

Order issuance is now a completed accuracy contrast:
`POTASSIUM_ORDER_CONTEXT_FINDINGS_20260911.md`.28source fits, original/placebo
gain0/7 own/full. Final cancellation and dosage/start/stop excluded; orders
that were eventually canceled retained. This prevents one known hindsight
filter, not a guarantee of independently verified availability. Simple order
count/recency is not a demonstrated fix; richer execution semantics untested.

Medication source coverage is larger than infusion, but semantically distinct:
`POTASSIUM_MEDICATION_COVERAGE_FINDINGS_20260911.md`. Alias repair recovers140
orders/12additional windows. Infusion support unchanged: not the cause of the
prior negative fit. Order windows739 versus infusion20 indicate source-scope
omission, not proven delivered-dose knowledge. Future cancellation and order
availability cannot be assumed known at anchor. No forecast improvement proved.

Treatment snapshot append is now tested, not pending:
`POTASSIUM_TREATMENT_CONTEXT_FINDINGS_20260911.md`.28 source-only refits,
own gain1/7/harm1/7, full gain0/7, both versus placebo0/7. Three full trees
ignore treatment entirely, four use few insulin splits. No potassium-treatment
split. This narrows the evidence to sparse snapshot uptake under the fixed
readout, not complete treatment information or a unique root cause. Full
delivered-dose/availability semantics remain unestablished; no runtime change.

The apparent zero potassium/bicarbonate infusion-rate support is now traced
to dimensional exclusion, not numeric parsing failure. Those records specify
solution mL/h, not delivered drug mEq/h. A separate solution-flow channel
retains20/36 query windows while keeping drug rate unknown. This fixes an
information-representation omission in the diagnostic, not the prediction
failure; concentration and independent availability remain unestablished.

The treatment-input gap is now quantified in
`POTASSIUM_TREATMENT_SUPPORT_FINDINGS_20260911.md`. Recent lab-only contrasts
omit treatments; exact-query insulin recognized rates cover497/6174windows,
potassium event evidence20/6174 but rate0 under current parsing. Infusionoffset
has no independent availability timestamp in the local table. Existing helper
action scopes are not complete potassium coverage. These facts constrain the
next safe comparison; they do not prove treatments explain the residual error.

Uniform amplitude repair has now been tested without the previous [0,1] cap:
`POTASSIUM_EVENT_GAIN_FINDINGS_20260911.md`. Source-tune patient-equal LAD
selects full-history gain>1 in6/7; evaluation improvement over original0/7.
Thus retrospective large-change attenuation is not sufficient evidence of a
global scale defect. Conditional errors remain unresolved; no runtime change.

Frozen source change decomposition now separates the error pattern:
`POTASSIUM_EVENT_CHANGE_FINDINGS_20260911.md`. Large actual changes have
direction accuracy .906 but predicted absolute movement .401 versus1.090;
full-own benefit0/7 there,5/7 in smaller changes. Thus one uniform amplitude
increase or a looser persistence switch is not justified. Future-defined
groups do not prove advance recognition. A floating-point grouping defect was
fixed with pooled predictions unchanged; this is audit correctness, not an
accuracy repair. Conditional extreme selection can itself expose attenuation,
so these findings do not identify one unique underlying mechanism.

Fixed-grid history also omits some within-window extrema. Adding known-event
mean/extrema/slope/change summaries on unchanged source pairs improves the
original full model1/7, not stably; see `POTASSIUM_EVENT_WINDOWS_FINDINGS_20260911.md`.
This establishes a representation loss, not its primacy as a forecast cause.
Do not repeat a summary-only repair as if untested; exact event encoders are a
different hypothesis and remain unproved by this comparison.

The potassium-native availability-anchor hypothesis has now been tested:
`POTASSIUM_NATIVE_ANCHOR_FINDINGS_20260911.md`. Same patient roles and frozen
tree capacities,21refits. Full-vs-own external0/7both; native full-vs-old full
on identical native rows0/7gain in every role. Query support changes and some
patients lose labels; this does not isolate timing alone. Do not repeat
glucose-conditioned query removal as an untested explanation of the entire gap.

Keep the proven amplitude and observation-contract fixes in compatible
candidates; never retrofit incompatible release weights/calibrators. Do not
remove the own-history control or relax significance to manufacture success.
Before another architecture sweep, the next distinct causal diagnosis would
need a source-only intervention on one unresolved contract, such as complete
availability-safe pre-anchor treatment context, while fixing the patients,
labels and controls where
possible. These are hypotheses, not guarantees. A final generalization claim
still needs untouched validation.

No runtime, interval or promotion changes. This document separates verified
contributors from the still unresolved explanation of incremental instability.
