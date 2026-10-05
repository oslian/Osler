# WholeBody-JEPA Project State

2026-10-05 E44 COMPLETE/VERIFIED: fixed strong/weak tangent-direction ablation,
not a new fit or formal promotion gate. Original E43 lambda and alpha=.5 retained.
Strong-only MAE fit/selection/same-source .746151/.892978/.787198 versus zero
.753682/.876270/.781413. Strong directions retain88.65-96.37%parameter squared
norm; removing weak directions does not repair transfer. Strong linear surrogate
also fails query averages. Full rootcause and validated repair remain unresolved.
Fivefolds/20480queries independently replayed with error0; scalar3.553e-15.
Saved findings record39/30bindings and34tests. Numerical casting/Gram failures
and equivalent direct-SVD repair are disclosed, not claimed as model improvement.
Fold0/2identical; inherited support/exposure limitations; reused development.
Close fixed ablation, no cutoff search or query-selected per-target routing.
RECONSTRUCTION_TANGENT_SPECTRUM_FINDINGS_20261005.md and native_tangent_spectrum
aggregate/verification JSON supersede earlier pending-E44 notes. Documentation
handoff only reconciles completed evidence; no experiment rerun or runtime change.
Read ROOT_CAUSE_LEDGER.md before every task; graph.md records actual reading scope.

2026-10-05 E43 COMPLETE/VERIFIED: full-fit shared-bridge tangent solve followed
by the actual nonlinear recurrent model. All5folds retain2048 numerical fitting
directions; conditioned solve24-36iterations; half step is first accepted in all.
Actual patient/target-macro MAE fit .753682->.745849, selection .876270->.893038,
same-source .781413->.784977. Linear diagnostic at accepted step .407425/1.332393/
1.163262: local fit feasibility does not imply finite-step fidelity or transfer.
23/33 matched fit improvements lose mean advantage in each query role; not
statistical harm counts. All11screens/full0. 5models/20480replays exactly match,
scalar/paired max6.661e-15;45/33bindings;23unittests. Fold0/2identical.
Initial failed solver and reversible numerical amendment disclosed and bound.
Close fixed one-linearization branch; no parameter sweep or promotion. Current
reconstruction only, reused development, not future/self-learning or lockbox.
Runtime/weights/intervals unchanged; full rootcause and repair remain unresolved.
RECONSTRUCTION_TANGENT_BRIDGE_FINDINGS_20261005.md. Supersedes pending E43 notes.

2026-10-05 E42 COMPLETE: saved E40 readout reachability, no training.
53x64 matrices have six zero rows, all E41 unsupported/unscored. Nonzero47,
scored46 and supported27 subsets each full-row-rank in all saved folds.
Supported condition15.45–19.49, float32 right-inverse error <=3.5763e-7.
NumPy/torch agreement,15bindings,7unittests. Rules out hard affine output
dependency on supported targets; does NOT prove encoder informativeness,
reachable recurrent states, generalization repair or promotion. Fold0/2not
independent; zero heads are reconstruction-only, not release forecast claims.
No runtime/weights/gates/interval changes. Full repair remains unresolved.
RECONSTRUCTION_READOUT_REACHABILITY_FINDINGS_20261005.md.

Mandatory pre-task handoff: read `ROOT_CAUSE_LEDGER.md`, especially its one-page
handoff on old conclusions versus new evidence. The entries below are historical
snapshots; later completed evidence supersedes earlier pending notes. The latest
documentation task adds no experiment, efficacy result or runtime change.

2026-10-05 E41 COMPLETE: fixed E40 screen failure diagnosis, no training.
53 current reconstruction targets (not forecast cells):7unsupported,
19below minimum support,27supported; all27 fail at least one control under
patient/component mean direction before significance. Against raw:23fail
both-weighting conjunction,4mean gains failCI (HR,lactate,K,RR). Only4 individual
comparisons are multiplicity-only and their targets fail other controls earlier.
All460original screens reproduced;5120saved rows independently scalar-checked,
maximumroundoff3.916e-13;81bindings,11tests. p_less is t-test, not bootstrap-tail.
Fixes failure attribution, not model efficacy. No thresholds/runtime/promotion.
RECONSTRUCTION_SCREEN_DIAGNOSIS_FINDINGS_20261005.md; native_screen_failures JSON.

2026-10-05 E40 COMPLETE/VERIFIED: actual-step half-space intervention after
PCGrad+AdamW, same shared bridge/all5folds. Local cross-target adverse dots
1814raw->0applied on candidate trajectory;1709projections,441rounding-veto steps.
Finite cross-target increases489,all <=1.192e-7;112same-target harms not protected.
Fit/selection/same-source MAE .749089/.876494/.777282 versus PCGrad
.748895/.876528/.777131. Ten comparisons/fullscreen0. Only15/47fit target means
improve vsPCGrad;4selection/8same-source fit gains reverse. Not solely external
shift; local safety is insufficient for aggregate fitting/conditional transfer.
5saved models/20480queries replay0,scalar/paired3.918e-13,58/69bindings,28tests.
Fold0/2identical. Close fixed branch; no optimizer/projection parameter sweep.
No runtime/weights/interval/promotion. Full rootcause/repair still incomplete.
RECONSTRUCTION_ACTUAL_STEP_FINDINGS_20261005.md; native_actual_step artifacts.
Supersedes earlier notes that actual-step intervention is untested.

2026-10-05 E39 COMPLETE/VERIFIED (documentation handoff): original E38 ten
models/tensors/logs exactly reproduced;40960 training-example steps, not patients.
PCGrad1921 actual local harmful steps first become harmful at proposal43,
momentum1759,preconditioning119. Preconditioning also fixes749 and introduces123;
no new harmful classification from decay/roundoff on this trace. Not causal
fractions or a reason to remove an optimizer stage. Finite-loss reconstruction
exact; scalar roundoff2.711e-20;verification46986 exited0. Three locally safe
dots still have finite loss increases. This is diagnosis, not a new repair or
unseen-patient efficacy test. No runtime/weights/interval/promotion changes.
See ROOT_CAUSE_LEDGER.md section6 and native_adam_stages*20261005.json.
Historical E38 pending stage-accounting statements below are superseded here;
full generalization repair remains unresolved.

2026-10-05 E38 COMPLETE/VERIFIED: matched paired mean versus PCGrad intervention.
All5fold/10models/20480queries; saved-model replay maxerror0, independent scalar
and paired-statistic roundoff1.833e-13;69original/74verification bindings,24tests.
Proposed harmful gradient dots670->67; ACTUAL Adam dots2197->1921. Local surgery
works but actual update protection is incomplete. Selection MAE .876553->.876528,
same-source .777108->.777131; all9screens/full0. Versus mean,15fit gains reverse
in selection and15in same-source (descriptive). Fold0/2 checkpoints identical,
not five independent replicates. Close fixed branch, no hyperparameter sweep.
Optimizer-stage responsibilities and own-direction transfer remain unresolved.
RECONSTRUCTION_PAIR_GRADIENT_FINDINGS_20261005.md and verification artifact.
No runtime/weights/interval/promotion; full rootcause/repair not delivered.

Historical entry (superseded by completed E38 evidence above):
2026-10-05 DOCUMENTATION HANDOFF: E37 remains the latest completed evidence.
E38 paired mean/PCGrad protocol, trainer and tests exist; no completed test,
training, assessment or independent verification evidence is recorded yet.
This documentation task does not run E38 or change runtime/weights/intervals.
Read ROOT_CAUSE_LEDGER.md before every task; core conclusion unchanged.

2026-10-05 E37 COMPLETE/VERIFIED: actual Adam momentum-origin tracing.
All5fold original E34 immediate tensors/logs reproduce exactly,4096steps each.
All20480 E36 queries/53target origins plus decay/rounding;52bindings,15roles.
Selection18 harmful local directions split8 own-helpful/other-harmful and10
own-harmful;fit7=7+0,same-source11=9+2. Component13=13+0,20=8+12,17=14+3.
Creatinine/Hgb benefit from both origins: no blanket task isolation repair.
Vector/tangent sums max1.111e-16,zero parity0,scalar1.041e-17;25tests.
Fold0/2 origin vectors identical; not five independent replications.
Conditional joint-Adam decomposition is NOT leave-task-out causality or
finite-step repair. No runtime/weights/interval/promotion. Full repair still
unresolved. RECONSTRUCTION_ADAM_ORIGIN_FINDINGS_20261005.md and aggregate.

2026-10-05 E36 COMPLETE: saved immediate bridge PARAMETER path, not output chord.
All5fold/3roles/20480queries/53inventory. Selection22mean harms:18initial parameter
MAE derivatives harmful,4beneficial; BUN/temperature reverse sign versus chord.
Same-source15harms:11/4; fit8:7/1. Component selection25:20/5, same18:17/1.
Local direction failure remains even near zero for most targets; nonlinear path
is not convex, no claim every finite scale fails. Initial-benefit cases cannot
be generalized across roles (BUN same-source initial direction harmful).
Allfloat32 endpoints reproduce exactly; double maxdrift1.264e-6;670fixed FD
checks maxerror1.036e-8. Allscalar/fsum max1.777e-15;21tests,42bindings and
alloriginal columns/support checked. Diagnostic time-dtype cast bug repaired,
NOToriginal model rootcause. No training/runtime/weights/interval/promotion.
RECONSTRUCTION_PARAMETER_PATH_FINDINGS_20261005.md; native_parameter_path JSON.
Conditional/input/training causes of direction transfer still unresolved.

2026-10-05 E35 COMPLETE: frozen E34 output geometry, no fit/inference.
All5fold/3channels/3roles,20480queries/61440candidate outputs,53inventory.
Selection immediate:22patient-mean harms,20positive MAE right derivatives,
2crossing-driven harms;4SmoothL1-gain/MAE-harm targets. Of18fit-to-selection
reversals17have positive derivatives. Uniform positive OUTPUT scaling cannot
improve those fixed target MAEs; not a claim about recurrent weight scaling.
Macro derivative remains negative, so aggregate shrinkage potential must not
be confused with rescuing every target. E9 teacher geometry and old exact-MAE
repair already tested related ideas; no blanket loss-switch proposal.
20tests,30bindings,all rows scalar/fsum verified,maxdiff3.553e-15,oldMAE reproduced.
Findings RECONSTRUCTION_BRIDGE_GEOMETRY_FINDINGS_20261005.md. Direction-transfer
cause unresolved; no promotion/runtime/weights/interval changes.

2026-10-05 E34 COMPLETE/VERIFIED, supersedes prepared-only notes below.
Same E29 start/data/order/budget, bridge-only training with fully fixed
encoder/state/readout.15models,20480queries,all53inventory; 20tests.
Saved-model explicit replay parity0; original controls/weights checked;
independent scalar MAE/paired statistics maxdiff8.882e-15. All7screens/full0.
Matched-start patient macro fit .753682->.747468, selection .876270->.876812,
same-source .781413->.776376.39/47fit mean gains,18selection and12same-source
fit-to-query non-gains (descriptive, not significant counts). Co-moving
backbone is not necessary for these reversals; freezing alone insufficient.
Same-source mean beats joint, but BUN/glucose/Hgb selection favor joint;
no universal replacement or loss-of-all-signal claim. Close fixed version,
no parameter sweep. Runtime/weights/intervals unchanged, full repair unresolved.
RECONSTRUCTION_FROZEN_BRIDGE_FINDINGS_20261005.md; verification artifact.

Historical entry (superseded by the completed E34 evidence above):
2026-10-05 DOCUMENTATION ONLY: latest user asks for a mandatory pre-task
Markdown handoff, not another experiment. ROOT_CAUSE_LEDGER.md remains the
single decision record. E33 is the latest completed evidence. E34 protocol,
trainer and test files exist; tests/training/scoring/independent verification
are not completed. Do not resume E34 from an earlier plan without checking
the latest request. No new research evidence or runtime/weight/interval change.
The full generalization repair remains unresolved.

2026-10-05 E33 TRAINED-BRIDGE ABLATION COMPLETE. All15checkpoints,20480queries,
fit/selection/same-source, all53inventory. Original-on/off-native parity0;
unchanged weights checked. Independent scalar aggregation maxdiff3.553e-15,
17tests. Immediate bridge mean-harm26/46selection,26/44same-source;
25/24 previously helpful in fit. Selection macro on .876522/off .875705/
unbridged .875949/raw .853856. Bridge contribution+.000818, remaining-.000245;
same-source+.000421/-.000136. Local inference-path responsibility, NOT isolated
training causality; Hgb/renal counterexamples forbid global removal. No new
gate/promotion/runtime changes. Frozen-backbone vs joint training at identical
initialization needs history audit, not yet executed. Findings:
RECONSTRUCTION_BRIDGE_ABLATION_FINDINGS_20261005.md. E32 proposed ablation is
now completed; earlier pending notices are historical. Full repair unresolved.

2026-10-05 E32 COMPOSITION DIAGNOSIS COMPLETE. Frozen E29 fit vs E31 third
role, all53x5x4. Percontrol206common-support/49missing-role/10no-common-source.
Unbridged comparison:170/206positive within-source gain erosion,185/206within
absolute term dominates composition. This is NOT170new baseline losses or
independent significant tests. Every decomposition includes at least one
source-role under30; patient mixture/support/noise remain. Existing midpoint
helper reused; independent SQLite824contrasts maxdiff1.388e-17,31bindings,
13tests. No training/promotion/runtime changes. E29 jointly adapts bridge and
state/readout; fixed-checkpoint bridge ablation needs history audit before any
new intervention. RECONSTRUCTION_ASSIMILATION_COMPOSITION_FINDINGS_20261005.md.

2026-10-05 E31 COMPLETE/VERIFIED: same-source adaptation-unseen role on frozen
E29 models,5120queries/1613people/44scored. Of22previous selection reversals,
13also lack same-source mean gain,8retain mean gain,1unsupported. Descriptive,
NOT significant-gain screens; some targets have2-27people.5105queries are
epoch10-update-exposed. Source switching is not necessary for these observed
mean reversals; composition/support/sampling and adaptation failure remain.
Verifier14534 exit0: five arms/allfolds predictions exactly replay, role/exposure/
support/summary checked, max mean roundoff8.882e-16;16tests pass. No model repair
or promotion. See RECONSTRUCTION_ASSIMILATION_TRANSFER_FINDINGS_20261005.md.
Next diagnosis requires same-fold/target common-source error decomposition,
not another hyperparameter search. Earlier pending notices are historical.

2026-10-05 DOCUMENTATION HANDOFF: ROOT_CAUSE_LEDGER.md is the mandatory
pre-task decision record, linked by both AGENTS.md files. E31 existing inference
29146 exited0 naturally; aggregate saved, independent verification NOT run.
Do not treat raw E31 output as a verified repair or promotion. Latest request
is documentation only; no new experiment, runtime change, commit or push.
The E30 statement "NOT yet inferred" below describes its historical endpoint.

2026-10-05 E30 ROLE IDENTIFIABILITY AUDIT COMPLETE. E29 fivefold fit/selection
source-component overlap0 and patient overlap0 in every fold. Of53x5 inventory,
219target-folds have both roles but no common source,46lack role support.
Poolingfolds yields5common components but cannot create same-model controls.
23bindings checked; independent SQL confirms intersections;9tests pass.
Cannot separate in-sample optimism/new-patient error from new-source error.
Stop unsupported within-source decomposition. A third adaptation-unseen,
same-source role needs original-encoder exposure audit before interpretation;
NOT yet inferred. No model repair, training, runtime or promotion change.
RECONSTRUCTION_ASSIMILATION_ROLES_FINDINGS_20261005.md and ROOT_CAUSE_LEDGER.md.

2026-10-05 E29 COMPLETE AND VERIFIED; NO PROMOTION.
Original training49649 exit0; verification13535 exit0, all15models/bothroles
prediction differences0, baseline/hash/statistics checked, mean roundoff1.776e-15.
Same-body immediate organ assimilation:2048patients/5120selection occurrences,
46scored targets,27minimum-support. Patient-macro MAE immediate .876522,
delayed .876776/body .877487/unbridged .875949/frozen .877001/raw .853856.
All five comparison screens and full screen0; not accuracy percentages.
Diagnosis4169 exit0:46/46fit means improve over unbridged,22selection reversals;
28/46fit means fail to improve over body control. No saved-hook tanh saturation.
26tests pass. No retraining/runtime/interval changes; full root cause unresolved.
RECONSTRUCTION_ORGAN_ASSIMILATION_FINDINGS_20261005.md links assessment,
verification and diagnosis. Assessment verification=false is preserved historical
snapshot; separate complete verification artifact is authoritative for replay.
Close this fixed bridge/budget, not all assimilation methods. Delay/count and
body-input-rank confounding remain. Read ROOT_CAUSE_LEDGER.md before every task.

2026-10-04 E28 REACHABILITY DIAGNOSTIC COMPLETE: all15 E27 saved models have
no last-organ-proposal path to body-only same-time reconstruction; one-hour
advance activates path, body observation bypass remains active. Exact-zero
final gaps9403/10240fit and4634/5120selection occurrences, not unique people.
Local branch timing constraint, NOT90.51% information loss/error attribution.
Two structural tests pass; no training, prediction quality or promotion claim.
E25 direct-organ-reader negative remains; immediate coupling needs matched
controls before repair claims. RECONSTRUCTION_ORGAN_REACHABILITY_FINDINGS_20261004.md.

2026-10-04 E27 COMPLETE, NO PROMOTION: fivefold/three-arm15 model adaptation;
5120selection queries/2048people/46targets,27 meet minimum support. Binding
patient-macro .875949 vs native .876430/postpool .876059/raw .853856; all screens0.
Native comparison:23/46 fit means not improved,9 fit gains reverse on selection.
Original verifier40442 failed exact mean comparison:143MAE-only differences,
max1.776e-15, vanish when identical numbers use original DataFrame construction.
Only MAE now uses established64eps roundoff check; paired stats/support/screens
remain exact.30regressions pass; full verifier31688 exit0,15models both roles
replay exactly, raw and all statistics/summary verified. No weights/runtime/
interval changes. Fixed pre-pool binding adaptation insufficient, not all possible
bindings disproven; full generalization repair unresolved. Read ROOT_CAUSE_LEDGER.md
and RECONSTRUCTION_ORGAN_BINDING_FINDINGS_20261004.md. Historical E26 follow-up
below is no longer untested; this did not retrain the event encoder end to end.

2026-10-04 E26 ORGAN ASSIGNMENT DIAGNOSTIC COMPLETE: mean(phi(numeric)+identity)
erases within-organ value/identity binding.224same-organ pairs x6synthetic cases,
fivefold mappings/four distinct epoch10 checkpoints: original organ proposal,
update and6h state distinguish0/1344 each; body distinguishes1344/1344.
Parameter-free pre-pooling SiLU binding distinguishes all tested pairs. This is
a local structural repair candidate, NOT real-outcome accuracy or globalcause.
Existing E19 fit inputs:10240contexts/3094uniquepeople,9855contexts have distinct
co-observed same-organ entries after target/sibling masking; exposure not error
attribution.13tests; no runtime/weight/interval/gate/training change. See
RECONSTRUCTION_ORGAN_ASSIGNMENT_FINDINGS_20261004.md. Follow-up matched adaptation
is now completed in E27 above; do not redeploy old weights on changed coordinates.

2026-10-04 E25 COMPLETE, NO PROMOTION: all5319744 queries/10roles verified;
940new HGB100 readers trained on full fitting support,235raw models contract-checked.
Selection47targets/235groups/1797884queries/29580patients; body .854767,
organs .849663, expanded .853201, raw .825737 patient-macro MAE.
Organs beat body2/shuffled3/native9 fixed screens, expanded0/raw0; joint0.
112/235fit groups already lose raw; all47 selection patient means lose raw.
Full support alone is not a sufficient repair for this fixed concatenation family.
Independent1175reader replay/statistical verification83993 exit0, max roundoff
2.9976e-15; verifier fsum fixes accumulation, original64eps/gates unchanged.
41regression tests pass. Independent organ trajectory replay remains pilot1024,
not all5319744. All sessions ended; runtime/weights/intervals unchanged.
See RECONSTRUCTION_FULL_ORGAN_FINDINGS_20261003.md and ROOT_CAUSE_LEDGER.md.
Close fixed branch; rootcause/generalization repair still unresolved. Older
incomplete snapshots below are historical, not the current E25 status.

2026-10-03 E25 CONTINUATION, STILL INCOMPLETE: fold0 fit resumed by100000,
now108704/698766; selection8704/363526; total117408/5319744, zero complete roles.
Native body/point parity0, all stored payload/reference keys and prior pilot
prefix revalidated. Extraction48626/progress6968 exited0, no live run claimed.
Full-support trainer and saved-reader scorer implemented: four HGB100 arms,
verified original raw reuse, exact role/schema/weight bindings, five fixed
paired contrasts, all53 inventory. All ten complete caches required BEFORE fit;
real partial-grid rejection verified.37tests incl synthetic SQLite-to-fit/save
entry and raw replay. This is engineering progress, NOT real reader efficacy.
Remaining roughly21.45h/9.35GiB by ordered-prefix rates, not upperbounds.
Independent full assessment verification and full fitting still required.
Protocol/extractor/checkpoints/runtime unchanged. See E25 progress document.

2026-10-03 E25 INCOMPLETE: full-support organ readout protocol admitted after
E24 weak-control diagnosis. Resumable native extraction/identity/parity cache
implemented; fold0 fit+selection each8704, total17408/5319744 queries. No reader
training/efficacy assessment yet. Raw/truth exact; body/native0error; independent
stepwise organ replay1024 exact, retainedprefix checked after real resumption.
34tests. Allcommands exited; no claimed background fullrun. Remaining extraction
roughly22h/9.53GiB by observed prefix rates, not reliable upperbound. Full fivefold
scope unchanged. RECONSTRUCTION_FULL_ORGAN_PROGRESS_20261003.md and protocol.
No runtime/weights/interval/promotion change; complete rootcause/repair unresolved.

2026-10-03 E24 COMPLETE: identical E23 queries, saved full-fit vs bounded-fit
raw265/HGB100. No training. Fivefold131groups/4569queries/1851people/28targets.
Fullraw .638187 versus smallraw .799944/native .715713 patientmacro;
component .716334/.947003/.842483.27/28 mean-wins versus small onbothlevels,
20/28 versus native. Allthree53x3correctedscreens0; NOTpromotion/organrepair.
Fitmedian26325vs75; actualtreeleaves15vs2.785. Smallreader NOT a proxyforfullfit.
Population/weightschange too, notpureNcausality.262savedreaders exactreplay,
independentlossaggregation,pairedstats/screens identical,roundoff8.88e-16;
12tests. RECONSTRUCTION_READER_SUPPORT_FINDINGS_20261003.md. Fullorgan
readout still requires adequate matchedfit support andstrongnative controls.
Runtime/weights/intervals unchanged; fullrootcause/repair remains unresolved.

2026-10-03 E23 COMPLETE, not promoted: full organ compartments added to the
same frozen masked-state readout did not improve robustly. Fivefold131groups,
28supported targets/1851people/4569queries;1310readers, eightcontrasts0screens.
HGB body .783467->organs .789964; native .715713, boundedraw .799944.
Fitmedian75/target: do not use weaker small-sample readers to conclude no
organ information. Ridge131fit improvements/100selection reversals; HGB110/56.
Savedreader predictions verified; independent15360context body/organs/native
replay exact. Aggregate-key error repaired by separate scorer, no retraining.
See RECONSTRUCTION_ORGAN_READOUT_FINDINGS_20261003.md and ROOT_CAUSE_LEDGER E23.
No release changes; full generalization repair unresolved. No more parameter
sweeps on this bounded direct-concatenation branch.

2026-10-03 E22 SUFFIX DIAGNOSTIC COMPLETE: same E19/E20 contexts, post snapshot
versus subsequent advance-only, update-only, full. Fivefold177groups/708newreaders,
88122queries/2048patients. Bothreader screens10post<update,8advance<body;
0jointpost<advance,0eachupdate<body. HGBmacro .686716/.727531/.885117/.893199.
Later assimilation contributes more consistent known-content readability gaps;
NOT masked inference harm/repair, irreversibleloss, causal physiology or forecast.
Allsavedreaders reproduced, originalpost/final0error, pairedstats/screens exact;
MAEaggregate roundoff<=7.11e-15,37tests. Runtime/weights/intervals unchanged.
RECONSTRUCTION_SUFFIX_FINDINGS_20261003.md and7aggregateJSONs. Do not repeat
suffix separation as untested; old-value retention alone is not solution (E21).
Full rootcause/generalization repair remains open; no promotion.

2026-10-03 E21 CONTENT TRAINING COMPLETE: frozen event encoder, original state
and masked head, equal-capacity primary/detached vs real vs shuffled aux labels.
Five folds/15 models/94908 queries/2048 patients; same E19 hash contexts.
Masked46 scored(27 minimum support): primary .875550,real .874808 patientmacro,
0correctedscreens; known content47(41 minimum support): .887196->.885561,
hemoglobin/respiratory_rate pass both comparisons, NOT unknown inference.
All15 saved models exactly reproduced, encoder unchanged,15360 original context
states reproduced with0error,24tests. No runtime/interval/forecast/promotion.
Fixed retention intervention NOT universal repair; no weight/budget sweep.
RECONSTRUCTION_CONTENT_TRAINING_FINDINGS_20261003.md and7aggregateJSONs.
Full cause/repair remains open; do not re-list this intervention as untested.

2026-10-03 E20 ASSIMILATION CONTENT COMPLETE: same E19 cohort/contexts/frozen
encoder,177variable-folds/354newreaders/88122queries/2048selectionpatients.
Separately fit immediate post-update readers,original event/final predictions
unchanged. HGBevent/post/final .628675/.686716/.893199patient; Ridge
.911219/.999468/1.259259. Jointreader screens12 immediate/13later/9both.
Known-content accessibility gap begins at assimilation for supported targets;
late advance versus update and irreversibleloss remain UNRESOLVED. Notrepair.
Event/final exactreproduction0;177scalersfinite;24tests; no runtime/promotion.
RECONSTRUCTION_ASSIMILATION_FINDINGS_20261003.md +7aggregateJSONs,ledgerE20.
Next intervention must protect known content AND original masked inference;
rawcopy/pooledreplacement not success. Full generalization repair remains open.

2026-10-03 E19 OBSERVED CONTENT COMPLETE:53 considered,36 supported,35/all5folds.
177 variable-folds,1062readers,88122queries,2048unique selection patients.
Same observed value,last-observed event64 vsfinalbody64,both+159process;
Ridge event/body macro .911219/1.259259,HGB .628675/.893199patient.
Event mean better36/36both;23Ridge/19HGB correctedpaired screens,intersection16.
Accessible known-content fidelity declines along state path; NOT irreversible
loss proof,masked inference,forecast or promotion. Immediate vs later update
not yet isolated. Predictions fully reproduced,531scalers valid,19tests.
RECONSTRUCTION_OBSERVED_CONTENT_FINDINGS_20261003.md and sevenaggregateJSONs.
ROOT_CAUSE_LEDGER E19; no runtime/weights/interval change,full repair unresolved.

2026-10-03 RAW BLOCK DIAGNOSTIC COMPLETE47/all5,940 fits,59629queries/5703patients.
Same E17 hash512fit/256selection, frozen original raw cache, target/sibling mask.
Refit remove-mean/remove-latest/remove-lag-count; keep presence, fixed HGB.
Patient macro raw .869179, latest_process .873490, mean_process .876621,
values_presence .870159, body+process .885501. Seven comparisons0correctedscreens.
Lag/count deletion retains average raw advantage, not pure-physiology proof;
no universal latest/mean objective justified. Full235 baseline reproduction,
saved models/predictions/bindings checked,17tests. E18 in ROOT_CAUSE_LEDGER,
RECONSTRUCTION_RAW_BLOCK_FINDINGS_20261003.md. No runtime/interval/promotion.
This completes block necessity diagnostics, not encoder information-loss proof.

2026-10-03 STAGE PROBE COMPLETE47/all5,2350 readout fits,59629queries/5703patients.
Same masked histories,frozenepoch10; eventmean/last/body64+same159process;
raw265/process159 controls;512fit/256selection hash sample per target/fold.
HGB macro eventmean .911521,last .895912,body .885501,raw .869179patient.
Bodymean beats eventmean41/47HGB,36/47Ridge; eight contrasts0correctedpasses.
Reject simple event pooling as universal state repair, not information-loss proof.
Ridge raw does not win overall: reader dependence remains.12tests,all processes
exit0,bindings/query identities/truth verified. No runtime,outer or promotion.
RECONSTRUCTION_STAGE_PROBE_FINDINGS_20261003.md, ledgerE17. Bounded diagnostic,
not full-cohort efficacy; do not repeat as untested stagewise comparison.

2026-10-03 HISTORICAL STATE-PATH RECONCILIATION, no new model experiment.
ROOT_CAUSE_LEDGER E16 now includes bounded7 body/organ ablations, fitting
exposure, weighted objective, integration resolution and protected finite steps.
Eight aggregate reports and22 nonprivate code/report binding entries checked;
10 focused tests pass; private data/weights not rehashed or replayed.
BUN dynamics harm already occurs in fit, but global deletion harms24/33 and
21/33 target means. These are OLD findings, not epoch10 diagnosis or promotion.
Do not repeat deletion/finer-step/protected-direction as untested universal fixes.
Intermediate event representation versus persistent aggregation remains to be
tested with separately fitted controlled probes; final-readout traces do not
prove information loss. No new training, runtime or interval changes.

2026-10-03 OBSERVATION-ALIGNED COORDINATES COMPLETE47/all5,705HGBheads.
Fixedbody64+fit-onlydecoded265, equal-widthrandom/shuffledcontrols; noqueryraw.
Bodymacro .857139->.854847patient/.811241->.809938component(.2675/.1606%gain).
0correctedwinsvsbody/raw/random/shuffled,0fullscreens;45/47meanharmsvsrawbothlevels.
Fit229/235improvebodybut153/235stillloseraw: notonlydomainshift/gates.
470scalersfinite,negativevariance>=-9.202e-34guardedscale1;15tests.
FixedalignmentbranchCLOSED,noalpha/dimension/seedsearch,noouter/runtimechange.
RECONSTRUCTION_ALIGNED_COORDINATE_FINDINGS_20261003.md +8aggregateJSONs;
ROOT_CAUSE_LEDGER.md E15 and graph.md updated. Globalcause remains unresolved.
Supersedes the historical paused/partial state below; training/assessment complete.

2026-10-03 DIAGNOSTIC LEDGER, DOCUMENTATION ONLY. Read ROOT_CAUSE_LEDGER.md
before every task; AGENTS.md enforces it. Separates confirmed local repairs,
readout-coordinate confounding, target-specific first failures and unresolved
global cause. Closed kernel branch is not an untried repair. User paused research;
aligned-coordinate run stopped at folds0/1/2 complete47 each, fold3 complete17,
fold4 unfinished; no aggregate assessment, no efficacy conclusion. Do not resume
from a documentation/status task. No model/runtime/interval/promotion changes.

2026-10-03 SAME-MAP Cx10 COMPLETE47/all5,940fits allconverged77-4517iterations.
Frozen scaledmap/encoder/labels/roles, relearncoefficients, noqueryselection.
Bodymacro .886598->.873760patient/.835116->.823452component(1.45/1.40%gain);
raw .913889->.902195/.863185->.851674. Body3/raw6correctedwinsvsoldC,
0vsnative/bodyHGB/rawHGB,0fullscreens. Body/raw/processFITimprove235/235but
stillloseHGB232/235. Penaltypartialcause,notstrongreadoutrepair. Body31/30,
raw46/42targetmeansworseHGB;notgate-onlyfailure.23tests,noouter/runtimechange.
RECONSTRUCTION_KERNEL_RELAXATION_FINDINGS_20261003.md +6JSONs. KERNEL REPAIR
BRANCH CLOSED: nofurtherC/landmark/bandwidthsweepsonreusedqueries. Returnto
originalshared-stateevidence;globalrootcause/generalizationstillunresolved.
No weights/interval/promotion. Do notrepeat thisasuntestedregularizationfix.

2026-10-03 FIT kernel objective/spectrum COMPLETE47/all5/1410savedmaps.
Scaledbody/raw effective ranks31.39/33.91,95%rank58: notconstantcollapse.
L1rayoptimalgains1.3666/1.7865 improveFITMAEonly.690/1.311%;penalizedobjective
worsens235/235both. Regularizedray improvement<=5.23e-7: no large ray-solver
failure,NOTfullKKTproof. BestTRAINgainstilllosesHGB232/235;recoversmedianonly
5.04/7.06%positivefitgap. Amplitudeshrinkageexistsbutnotmaincompletefix.
Same-mapcoefficientdirection/finitefeatureexpressivityremainunresolved,not
proveduniqueencoderdefect. RECONSTRUCTION_KERNEL_OBJECTIVE_FINDINGS_20261003.md
Evidence: objective/ray_gapJSONs,20tests. No refit/query/outer/runtime/interval/promotion.
Nextprespecifiedfit-internalregularizationinterventiononSAMEfeaturemapunrun;
do not deploy retrospectivefitgains or repeatglobalrescaling asuntried.

2026-10-03 fit-only SCALED KERNEL COMPLETE47/all5,940fits allconverged.
Samefrozenbody/raw+StandardScaler, noquerytuning. Rawlagfraction99.34->22.31%,
numeric.412->29.23%;rawMAE .937564->.913889patient/.889135->.863185component,
2.53/2.92%gain;10correctedwinsvsoldrawkernel. BUT0fullscreens(bodyorraw),
0winsvsnative/bodyHGB/rawHGB. Body .886598/.835116 slightlyworseoldkernel;
rawlosesALL47patientmeansvsrawHGB. EachscaledsourcefitlosesHGB232/235.
Scale defect partlyfixed,notsharedstate/generalizationrepair. 705scalersfinite;
tinynegativevariance>=-9.21e-34 guardedscale1,notnonfinitepredictionfailure.
Rawqueryzmax3566.7/21constantfitvariationwarnings,noclipping.19tests.
RECONSTRUCTION_SCALED_KERNEL_FINDINGS_20261003.md+7JSONs. Noouter/runtime/
weights/interval/promotion. Nextfit-onlykernel spectrum/objective diagnosis
unexecuted; don'trepeat basicstandardization asuntested universalfix.

2026-10-03 rotation-invariant KERNEL COMPLETE47/all5,940fits allconverged.
RBFNystroem128+L1SVR, samefrozenbody/raw/process/shuffle; noouterqueries.
Bodymacro .884742/.834098 losesbodyHGB .857139/.811241;0fullscreens/0HGBwins.
Bodybeatskernelraw/process/shuffle17/18/22,notnewability. Rotationfeature/output
errors<=1.69e-9/5.10e-11. PostresultFITmetricdiagnosis235combinations: EVERY
medianlagdistancefraction>90%,median99.3419%;numeric0.4120%. RawEuclidean
metric dominatedbyhourlag/missing49,notappropriateevidenceofabsentphysiology.
Eachbody/raw/processkernel losesitsHGBfitting232/235: notonlytransferfailure.
Rawmetricdefectdoesnotexplainallbodyfailure. Fixedscale/covariance-aware repair
stilluntested; nohyperparametertuningonqueries. RECONSTRUCTION_KERNEL_FINDINGS_20261003.md
+7JSONs;17tests484assessmentbindings. No runtime/weights/interval/promotion.

2026-10-03 LOSSLESS coordinates COMPLETE47/all5,940fits1797884sourcequeries.
FullPCA/randomorthogonal bases keepall64/265dims, no whitening/scaling/truncation;
maxinverseerror7.89e-13. Rawrotation macro .825737->.916796patient/.787947->
.875915component (+11.03/11.16%);47/47patient46/47component meanharms despite
identicalinformation. FiniteHGBcoordinateaccess is nowCONFIRMEDconfound; rawvs
bodyperformancegap alonecannotproveencoderloss. BodyPCA .858206/.816076 vs
body .857139/.811241:0increment,0fullscreen,notrepair. BodyPCA harms43/47patient
32/47componentmeans. Notproofalllatentinformationretained/onlydecoderproblem.
RECONSTRUCTION_COORDINATE_FINDINGS_20261003.md +6JSONs;16tests484bindingsverified.
Samequeries/realtruth/masks;noouter/runtime/weights/interval/promotion change.
Singlefixedrotation, reusedsource, notpatientshuffle. Do notrepeatPCAasnewfix.

2026-10-03 native-anchor truth residual COMPLETE47/all5,940fits/1797884queries.
Same frozen body64+native65, real-label residual vs absolute/anchor-only/shuffled
state; fixed100HGB, all original query/reference parity. 0full screens,0increment
vsbody/raw/absolute/observation-skip;8vsnative,8scalarcontrol,11shufflecontrol.
Macro .855898/.810218 vsbody .857139/.811241 (only.14/.13%); raw .825737/.787947.
Residual loses46/47patient and44/47component means vsraw: notmerelystrictgate.
Nativeanchor helps overnative butnotnewnonlinearreadoutcapability. Teacherloss
mismatchremoved, observed representation/readout gapremains; NOTproofirreversible
encoderloss. Fitresidual vsabsolute wins130/235patient,113component,116bounded.
RECONSTRUCTION_NATIVE_ANCHOR_FINDINGS_20261003.md +6JSONs;16tests/485bindings.
Reusedsource only; fitnative notOOF,conditionalshuffle retainsrealanchor.
Noouter/runtime/release/interval/promotion change. Do notrepeat asuntestedfix.

2026-10-03 teacher error geometry COMPLETE47/all5,1797884source queries.
Exact retrospective decomposition, nofit/newpredictions/outerqueries. Gross
harm/benefit .057177/.060528 patient, .058103/.057824 component; net small
but cancellation large. 93.59/92.58% harm moves AWAY from truth,not overshoot.
70.42/72.40% harm where teacher worse; betterteacher+studentaway also harms
(17.40%patientmacro population). Twoobstacles: inappropriatepointteacher AND
failedstudentcorrection; no irreversibleencoderloss orprospectivegate proof.
RECONSTRUCTION_TEACHER_ERROR_GEOMETRY_FINDINGS_20261003.md +aggregateJSON.
11tests/239bindings verified. No runtime/weights/interval/promotion change.
Hindsight teacherquality/truthdirection MUST NOT become routing features.

2026-10-03 fixed teacher readability COMPLETE47/all5,705fits,1797884queries.
Same body64 learns fixed maskedraw HGB output; process/shuffle students controls.
23full imitation screens,30vsbody imitation screens;235/235fitpatientimprove.
BUT0truth increment vsbody,0vsteacher,0fulltruthscreens. Among23imitationpasses,
17patient/13component truthmeansharm. Truthmacrobody .857139/.811241 ->student
.853788/.811520; imitation .230677/.230200 -> .197151/.193756. BUN/creat/Hgb
closerteacher butworsetruth. Pointteacher imitation is not realaccuracy repair.
Optimalconditional-median teacher counterexample tested; possiblemechanism,
notuniqueclinicalcause. RECONSTRUCTION_STATE_TEACHER_FINDINGS_20261003.md +6JSONs.
17tests; noouter/runtime/weights/interval/promotion. Distinct fromold unmasked
retentiondistillation; notproofirreversibleloss norcompleteinformationretention.

2026-10-03 SAME-state observation skip COMPLETE47/all5,705newHGBfits.
1,797,884source queries/29,580people; native/body/raw baselines reproduce.
Body64+maskedraw265 vs same329process/shuffled-branch controls; noouterreads.
Macrobody->skip .857139->.828889patient/.811241->.784964component (3.30/3.24%).
10targetsbeatbody;6also beatbothaugmentedcontrols:albumin/ALT/AST/bicarbonate/
HR/totalprotein. BUT0beatraw-only,0fullsixcontrast screens,NO newpromotion.
Raw-only .825737patient/.787947component: sharedstateincrement remainsunproved.
46/47bothmeansimprovevsbody;chloridecomponentharms;ferritinstilllosesnative.
Observationaccess partlyrepairsbody/readoutunderuse,notproofirreversibleencoder
loss orfutureforecastgain. RECONSTRUCTION_OBSERVATION_SKIP_FINDINGS_20261003.md
+sixaggregateJSONs;12tests. No runtime/weights/interval changes. Do notrepeat
thisskip as untested; distinguish rawsignalavailability from sharedstatevalue.

2026-10-03 SAME-state readout blend COMPLETE47/all5,1,797,884source queries.
Component-held-out exact MAE convex weight;native/body100/equalblend/processblend/
shuffleblend controls. 0fullsixcontrast screens,0incremental vsbody100. Median
signed-error Pearson .98937,Spearman .96640,opposite-error fraction9.369%.
Body->blend macro .857139->.856785patient/.811241->.810552component,tinygain.
Ferritin/troponin mostly shrink toward native and still lose native;BUN/INR
slightly worsen. O2sat mean complementarity does not pass component CI.
RECONSTRUCTION_READOUT_BLEND_FINDINGS_20261003.md +aggregateJSON;26tests,
13bindings verified. No outer/runtime/weights/interval/promotion change.
Same-candidate support selector,early stopping andblend are now tested; do not
repeat as new universal repairs. Representation/conditional information remains
unresolved; correlated outputs alone did not provide a stable reconstruction fix.

2026-10-03 frozen nonlinear PREFIX intervention COMPLETE47x5, six stages.
Within-source-component held-out stage choice;1,797,884queries,1288supported/
23unsupported decisions per arm,44fullysupported targets. Native/body100/raw/
process/shuffle endpoints exact;20tests pass. 0fullfivecontrast screens,0versus
body100 corrected incremental screens. Macro patient .857139->.856416,
component .811241->.811243. Troponin improves body100 but not native/control
component gate;central fit harm emerges between sampled10/25 stages while
tail gains grow. Ferritin/BUN/INR patient means worsen when shortened.
RECONSTRUCTION_PREFIX_COMPLEXITY_FINDINGS_20261003.md +aggregateJSON.
No outer/refit/runtime/interval/promotion change. Early stopping partly reduces
troponin harm,not a universal repair or proof of identifiable individual signal.
Do not repeat support-only selectors/global offsets/early stopping as untested.

2026-10-03 SAME nonlinear heads fit/selection diagnostic COMPLETE47x5.
3,521,860 fit pair occurrences; all25 bindings verified,14tests pass. Body beats
native on fit232/235;54patient/58component comparisons reverse on selection.
Ferritin central fit improves then selection harms5/5;troponin central ALREADY
harms fit5/5 despite overall fit gain, worsens further on selection. Separate
conditional objective tradeoff from transport failure; not one proved root cause.
BUN/INR overall fit+selection gain5/5; prior corrected confidence gate still binds.
Original float64 median preserved; every float32 cached label verified.
RECONSTRUCTION_NONLINEAR_FIT_GAP_FINDINGS_20261003.md + aggregate JSON.
No runtime/weights/promotion/outer change. Next same-head prefix-complexity
source-component test can isolate overspecialization; unexecuted, not a new
encoder or support-selector sweep.

2026-10-03 SOURCE support selector COMPLETE47/all5,1797884occurrences.
Fixed withinfold component-held-out expected absolute-error difference, masks/
measurementages only: aggregate4/variable106/shuffled106/constant controls.
1288supported/23unsupported groupfits retained;44targets complete. Variable
choice macro .863143/.816275 loses alwaysbody .857139/.811241 andconstant;
0fullfivecontrast screens,0incremental vsbody/constant/shuffle. Gainsvsnative
alone notselectionbenefit. BUN/INR constant choosesbody100%;detailedswitches
worsenmeans. Error-differenceMSE beatsconstant1/45patient2/45component only.
12tests; RECONSTRUCTION_SUPPORT_SELECTOR_FINDINGS_20261003.md +aggregateJSON.
No runtime/promotion/weights change. Observation-support alone is not thisfix;
notproofphysiologyabsent. Next compare SAMEheads fit/selection error tolocalize
fitfailure versus overfit/transport,not newselector/architecture search.

2026-10-03 nonlinear SOURCE error decomposition COMPLETE47/all5,7tests.
Ferritin/troponin harm mainly near fit center, NOT tail: central contributions
+.02846/+.12710 vs tail -.00258/-.07776. Their matched local-offset comparison
still loses native; O2sat tiny mean ordering reverses after matched centering,
not transferable/significant calibration proof. BUN gains outside center;INR
all3bands gains. Label bands are retrospective,NOT clinical normality or legal
router inputs. RECONSTRUCTION_NONLINEAR_SOURCE_ERRORS_20261003.md +aggregateJSON.
No weights/runtime/promotion/gate change. Next source-only masked input support
diagnosis: determine whether pre-query observability identifies adverse cases;
not more tail removal/globaloffset sweeps. Fullgeneralization stillunresolved.

2026-10-03 nonlinear SOURCE component diagnosis COMPLETE47targets/fourcontrols.
Newevidencequalifies "unstable":BUN/INR all12group meansbettervsnative/process/
population/matchedshuffle;no supportednominal/twohalfharms;leaveanygroupstill
benefit. Magnitude/certainty,notdirectionsignreversal,blocksscreens. INRsingleton
retainsunknownvariance;BUNprocessspreadnotonlysamplingunderconditionalestimate.
Otherfailuresrealwarning:troponin8/10nativegroupsharm(4nominal,5twohalf),ferritin
10/12nativeharm,O2sat6/12. Donotcallallfailurespower-only. Ninefocusedtests;
RECONSTRUCTION_NONLINEAR_SOURCE_COMPONENTS_20261003.md +aggregateJSON,alljobsdone.
No fitting/gate/runtime/promotionchange. Nextfrozenheaddevelopmentcomponent
evidenceforBUN/INR withencoder/head-fitdisjointroles,notposthocthresholdrelaxation;
adverse-targeterrorsrequireseparatediagnosis. PriorLAD/loss/siteconstraintsalready
triedothercohorts;notuntesteduniversalfix. Fullgeneralizationstillopen.

2026-10-03 matched nonlinear placebo COMPLETE235fits/all47/all5. Same body64/
HGBcapacity, frozen realtrainedencoder; withinfitcomponent cyclicstate shuffle.
Source29/47,outer26/47 beatmatchedshuffle patient+component47screen;overlap24.
All7priorouterincremental+controltargets survive(ALT/AST/BUN/Ca/HR/INR/K),but
sourcefullconjunction4(ALT/AST/Hct/phosphate),bothrolesonlyALT/AST. Not7promoted.
Outerbody .891238patient/.877110component vsnull .983810/.965183. Individual
pairing useful, notphysiology-onlyproof orfullyshuffledencoder;onenullseed.
Source12componentsvsouter36-38forBUN/K/INR: insufficientfamilycorrectedsource
componentbenefit remainsfirstblockerforfiveotherouterpositives,notabsenceof
personsignal/power-onlyproof.24tests,alljobsdone; no runtime/forecast/interval
change. RECONSTRUCTION_NONLINEAR_PLACEBO_FINDINGS_20261003.md +sixaggregateJSONs.
Nextsource-only conditionalcomponent diagnosis, notouterrouting/gatelowering
orpriorfailedglobaloffsetagain. Fullgeneralization remainsopen.

2026-10-03 nonlinear probe COMPLETE all47/fivefolds, no encoder changes. Full
source1797884occurrences29580people; outer1800214pairs73670people, allbindings/
roles/native reproduction pass. Frozen body64->fixedHGB outermacro normalized
MAE .917355->.891238patient/.894102->.877110component (2.85%/1.90% reduction).
Raw265->sameHGB .862525/.854088 (5.98%/4.48%). Body7targets beat native AND
new process/pop controls under8x47 bothlevel diagnostic screen: ALT/AST/BUN/
Ca/HR/INR/K; sourceincrementaloverlap4(ALT/AST/BUN/K). Raw15jointscreen but
notdeployment. Body15targets harm>=1mean; excludingALT/AST gains1.23%/.88%.
Readout underuse partly demonstrated, encoder loss notisolated (capacity/loss/
optimization differ); no blanket no-signal conclusion.17tests; alljobsdone.
RECONSTRUCTION_NONLINEAR_FINDINGS_20261003.md +source/outer/robustnessJSONs.
Current masked reconstruction only, reuseddevelopment, notforecast/newexternal/
promotion; runtime/intervals unchanged. Nextmatched nonlinear shuffled-state
control stillneeded; priornativepermutation notproof fornewreadout. Do not
choose targets/heads fromouter. Fullgeneralization remainsunresolved.

2026-10-02 INPUT DELIVERY gap confirmed/fixed. Prior export strictloaded but
omitted documented native-history builder; private cached-tensor parity missed
the actual CSV boundary. Added canonical event CSV->history->normalized input
CLI/schema, synthetic example and complete bilingual guide. Isolated package
4tests +188 fold-target exact tensor/prediction checks pass; target omission
preserves output. Four weight SHA256 unchanged; no patient data uploaded.
Private release coverage-reconstruction-20261002-input-complete published and
remote hashes verified. COVERAGE_INPUT_DELIVERY_FIX_20261002.md. Source-specific
item/unit/availability qualification remains upstream; not generic raw EHR import.
No forecast/promotion/interval change. Separate frozen nonlinear outer probe
inference continues uninterrupted; source results are not external promotion.

2026-10-01 SOURCE group-bias/assignment diagnosis COMPLETEall47/fivefolds.
Two-way patienthash localmedian correction ALLarms: recurring>=30personmean
harms56process/69population;aftermatchedcentering35/45stillharm. Ofthese29/37
realoutputs beat expectedwithinhalfoutputpermutation: usefulindividualsignal
can coexistwithworseMAEthanconstant. Notindependentgroups/statisticalharm or
deployablecalibration. Lowmedianrho.191/.156 andIQRratio.289/.369notproofscale
bug/informationabsence.30tests pass. RECONSTRUCTION_SOURCE_ERROR_MECHANISM_20261001.md
+twoaggregateJSONs. Constantbiasonlyandnoindividualsignal explanationsboth
insufficient;encoder/readout/conditionalshift stillnotisolated. No runtime/
promotion/neuralweightchange/livejob;priorglobaloffset/affinefailuresretained.

2026-10-01 fullSOURCE10 ALL5 COMPLETE1797884queryoccurrences720470uniquepairs/
29580people47targets. Allroles/grids/hashes pass;monitorparity<=4.44e-16;18tests.
Patientmacro improves3/5,component4/5;16-26/47targets improveboth per fold.
Full-vs-monitor changedirections differ60/235patient33/235component;fullsource-
outer still63/233patient103/233component. Samplingnotsoletransferfailurecause.
Of26outerbasefailures17have>=30personsourcecomponent recurringmeanharmsvsone
control(13process15population),notstatisticallyprovedharm/independentreplication.
RECONSTRUCTION_FULL_SELECTION_FINDINGS_20261001.md +aggregateJSON. Sourcegrid
completionrepaired,notmodelgeneralization. No fitting/promotion/runtimechange;
alljobscomplete. Next sourceconditionalerror mechanism,notgeneric offset/CV repeats.

2026-09-30 full SOURCEselection10 folds0/1 COMPLETE363526pairs14845people47targets
EACH(samequerypatients,differentfits,not29690independentpeople). Fold0certified
exactouter2reuse;fold1newfullinference uninterrupted. Hashes/roles/keys/complete
supportchecked;bothfixedmonitor metrics reproduce<=2.22e-16.8tests pass.
RECONSTRUCTION_CONTINUATION_SELECTION_PROGRESS_20260930.md. Folds2/3/4stillpending
thenfullpairedsourceassessment;no partialefficacy/promotion/runtimechange/livejob.

2026-09-30 SOURCE warning/monitor fidelity COMPLETE/all47. Of26outerbasefailures,
18have>=30person sourcecomponent meanharm vspopulation,17vsprocess at8.17of21
recurringouterpopulation-harmtargets also show sourceharm. Meansignsnotproved
harm/independentreplications. Fullsource8 vssampled8 componentmargin signs differ
17/235process15/235pop;monitor8->10vsouter8->10changedirection differs103/233
component84/233patient. Two unsupportedoutercells explicitnull;10tests pass.
RECONSTRUCTION_CONTINUATION_SOURCE_20260930.md +aggregate. Fullsource10 still
uncomputed;smallmonitor notsubstitute,notselectioncause(fixedendpoint). Need
fullsource10pairedcomponents before nextsource-onlyrepair;no runtime/promotion.

2026-09-30 continuation CONTROL-DRIFT diagnosis complete/all47. New real versus
OLD process component screen retains chloride/fibrinogen/platelets, notPTT/uricacid;
oldreal versusNEWprocess passes latter2. MAPrelative margin improves whileactual
MAE worsens. Notnewfullgates orproof no realgain. Of26basefailures21have>=1
supported(30patients)component recurring meanharm versusFIXEDpopulation at8/10;
notindependentreplication/statisticallyprovedharm. Five lackthispattern(Hgb/
lactate/LDH/Mg/pH),notautomaticpowerdiagnosis.13tests pass. See
RECONSTRUCTION_CONTINUATION_MARGIN_20260930.md andaggregateJSON. No fitting,
runtime/promotion/intervalchange. Nextsource-only verificationofharm pattern,
notouter-derivedrouting;prior correction/weightingnegatives retained.

2026-09-30 continuation FULL OUTER COMPLETE5folds1800214pairs73670patients47targets.
All identity/role/scale/provenance/dependency checks pass;5synthetic tests pass.
Real targetmacro patient .920727->.917355,component .896169->.894102(~.37%/.23%).
Base descriptive screens17->21(retained16,lostbicarbonate,addedchloride/fibrinogen/
platelets/PTT/uricacid). OnlyALTpasses BOTH base and incremental47family screens:
patientMAE44.4342->43.5845;component33.0940->32.5182,CI[-.88475,-.26501].28/37
componentsimprove,9harm;leaveanyoneout meanstillnegative. Candidate only,
NOTfresh external evidence/forecast/interval/runtime promotion. Remaining26base
failures:21allmeansbetterbutuncertain;23population-component,21process-component
blockers(overlap). Notsolecauseproof. RECONSTRUCTION_CONTINUATION_OUTER_FINDINGS_20260930.md
and aggregateJSON. Fold4interrupted thenidenticalrerun completed;no livejob.
Latest sanitized epoch10 real weights(fits0/1/3/4;fold2equivalent0) uploaded to
PRIVATE GitHub prerelease coverage-reconstruction-20260930-epoch10-four-fits,
with matching source/scalers/usage/hashmanifest. Remote SHA256 verified; no
patient rows or optimizer. Candidate only; runtime unchanged.

2026-09-30 all fixed SOURCE monitors complete47x5epoch8/9/10real/process.
Fold2 certified equivalentfit0 at epoch10 but separate7167 selection queries:
real patientMAE .856151->.860299,component .792579->.795414 (worse).
Other4fitfolds realmacro improve slightly; folds overlap/notindependent studies.
Fold2 FULL OUTER new inference complete363526pairs14845patients,exactbasekeys:
real normalized patient .840887->.840640,component .825005->.824666 (tinygain);
process .892140->.896812,.876682->.882121 (worse).26/47bothrealmeansgain.
Monitor/full sign disagreement limits monitorproxy. Notsolecauseproof, no
promotion. RECONSTRUCTION_CONTINUATION_FOLD2_20260930.md. Remainingfullouter
folds0/1/3/4 and fixed paired47target assessment. Runtime unchanged/no livejob.

2026-09-29 fold3 BOTH monitored arms COMPLETE epoch10. Fixed7167queries47targets,
pairedRNG/pending/seen exact350083. Realpatientmacro .893425->.892299,component
.829798->.825874;process .942637->.945477,.876667->.878324. Real26/47patient,
28/47component improve; descriptive smallgain notuniform/notindependentfolds.
RECONSTRUCTION_CONTINUATION_FOLD3_20260929.md. Remainingfold4both,fold2explicit
certification andcompletepairedassessment. No runtime/promotion/releasechange;
no livejob.

2026-09-29 fold1 BOTH monitored arms COMPLETE epoch10. Fixed6862queries,
47targets,pairedRNG/pending/seen exact358799. Realpatientmacro .910917->.908001,
component .903163->.900541; process .973157->.973328,.967397->.967953.
Real24/47patient22/47component improved: macrobenefit notuniform. Folds0/1
overlap,notindependentreplications. RECONSTRUCTION_CONTINUATION_FOLD1_20260929.md.
Remainingfolds3/4both,fold2certification,fullpairedassessment. No runtime/
promotion/releasechange; no livejob.

2026-09-29 fold0 BOTH monitored arms COMPLETE epoch10. Exact paired RNG/pending/
seen sets358828,50384updates per extraepoch. Fixed6862sourcequery monitor real
patientmacro .911950->.908737,component .906847->.904325; process .961297->.965707,
.959165->.964630. Real29/47patient and28/47component improvements; notuniform.
Descriptive onefold,notpromotion orrootcauseproof. ProcessSmoothL1down butMAEup.
See RECONSTRUCTION_CONTINUATION_FOLD0_20260929.md. Remainingfolds1/3/4both,
fold2equivalence certification andpairedfullassessment. No runtimechange/livejob.

2026-09-29 latest weights upload COMPLETE. Fold0 real epoch10 finished with
fixed6862-query monitor; other continuation arms/folds stillpending. Published
GitHub prerelease coverage-reconstruction-20260929-fold0-epoch10: sanitized
actual weights, matching source archive, usage, manifest and47-target exact
prediction/scaler parity report. Isolated strictload passes; remote weight/source
SHA256 matches local. No patient rows, optimizer or cache uploaded. Current-value
reconstruction candidate, not future forecast promotion. Runtime41/27/14unchanged.
No livejob. Tag references main; use attached matching source, not auto source ZIP.

2026-09-29 monitored continuation wired,fold0real epoch8 monitor parity passed
6862fixedqueries. Epoch9 COMPLETE50384updates,immutablecheckpoint+monitor saved
under monitored_continuation_20260929. Targetmacro patientMAE .91194955->.90996007
(28/47better),component .90684662->.90708977(27/47better),SmoothL1 .60626979->.60433165.
Onefold/arm descriptive,no efficacyclaim. Fixed endpoint10,allfoldreal/process
stillrequired,fold2reusemustcertify. ProtocolRECONSTRUCTION_MONITORED_CONTINUATION_PROTOCOL_20260929.md.
Original8epochweightshashverifiedunchanged;no runtime/promotion/livejob.

2026-09-29 fixed SOURCE monitor implemented and epoch8baseline built. All47
targets5folds;identityhashmax32percomponent,6862/6862/7167/7167/7167pairs.
Strictselectionroles/privatekeys,duplicate/missing/numericchecks.4tests pass.
RECONSTRUCTION_FIXED_MONITOR_20260929.md and epoch8aggregateJSON. Future
continuation NOTRUN orwiredintotrainer yet;need predeclaredbudget and immutable
per-epoch checkpoints+samequeryevaluation. No retrospectivecurve/promotion.
No runtimechange/livejob.

2026-09-29 training-loss comparability audit COMPLETE4independentfits schedules
exactRNG/pending/seenreplay. Constantfitmedianloss fluctuates withonline modelloss
rho .699-.799;cannot infer convergence fromepochlog. Olderweights overwritten,
no retrospective fixedquerycurve. RECONSTRUCTION_COVERAGE_LOSS_COMPARABILITY_20260929.md
+JSON;objectiveparitytestpasses. Nexttraining needs prespecified fixedsource
monitor and checkpointedcurves;notproofmoreepochshelps. Existing8epochcontrast
immutable,no outerepochselection/runtimechange/promotion/livejob.

2026-09-29 groupvariance diagnosis complete.30failedtargets only9fullyestimable;
21include singletoncomponents,noiseunknownnotzero. Estimable9median noise/spread
.6766process/.5574population,notgeneralizableto30.18/19targets>=1supported
nominalharmgroup(process/pop),UNADJUSTEDsubgroupCIs notdiscoveries.
MAP2singleton;>=30persongroups0nominalharm17benefit eachref. Notnewpromotion.
RECONSTRUCTION_COVERAGE_VARIANCE_20260929.md +JSON;3tests. Distinguishgroup
precisionproblemfrompotentialdomainharms,avoiduniversalencoderfailureclaim.
No runtimechange/livejob; furtherrepair/source-onlyuncertaintyworkpending.

2026-09-29 COMPONENT INFLUENCE complete. Of30failedtargets29majoritycomponents
improvevsprocess,26negativeevenleaveanyoneout;4signfragile. Vspop27/24/2.
Notsoleluckysource explanation. MAP9smallgroupsmean+.38937vs27supported-.15183
vsprocess,23/36allimprovebutleaveonefavorablecanflipmean. DoNOTdeletegroups.
See RECONSTRUCTION_COVERAGE_COMPONENT_INFLUENCE_20260929.md +JSON;3tests.
Next withincomponent uncertainty vsbetweencomponentspread,allgroupsretained;
notprovepower ornewpromotion. Runtimeunchanged,no livejob.

2026-09-29 matched NUMERIC PATH COMPLETE47x5,29939sourcequerypairs(max128fixed
identityhash/targetfold). Exact originalpredictions reproduced; masks/ages/time
heldfixed; numericnormalizedzero. Neutralizationharms207/235coverage197/235old;
medianMAEincrease .048253/.040595. Numericpathactive,NOTtransfer/causalproof;
OODperturbation and overlappingpatients/fits. Existingouterconstraintsunchanged.
RECONSTRUCTION_COVERAGE_NUMERIC_PATH_20260929.md +aggregateJSON. No runtime/
promotion or live job. Next matched in-support perturbation/sourcecomponent
verification to distinguish useful numeric sensitivity from source reliance.

2026-09-29 coverage SOURCE support diagnosis47x5complete. Withincomponent
centered rankassociation numericadvantage vsobservedcount medianrho .02044,
fresh1h .01559,fresh6h .02175,meanage -.00317. Only17/12/13/8of235 absrho>=.1.
Weakaggregateassociations doNOTjustifygeneral freshness/supportselector;cannot
ruleoutvariable-specific/nonlinearsupport. Notindependent235experiments.
RECONSTRUCTION_COVERAGE_SUPPORT_FINDINGS_20260929.md;aggregateJSON.
Next inspectpriorperturbationtests then isolate frozenencoder numericvalueuse
with masks/agesheldfixed;nooutertuning. No runtime/promotion or live job.

2026-09-29 failure decomposition complete:47targets17screenpass25allmeansbetter
butuncertain5atleastonemeanfail(BNP/myoglobin/O2sat/troponinI/TSH).
Of30failures29process-component,23population-component,5population-patient,
2process-patient(overlap). Mainstatisticalblocker numericincrementacrosssource
components,notmostlypatientmean. MAP vsprocesspatient-.19663component-.01653
CI[-.15996,.15584]. Rootmechanism NOTproved; cannot infer merelylowpower.
RECONSTRUCTION_COVERAGE_FAILURE_DIAGNOSIS_20260929.md;aggregateJSON;3testspass.
Next source-only support/freshness vsnumericadvantage; no outerfit/thresholdchange.
No runtime/promotion change or live job.

2026-09-29: COVERAGE FULL OUTER COMPLETE all5folds1800214pairs73670people47targets.
Allroles/expectedidentities/labels/hashes and exact oldbaseline reproduction pass.
Patient targetmacro .925567->.920727;component .901157->.896169 (~0.5%gain).
Base descriptive screen13->17,all13retained;addedALP/bicarbonate/HR/osmolality.
Noneof4new screen targets passes incremental old-real comparison. OnlyMAPpasses
incremental47-family patient+component screen, but MAPfailsbasecontrols,6/36
componentsharm.20/47bothmeansimprove. Sampling matters but no uniformrepair;
processcontrol alsoimproves,notproofofstrongerphysiology. No promotion/runtime
change. See RECONSTRUCTION_COVERAGE_OUTER_FINDINGS_20260929.md and aggregateJSON.
Next diagnose which control blocksfailedtargets and incremental numericvalue.
No live process;fold4rerun completed after priorhandle disappeared/noartifact.

2026-09-28: Coverage OUTER fold3 COMPLETE380292pairs14754people,fourarms.
Verified test membership,uniqueidentity,finiteoutputs,SHA256 and exact original
real/process baseline reproduction including labels/scalers/population.
Private coverage_outer_20260927/fold3.parquet SHA256
73b144155ae07bc72de458ac9162a0701ff10c846da35552d074189f6ea7c87e.
Outer0-3complete;onlyfold4 pending before aggregate. No partial outer score
interpretation, no runtime/promotion change. Process exited0,no live job.

2026-09-28: Coverage OUTER fold2 COMPLETE via certified equivalent inference
reuse from selection0:363526pairs14845people. Allfour state_dicts exactlyequal,
fit scalers/medians/querypatientsets/sharedcache equal; original baseline paired
identity/truth/scale/population/predictions exact against prior outer2.
Source inference script has since gained role CLI; original digest retained as
provenance, all other source bindings and trained dependencies checked unchanged.
scripts/certify_coverage_outer2.py; private coverage_outer_20260927/fold2.json.
NOT independent training or inference replication. Outer0-2complete,3-4pending.
No outer score interpretation, promotion/runtime change or live job.

2026-09-28: Coverage OUTER fold1 COMPLETE356944pairs14735people,fourarms.
Verified test membership, unique identity, finite predictions, SHA256 and exact
original baseline real/process reproduction, including labels/scales/population.
Private coverage_outer_20260927/fold1.parquet SHA256
7647e1b0a4a27085816c6e156f7ea1852064352e7aed54833fd935770d53dd39.
Outer0-1 complete,2-4 pending. No scores interpreted or runtime/promotion change.
Process exited0; no live job.

Coverage OUTER fold0 COMPLETE380978pairs14759people,four frozen arms.
Verified test-only roles,unique identities,finite predictions,hash and EXACT
original replacement-real/process reproduction against outer_20260924/fold0.
Private coverage_outer_20260927/fold0.parquet SHA256
f8bbcb12ee685309244b2f3933282bca8f93cf7650041141be8f47cdce968832.
Same inference now accepts explicit --role test; selection files unchanged.
Outer1-4 pending; no outer scores interpreted or tuning/promotion/runtime change.
Process exited0, no live job.

Coverage SOURCE-SELECTION ALL5 COMPLETE. Fold3/4 each356944pairs14735people;
all5 hashes/roles/identities/finite values verified. Descriptive paired summary:
coverage patient macro improves3/5, component macro5/5; not significant passes.
Patient target improvements26/28/29/35/23of47. Folds overlap in patients/fits;
no independent replication claim. Sampling has effects, not a uniform repair.
See RECONSTRUCTION_COVERAGE_SELECTION_FINDINGS_20260927.md and aggregate JSON.
Next fixed outer inference/comparison; no retuning from selection results.
No live process, runtime/promotion unchanged.

Coverage source-selection fold2 COMPLETE:356944 pairs,14735 people,four arms.
Verified selection-only membership, unique patient-target identity, finite
predictions and output SHA256. Private coverage_selection_20260926/fold2.parquet
hash ce58230f60183b0da7dc67c68c5e7e6709af422ee7213ef7d62a557e00b5e867.
Fold2 selection differs from fold0, so inference was executed rather than
copying predictions despite certified identical fitted weights. Selection3-4
and outer comparison remain. No scores interpreted, no promotion/runtime change.
Inference exited0; no live job from this turn.

Last verified: 2026-09-27

CoverageSOURCESELECTIONfold1 COMPLETE363526pairs14845peoplefourarms.
Rolemembership/uniqueidentity/finitepredictionsverified. Samequerypopulation
asfold0butdifferenttrainedfit;notindependentpatientreplication. Selection2-4
andouterpending;noscoresinterpreted/no livejob/runtimechange.

Coverage SOURCESELECTIONfold0 inference COMPLETE363526pairs14845people,
4arms(newreal/process,oldreal/process). Alltraining/hashchecks passed. Exact
identity/label/scale/component+oldpredictions matcholdouterfold2(samefit0/2),
confirming pairedinputcontract. Noscoresreadyet;remainingselection1-4andouter
pending. Newprivatecoverage_selection_20260926/fold0.parquet. No livejob.

Coverage ALLTRAINING COMPLETE:fold4process8epochs403072updates305209seen
(loss.69662543). Verifiedall5foldreal/processsameepoch/updates/seen/RNG/pending.
8independentlytrainedarms64epochs+fold2certified16equivalentepochs;NOT10
independentfits. No heldoutaccuracyyet. Nextfixedsource-selectionandouter
pairedinference. No livejob/runtime/promotionchange.

Coveragefold4real COMPLETE8epochs403072updates305209seen(matches simulation),
fitloss.65003358. Onlyfold4processremainstraining. Noheldoutscore/promotion;
no livejob, runtimeunchanged.

Coveragefold3process COMPLETE8epochs403072updates290051seen(loss.62224223).
Real/processseen,RNG,pendingandbudgetverifiedexact. Onlyfold4real/process
remaintraining. Noheldoutevaluation/promotion/runtimechange;no livejob.

Coveragefold3real COMPLETE8epochs403072updates290051seen(matches simulation),
fitloss.5732823. Remainingtraining:fold3process,fold4real/process. Noheldout
efficacy/promotion/runtimechange;no livejob.

Coveragefold1process COMPLETED8epochs403072updates296623seen(loss.6118351).
Matchedreal/processseen/RNG/pendingverifiedexact. Folds0/1trained,2certifiedreuse;
3/4botharmsstillpending. No heldoutaccuracy/promotion/runtimechange;no livejob.

Coveragefold1real COMPLETE8epochs403072updates296623uniquepairs(matches
pretrainingsimulation),fitloss.56973137. Processfold1andbotharms3/4remain.
Noheldoutscore/promotion/runtimechange;no live process.

Coveragefold2reuse CERTIFIED:fitIDs/scalers/medians/all8orderhashes/RNG/pending
exactlyfold0;cache+allcheckpointbindingschecked. Copiedreal/process withexplicit
reusecertificate,newsha. NOTindependenttraining. Coveragefolds0/2ready;1/3/4
stilluntrained. artifacts/native_reconstruction_coverage_fold_reuse_20260925.json.

Coveragefold0process COMPLETED8epochs403072updates296652seen(loss.6273514).
Verified real/process identicalseen sets,samplingRNG,pendingqueues,epochbudget.
Fold0botharmscomplete;fullfivefoldexperimentstillincomplete,noselection/outer
efficacyyet. No live trainingprocess. Processweightsnotpublished/promoted.

Publicexport parity VERIFIED47source-selection histories(onepertarget):scalers
bitwiseequal,normalizedpredictions torch.equal privatevs sanitizedexport.
No truth/error scoring,notaccuracytest. artifacts/native_reconstruction_export_parity_20260925.json.

Coveragefold0real TRAINED8epochs403072updates296652uniquepairs,loss.5839704.
Noefficacyevaluationyet. Sanitizedstate-only12.64MBexportstrictloadsfromisolated
/tmpwithoutprivatedata. UserrequestedGitHubupload:PUBLISHEDprerelease
coverage-reconstruction-20260925-fold0(weights+matchingcodearchive+manifest+usage).
Originalprivatecheckpoint/seenindices/optimizerNOTuploaded. Runtimeunchanged.
GitHubtagpointsremote main;matchingcandidatecodeisexplicitsourceasset,notdefault
GitHubsourcezip. Otherfolds/processarmsremainunfinished. No trainingprocesslive.

Coverage training arm wired,NOTRUN:train_coverage_reconstruction.py preserves
originaltrainer/hashboundcheckpoints and sameoptimization;newisolateddirectory,
checkpointpendingcycle+RNG. CLIimport/help anddiffcheckpass. Fixed47targets5folds
real/process8epochs protocol RECONSTRUCTION_COVERAGE_TRAINING_PROTOCOL_20260925.md.
Noaccuracyresult yet;next execution startsfold0real with full8epochendpoint.

Coverage sampler implemented/tested,NOTtrained. Originalreplacementstreamexact
checkpointseen reproduction4fits. Sameupdatescoverageunique296652/296623/
290051/305209vs236509/236632/230194/243475(~25%more).3new5oldtests. Weighted
ordering changescyclefrequency semantics;notisolatedduplicateremovalclaim.
RECONSTRUCTION_COVERAGE_SCHEDULE_20260925.md. Accuracy remainsuntested.

Encoder exposure checked fromrealcheckpointseen sets:4independentfits cover
32.7-33.9%eligiblepairs;common-target~25-26%;2-4targetsfullyseen.8epochsare
fixedreplacementdrawsNOT8fullpasses. Encoderweightsalreadybounded. Fixedepoch8
notselectionbest;changingdrawlossnotconvergenceevidence. Noqualityclaim/newfit.
See RECONSTRUCTION_ENCODER_EXPOSURE_20260925.md;nextsourceprobe/matchedbudget
coveragecomparisonmustdistinguishsamplingfromsimplymoreupdates.

Weightresponse sensitivity:ESSratio vsrelativeMAEchange rho+.3421(source163,
descriptiveoverlappingfolds,noindependentp). Querysizeharms163/358small,
168/391medium190/454large;allmedianrelativechangesnegative. Harmsnotonlysmall
domains;weightingcontributesbutnotsolecause. See PATIENT_WEIGHT findings update.

PATIENTEQUAL correction contrast COMPLETE:fixedsourceLOCO47x5,163matched.
Vsbalancedwins109component/95patient;vsunchanged81component/75patient
(balanced55component SAME163).29beatbothatbothlevels;NOtarget5foldjointwin.
Smallqueries163/358harmvsbalanced. Weightingcontributesbutnotglobalrepair;
noouter/runtime/promotion.2tests. RECONSTRUCTION_PATIENT_WEIGHT_FINDINGS_20260925.md.

Correction TRAINING weight audit COMPLETE:231nonemptyroles223with<30person
components;equalcomponentweight medianKishESS/N.017181. Chloridefold1weight
ratio11908,MAPfold15630. Notencoderweights/notprovedinformationloss orrootcause.
Supportcheckdoesnotexcludetinyfitcomponents. Nextfixedsource-onlycontrast:
patientequalvscomponentequaltraining,identicalqueryscoring/noexclusions.
3tests,no runtimechange. RECONSTRUCTION_WEIGHT_FINDINGS_20260925.md.

Correction concentration audit COMPLETE:162matchedsourceCV;54meanwins,
43majoritywins36surviveanyonecomponentremoval,onlyHRfold0allcomponentswin.
Excluding<30querycomponents flips30/162means (sensitivity ONLY,not newgate).
MAPfivefoldbenefits survive these checks;chloridefold4doesnot. Noallreference
promotion established.2tests,noouter/runtimechange. See
RECONSTRUCTION_CONCENTRATION_FINDINGS_20260925.md. Distinguish average benefit
from distributed support;do not delete smallgroups to manufacture passing.

Point-only correction ablation COMPLETE:
SameHuber/sourceLOCO/weights;onlyfrozenpredictioninput,47x5,163complete,
162matchedfullHuber. Pointonlybeatsfull127,unchanged54(vsfull36),offset56,
allreferences15.97bothpoint/fullloseunchanged. Chloride/MAP5/5vsunchanged
butno5/5allreferencewinner. Supportsaddedcorrectionharm,notproofsharedencoder
orphysiologyuseless/soleoverfittingcause.2tests;noouter/runtime/promotion.
See RECONSTRUCTION_POINT_ONLY_FINDINGS_20260925.md. Universalcorrection
notjustified;retainbaseandsourceincrementalgates,notnewfeature/headsearch.

Numerical completion repair DONE:
Only3of160incomplete seen-domain targetfolds involved5nonconvergedfits.
Allfive300iteration failures reproduced, sameobjective warmstart+2700max
allterminateconverged;predictionslittlechanged,4/5stilllosebase. Original75
complete records EXACTunchanged;O2satfold0added=>76complete159supportlimited.
17bothwin8seenonly4unseenonly47neither;seenbetter61. No supportrelaxation/
outer/runtime/promotion.1newtest. Completionutilitysupportassertionfixed
beforeartifact,notoldmodelcause. See RECONSTRUCTION_NUMERICAL_REPAIR_20260925.md.

Seen/unseen MATCHED QUERY diagnosis COMPLETE/limitedcoverage:
Fixedsource20%patientholdout,seen/unseen equalfitrowcounts,samequeries,
noquerypatientfit.75/235fullysupported;160incomplete(notfail/success).
Seenbetterthanunseen60/75,but47/75bothloseunchanged;17bothwin,7seenonly,
4unseenonly. Notpureunseen-hospitalfailure;newpatientincrementalcorrection
alsofailswithseendomains. Case-mix/support/weightsnotmatched,onehashsplit,
descriptivemeansonly.2tests;noouter/runtime/promotion. See
RECONSTRUCTION_SEEN_DOMAIN_FINDINGS_20260925.md. Domainnormalizationalone
notestablishedrepair;requirewithin-source-newpatientincrementalbenefit.

Robust SOURCE-ONLY correction COMPLETE/notstable:
Huberfixed1.35/.1/300iterations,samefeatures/support/frozenstate,all47x5.
162completeconverged;numericbeatsunchanged36(vsridge25same162),ridge83,
allreferences13. ASTalonebeatsunchanged5/5butnotallcontrols;NOtargetall5
allreferencewinner.5nonconvergedcomponentrefits(amylase1/TSH0)explicit.
Prior138ridgeCVharms127majorityharm,122persistafteranyonecomponentremoved.
Notonesiteorloss-onlyrepair. No outerrun/runtime/promotion.2newtests+5prior.
See RECONSTRUCTION_ROBUST_SOURCE_FINDINGS_20260925.md. Furtherrepairneeds
sourcecomponent evidence,notanotherouter-drivenhead/relaxedgate.

Conditional snapshot correction COMPLETE/REJECTED:
All1800214outerpairs47targets;fixedsource-only weightedridgealpha.1,
process/actualnumeric context versus medianoffset/unchanged.0incremental
screens. Numericnormalizedmacro .925567->1.119574patient,.901157->1.117577
component;7/6meanimprovements only. EarliestfailureWITHINsourceLOCO138/163
supportedtargetfolds harm. Identicalfit replay reproducesouter;35/189fitMSE
improveswhileMAEworsens. Of138CVharms,27fitMAEalreadybad,111fitMAEimproves
thenCVworsens:lossmismatchnotsolecause. Five tests. No encoder/runtime/
interval/promotion;see RECONSTRUCTION_CONDITIONAL_REPAIR_FINDINGS_20260925.md.
Do not rerun outer-first readout sweeps;require sourcecomponent evidence.

Variable-identity residual diagnosis COMPLETE:
All1800214exactmasked snapshots,latestMEASUREMENT/knownrevision semantics,
privatehash+identityverified. All235targetfolds,6emptyroles;12137variable
comparisons,857observationfractionshifts>=.2. Similaraggregatecountsconceal
differentvariables(albuminfold2AG-.357,HR+.237). Withincomponent residual
correlations5160supported,1291signflipsbutonly4bothabs>=.1;531bothstronger,
527samesign. Descriptive/no significance/causalclaim. Notblanketdirection
reversalorprovenrepair. Sevennew+twopriorsupporttests;no headfit/runtime/
promotion. See RECONSTRUCTION_VARIABLE_TRANSPORT_20260925.md. Oldrawridge
negative retained;needsource-onlyconditionalvsprocess/interceptcontrast,
notouter-derivedfeaturedeletionorcountmatchingagain.

Support-stratified matching diagnosis COMPLETE/limited:
All235targetfolds,predictiontertilesx3supportmedians(up to24strata),136any
supportedoverlap;0retain>=80%bothroles. Albuminfold2retains13%/24%,MAPfold4
44%/46%;selectedresidualshiftpersistsbutnotwholepopulationdiagnosis. No
supportrelaxation/rowdeletiontoimproveclaim.3tests,hash+joinverified.
See RECONSTRUCTION_SUPPORT_MATCHED_SHIFT_20260925.md. No runtime/promotion.

Exactmasked observation support extraction COMPLETE:
All1800214pairs73670people47targets,payloadhash/fullouteridentityverified.
Outcome-free counts/freshness/historyspan;measurementageincludesknownlatency.
Albuminfold2cal/outer22.526/22.388variables,.802/.811hmedianlatestage(mean),
soaggregateolder/fewerinputsnotuniversalfailureexplanation. Unmatchedsummaries
notphysiologicalequivalence.2tests;no model/runtime/promotion. See
RECONSTRUCTION_ADDITIONAL_OBSERVATION_SUPPORT_20260925.md. Within-source-bin
supportmatching/variableidentitycomparisonremainsnext,notnewhead.

Prediction-bin shift diagnosis COMPLETE/all235:
Sourcepredictiontertiles,229decomposable/139everybinsupported.124/139absolute
within-binmeanresidualshift>compositionterm. NotMAE/median/causaldecomposition,
notproofagainstsamplingnoise/coarsemix. Albuminfold2highbinresidual-.212->+.088;
O2satfold3within+.96808vscomposition+.06444. Simplelow/highmixnotcomplete
explanation. No binfit/runtimepromotion.3tests;newreadonlypandaserrorfixed
beforeartifact. See RECONSTRUCTION_CONDITIONAL_OFFSET_SHIFT_20260925.md.

Source-CV offset activation COMPLETE/NOT PROMOTED:
Fixedcomplete>=4componentnegative/p<.05rule,all3arms. Real8/235active,
process5,pop0;5realcomponentgains3harms. Albumindisabled5/5butMAPalso5/5
despiteusefulunconditionalcorrection. Normalizedmacro .925567->.925203patient,
.901157->.900760component;0incrementalscreens. No thresholdretune/runtime/
promotion. Two tests. See RECONSTRUCTION_OFFSET_ACTIVATION_FINDINGS_20260925.md.
Simpleconservativeutilityswitchinsufficient;notonlyactivationrate,selected
correctionsstillfailtransport. Existing13reconstructionscreensunchanged.

Additional calibration source-CV transport diagnosis COMPLETE:
235targetfolds,163complete/72incomplete. SourceLOCOhelp->outerhelp42/harm21;
sourcenonhelp->outerhelp47/nonhelp53. MAPsource+outerhelp5/5;albuminsourceharm
5/5outerharm4/5. Thus support-only correction ignores available negative
source evidence, butsourceCVnotperfectselector. HRfold3correctwarning,fold4
falsehelp. Two tests;no activation/threshold/fittingofouter/runtimepromotion.
See RECONSTRUCTION_ADDITIONAL_OFFSET_TRANSPORT_20260925.md. Next prospective
source-onlycorrectionutilityrule,allarms,notanotherunconditionaloffset.

Multi-component POINT OFFSET repair COMPLETE/NOT PROMOTED:
All5calinference/235targetfolds,189fit46unsupportedzero. All235offsets
reproduced;calweightedMAEminimization/hashverified. Outer1800214unchanged.
Normalizedmacro .925567->.917067patient/.901157->.897123component;
29/27targetmean gains but0incrementalscreensagainstunchangedANDcorrected
controls. MAP13.26625->12.82157 showscorrectablebias,stillcomponentcontrol
uncertainty;HR14.1764->13.94545,incrementalp=.006843failsmultiplicity.
Albuminworsens,sodiumstilllosescontrols,bicarbcomponentharm. Notglobaloffset
solution. Newfirst-foldfamilyenumerationbugcaughtbeforefit,repaireduniontest;
nothistoricalfailurecause. See CALIBRATION_FINDINGS_20260925. No runtime/
promotion/intervalchange;reusedvalidationnotfreshlockbox. Pending below superseded.

Multi-component point-bias calibration STARTED:
Protocol20260924fixed47targets,weightedmedianoffsetforALL3arms,sourcecalonly,
>=30patientsin>=3componentselsezero. Notconformal. Fold2cal380978pairs14759
peopleequivalentreuseverified;fold0cal356944pairs14735peoplenewinferencecomplete.
Exactcalidentity/labels/component/finitechecks. Two role tests pass.
Folds1/3/4pending;nooffsetfit/outerrepairscore/promotion/runtimechange.
Outerdataalreadyviewed:reusedvalidation,notnewconfirmation. See calibration
progress20260925andprotocol. Previous "notyetstarted" calibration superseded.

Additional-domain failure localization COMPLETE/all47:
MAP harm notsingleoutlier:17/27supportedcomponents loseprocess, supported
delta+.10717;allleave-one-componentmeansharm. Sodium18/30supportedloseprocess.
HR/bicarb ordinary95%CIsnegative butfail47targetmultiplicity;leave-one-out
meansnegative. Glucose3/30supportedloseprocess butcomponentuncertaintyremains.
Mixedmedianbiasdirectionsdonotproveoffsettransport. No exclusions/fitting/
promotion/runtimechange. Two tests. See
RECONSTRUCTION_ADDITIONAL_DOMAIN_DIAGNOSIS_20260924.md. Multi-domain source
calibration stillpending;do nottreat allfailures as no signal or smallN only.

Additional-domain ALL OUTER EVALUATION COMPLETE:
1800214pairs73670people47targets38components, exactidentities/labels/hashes
verified.13descriptive screens:albumin,ALT,AST,bilirubin,BUN,calcium,creatinine,
Hct,INR,phosphate,K,totalprotein,WBC.21nonpasses improveallmeans butfail
component evidence;13have>=1nonimprovingmean. Normalizedtargetmacro .925567
real/.975469process/.978512population;component .901157/.945913/.955007.
Currentmaskedreconstruction ONLY;notforecast/calibration/runtimepromotion.
Overlappingfits+explicitfold2reuse,notfreshlockbox. No matchedold4vsnew13
improvementclaim. See RECONSTRUCTION_ADDITIONAL_FINDINGS_20260924.md.
Fivefold pending entries below superseded. Source-only multidomain calibration
notyettested;oldselectedcohortgap remains;rootcause notuniversallyresolved.

Additional-domain OUTER INFERENCE fold2 COMPLETE:
363526 pairs/14845 people; exact TEST identity/labels, finite predictions,
positive scales and all hashes verified independently. Threeof5 outer folds
complete:1101448rows44339people;folds3-4pending. Fold2 uses certified equivalent
fit weights, not independent training replication. No scoring/promotion/runtime
change. Supersedes twofold pending counts below; see outer progress document.

Additional-domain OUTER INFERENCE fold1 COMPLETE:
356944 pairs/14735 people; exact TEST patient-target matrix and raw labels,
finite predictions/positive scales and all dependency hashes verified.
Two of five folds complete (737922 rows/29494 people); folds2-4 pending.
No score interpretation, fitting, calibration or runtime promotion. Earlier
fold0-only pending counts below are superseded. See outer progress document.

Additional-domain OUTER INFERENCE fold0 COMPLETE:
380978pairs/14759people,exacttestidentity+labels verified,finitepredictions,
hashesvalid. Other4foldspending;no MAE/winnerinterpretationyet. Fixed47target
evaluationprotocol corrects old39denominator before outcomes. Rawunitsand
fit-scalenormalizedmacro separated. No runtime/promotion. See outerprogress.

Additional-domain ALL FIXED TRAINING COMPLETE80/80:
64executed+16certifiedequivalent. Fold4real/process finalfitloss
.6539388/.7108092,243475uniquepairs/403072updates each. All10checkpoints
verifiedepoch/fold,fit-onlyseen,pairedRNG/medians,finite,dependencyhashes;
fold2reusecertificatehashesmatch. No outerinference/calibration/promotionyet.
Next fulltarget outerassessment,notmoretraining basedonfitloss.

Additional-domain64/80STATES AVAILABLE:fold3both8epochs COMPLETE.
403072updates/arm230194uniquefitpairs;real/process finalfitloss
.5549454/.6073397. Seen/RNG/medians/bindingsmatch,allfinite,onlyfit.
48trained+16equivalent;fold4both andouterassessmentpending. No promotion/
runtimechange. See additionaltrainingprogress doc;old counts superseded.

Additional-domain56/80TRAINING STATES AVAILABLE:
Fold2certified identicalfit/scalers/8ordered schedules/medians/RNG tofold0,
reuseexplicit notindependentreplication. Fold3real8epochs403072updates,
230194uniquepairs,finalfitloss.5549454,finite/hash/fit-onlyverified.
40actuallytrained+16equivalent;fold3process/fold4bothpending. No outereval
or runtime/promotion. See additionaltrainingprogress/reusecertificate.

Additional-domain FOLD1 BOTH ARMS 8EPOCH COMPLETE:
403072updates/arm,236632uniquefitpairs;real/process finalfitloss
.5469040/.5939681. Seen/RNG/medians/bindingsmatch,allfinite,onlyfit.
32/80statescomplete;folds2-4andouterpending. Corrected prose:fold0/1fit
counts equal but15differentpeopleeach;2locations/6scalesdiffer. No weights
reused,partitionsunchanged. No runtime/promotion/generalizationclaim.

Additional-domain FOLD0 BOTH ARMS 8EPOCH COMPLETE:
403072updates/arm,236509uniquefitpairs(33.85%);real/process finalfitloss
.5558083/.6010830. Resumeunchanged;seen/RNG/medians/bindingsmatch,allfinite,
onlyfit.16/80scheduledstates;remainingfolds/outerassessmentpending. No
generalization/promotion/runtimeclaim. See additionaltrainingprogress doc.

Additional-domain MATCHED TRAINING STARTED/fold0epoch1both COMPLETE:
`RECONSTRUCTION_ADDITIONAL_TRAINING_PROGRESS_20260923.md`.47targets,
29542fitpeople/698766pairs;eacharm50384updates/43132uniquepairs(6.17%).
FitSmoothL1real.6253965/process.6330501,NOTefficacy. Seenpairs/RNG/medians/
bindingsmatch;onlyfit,allfinite,98statetensorsdiffer.2of80scheduledstates
complete;resumeoptimizer/RNG. No outerevaluation/calibration/runtime/promotion.
Protocol `RECONSTRUCTION_ADDITIONAL_TRAINING_PROTOCOL_20260923.md`.

Additional-domain FULL HISTORY CACHE COMPLETE:
`RECONSTRUCTION_ADDITIONAL_FULL_CACHE_20260923.md`. Original20.7hprocess,
105sites/1800214examples/73670people,4.92GiBSQLite. Allfivefolds zeroobserved
unsupportedinputs. Labels/masks/historycounts verifiedduringbuild;independent
fullpatient-targetidentitymatrix andlabels matchscalingcenters;allhashesvalid,
SQLiteintegrityok. No additionalhistorytruncation,norestart. No trainingor
promotion. Next matchedfreshphysiology/processtraining onfixedfour-rolefolds;
oldselectedcohortlimitunchanged. Previous fullcachependingnotes superseded.

Additional-domain TENSOR PREFLIGHT COMPLETE:
Lossless SQLite cache follow-up:847pilotexamples roundtrip exactly,2.15MB
databasevs12.77MBrawtensors;4235transformsrepeat.2cachetests pass inclatomic
rollback/read-only/duplicates/corruption. No truncation/downcast. Fullcache/
trainingpending;pilotcompressionnotpopulationestimate. Newcache reportbound.
`RECONSTRUCTION_ADDITIONAL_TENSORS_20260922.md`. Priorfixed38personpilot yields
847acceptedexamples/37people;rawlabels,fulltarget/siblingmasks,eventcounts match.
4235foldtransformsfinite/nonvaluechannelsidentical;nounsupportedinputsINPILOT.
12.77MBpayload;illustrativefull25.29GiBbeforeoverhead,notrepresentativeestimate.
EmptyacceptedsiteParquetfilterfixedwithoutresampling. Fullinputaudit/cache/
trainingstillpending;no runtime/promotion. Use rawsharedcache,notfivecopies.

Additional-domain fit-only SCALING COMPLETE:
Follow-up:12randommixed-source streams match reference acceptedlabelvalues;
9combinedtests pass;scaling/upstream code+eventindex bindings reverified.
`RECONSTRUCTION_ADDITIONAL_SCALING_20260922.md`. All1800214acceptedlabels/
73670people recovered;five fit-only patient-equal raw-unit median/IQR scalers.
Heldout mutation leaves parameters exactly unchanged. Six schema channels
unsupported in every fold. New extraction SQL conflict regression fixed before
use;five tests pass,not an explanation of historical failures. No training,
runtime or promotion. Next fit-only input support and masked history preparation.

Additional-domain independent ROLES COMPLETE:
`RECONSTRUCTION_ADDITIONAL_PARTITIONS_20260922.md`. All73944candidatepeople,
38components;5outerfolds,fit/selection/calibration/test fullyseparate;testonce.
Fit29542-29634people13-18components;calibration14768-14772people7-8components.
36/47targets>=30in>=3componentsEVERYrole/fold;11thin-domaininclaniongap/
cortisol/TSH/pH/myoglobin. Notaccuracy/promotion;alltargetexamplesretained.
Correction:fold0/1samefitCOUNTS,notidentities:15differentpeopleeach;
locations2/scales6differ;no checkpointreuse. Selectionidentical. Fold0/2fit
identitiesmatchbutfulltrainingequivalenceunverified. Three tests;hashes
andacceptancecountsverified. No training/runtime/promotion. Nextfit-only
vocabulary/normalization+maskedtraininghistories,matchedphysiology/process.
Oldselectedcohortgapunchanged;no newuntouchedexternalclaim.

Additional-domain FULL masked-history acceptance COMPLETE:
`RECONSTRUCTION_ADDITIONAL_MASK_FULL_20260922.md`. All1810110anchors73812people;
1800214accepted/5467no-fresh-label/4429no-independent-history.73670peoplewith
acceptedexamples47targets;46targets>=30in>=6components(myoglobinonly2).
105shardhashes+allbounddependencies+EXACToriginalqueryidentityverified.
99.4533%isDATA ACCEPTANCE,notaccuracy/coverage/promotion.13.6h originalprocess
continuedthroughlosttoolhandle,notrestarted. Broadmaskedhistorysupportresolved;
selectedcohortmismatchunchanged. No roles/training/runtime/promotion.
Next prospectivecomponent-disjointfit/cal/test+role/vocabularysupportbefore
matchedphysiology/process fitting. Earlier full-run-pending notes superseded.

Additional-domain masked-history PARITY PILOT COMPLETE:
`RECONSTRUCTION_ADDITIONAL_MASK_PARITY_20260921.md`. Fixedonepatient/component,
38people850frozenanchors;existingreconstruction_examples acceptanceandcounts
exactlymatchaudit:847accepted,2no-fresh-label,1no-independent-history. Existing
sourceversion/conflict replay reused;quality/latestdraw/age/target+siblingmask.
Three tests incl8randomstreamreferencecomparisons pass. Full1810110anchor
replay NOT run;pilotfractionnotpopulationestimate. No training/roles/promotion/
runtime. Fullmodeimplemented;nextcompleteallanchoracceptancebeforepartition.

Additional-domain candidate ANCHORS COMPLETE:
`RECONSTRUCTION_ADDITIONAL_ANCHORS_20260921.md`. All105lab+bedsidehospital
shards verified;73812people/1810110patient-targetanchors/47targets,zero
duplicatepatient-targets. Fixedhashpriority acrossallstays/sites,no numerical
values/outcomes/roles/insulinflags. Noredrawafterlaterrejection. Fourtests,
privatehashandpatient-componentidentitychecked. Proposednotaccepted:source
revision/conflict replay andtarget+siblingmaskednonemptyhistory stillpending.
No roles/training/accuracy/runtime/promotion. Prior selectedcohortlimitremains.

Additional-domain SELECTED-COHORT preflight COMPLETE/insufficient:
`RECONSTRUCTION_ADDITIONAL_COHORT_SUPPORT_20260921.md`. 73944candidatepeople
have127992rawinsulinrows/5394people,but127565rowsareml/hr;only32accepted
units/hrrows/15people/fivecomponents,ratechangesfivepeople. Zero>=30component
supportforselectedrenal/metabolic targets. Morelab/vitaldomainsNOTsamecohort
expansion. Oldselectedlineage alsofilteredpotassiumfuturepair availability,
ratechanges,inheritedrolesandMAP;notdiagnosis/preanchorphenotype. No future
values shown entering currentencoder bythisaudit. Whole-stayflagsprivate,
inference_feature_allowed:false. No guessedflow-doseconversion. Fourtests;
no training/runtime/promotion. Complete maskedhistoryacceptancenotrunyet.
Proceedbroadcohortwithexplicitroles;oldselectedexpansionremainsunsupported,
not silently replaced. Anyclinicalsubgroupmustuseverifiedpreanchorevidence.

Additional-domain BEDSIDE extraction COMPLETE:
`RECONSTRUCTION_ADDITIONAL_DOMAIN_BEDSIDE_20260921.md`. 40,865,099mappednumeric
rows ->40,775,882timevalid ->40,714,118qualityvalid ->35,697,038freshunambiguous.
70770people,5vitals each27-28components>=30people. Lab+bedsideunion73812people/
47variables;69775both. Empty-hospital pandas string-reduction/isfinite bug
reproduced/repaired in nursing helper;16tests and105shardhashesverified.
Not historical model-failure cause. No roles/training/runtime/promotion.
Next target+siblingmaskedhistorysupport,selectedcohortqualification,and
component-disjoint fit/cal/test design; no new untouchedlockbox claim.

Additional-domain canonical LAB extraction COMPLETE:
`RECONSTRUCTION_ADDITIONAL_DOMAIN_LABS_20260921.md`. 13,067,972mappedrawrows ->
12,458,563unitqualitycanonical ->10,070,389freshunambiguousrows;72817people
withfreshlab,42variables/41with>=6components>=30people. Creat70003/BUN69915/
glucose71312 each30supportedcomponents;albumin29/protein27. Sameunit/quality/
availability/1hlabelcontract. Privateignoredhashboundshards,3tests,no training/
runtime/promotion. Stillneedbedsideevents,targetmaskednonemptyhistory,and
selectedcohortqualification/roleassignment. Labtablezeroisnotvitaldataabsence.

Additional development-domain inventory COMPLETE:
`RECONSTRUCTION_ADDITIONAL_DOMAINS_20260921.md`. Fullraw139367patients/208siteIDs/
57components. Exclude22011knownpeople/71completehistorysites inclallroles,
ancestor,broad,confirmation,currentcrossfit;excludeentireconnectedcomponents
usingALLrawhistories.86447directdisjoint minus12503transitivebridgepeople=
73944candidates/110083stays/105hospitals/38components. Privateunassigned
provenanceonly;NOTvalidatedtargetsupport or untouchedhistoricaldata. Three
tests,no extraction/training/runtime/promotion. Next samecontracteventquality
and per-target/component support, preserve selectedcohortdefinition explicitly.

Multi-domain calibration feasibility COMPLETE:
`RECONSTRUCTION_CALIBRATION_DOMAIN_SUPPORT_20260921.md`. Current10737patients/
14hospitals/8wholehistorycomponents; each existing innercal only1component.
Selectedcreat/BUN/glucose/albumin/protein eachonly3components>=30patients
(broad8). Enumeratealllegal>=3cal/>=3fitcomponentpartitions preservingouter:
70/70/70/20/0choices;0/165targetfolds meet>=30pergroup/target in3components
eachrole. Prospectivefeasibilityrequirement,NOTnewpromotiongate/full-eICUlimit.
Four tests, no runtime/promotion/training. Next inventoryadditionaleligible
developmentdomains under samecohort/timecontract+reserved-roleexclusions;
do not rerunsameonesitecalibration or optimizewhichhospitalsusingerrors.

Calibration transport diagnosis COMPLETE:
`RECONSTRUCTION_CROSSFIT_CALIBRATION_TRANSFER_20260921.md`. All165targetfolds,
sourceoffsetsexactlyreproduced;86supportedbothsides,20oppositesignmedian,
38equalgroupharm/34patientharm. Retrospectiveoracle diagnosticONLY,nullwhen
unsupported. Bitwiseequivalentfold0/1 requirealbuminoffsets .17065vs.56512
inexchangedpopulations: notdifferentencoder/optimizerexplanation. Latestrepair
failsuniversalbiastransportassumption,notidentifieduniqueclinicalrootcause.
Five tests,no runtime/promotion. Check sourcecalibrationdomainfeasibilitybefore
anothertraining;do not deployoracle or tuneonoutermedian.

Independent crossfit offset repair COMPLETE/NOT PROMOTED:
`RECONSTRUCTION_CROSSFIT_OFFSET_CALIBRATION_20260921.md`. All33x5folds,frozen
encoders, offsets fit only preassigned encoder_tune patients; no testoverlap.
247611outerpairsretained. Macro1.673005->1.673893;selected1.874418->1.886968;
broad1.663383->1.664499.24/14/24numericgains all/selected/broad,0incremental
screensvsunchanged+process+population. Creatininepatientgainsboth,butselected
equalcomponent+.01509harm. Albumin/totalproteinregress. Fixedinterceptnotfull
repair;thin1-2sitecalibrationcannotassumedomainsupport. Three tests,no runtime/
promotion/livejob. No outer-derivedshrinkagesweep or cherry-picking24targets.

Frozen numeric ablation COMPLETE:
`RECONSTRUCTION_CROSSFIT_NUMERIC_ABLATION_20260921.md`. All10192creatinineouter
patients/5folds,originalpredictionsexactlyreproduce. Separatelyreplacebicarb/K/
Hgb numeric histories withfitlocation,preservemasks/times. All3worsenoverall
selected+broad. Adverse156personcomponent improves-.01862bicarb/-.07812K,
but stilllosesprocess+.08425/+.02476;Hgbworsens. Thus stablemarginalKrho
doesnotensureappropriatelearnedresponse;unstablebicarbrhonotsolecause.
No globalfeaturedeletion or subgroupdeployment. Two tests,no runtime/promotion.
PosthocnumericOODsensitivity,notcausal;conditionalstrength/interactions/calibration
remainunseparated,not a demonstrated source-trainedrepair.

Exact outer variable-relation audit COMPLETE:
`RECONSTRUCTION_CROSSFIT_VARIABLE_RELATIONS_20260921.md`. Same247610examples/
33targets, latestavailable values/anchorages, alltarget+siblingmasks asserted.
Creatinine adversecomponent: potassiumrho broad/selected .260/.264, butbicarb
-.264/-.039 withsame .767hmedianage;phosphate294broad/15selected support.
Thus equalcounts/agesnotmatchedfeatureidentity orrelations. Glucoseinputages
also differ byvariable;correlationsgenerallyweak/heterogeneous. Descriptive,
unadjusted/observedconditional,notmodelrelianceorcausalproof. Three tests;
no runtime/promotion. Next controlled numericablation before relationrepair;
do not selectfeaturesusingtheseouteroutcomes.

Exact outer observation-support audit COMPLETE:
`RECONSTRUCTION_CROSSFIT_OBSERVATION_SUPPORT_20260921.md`. All247610eligible
examples/33targets, same shards/fold masks, exact labels/prediction bindings.
Glucose selected/broad fresh1h variables2.95/6.28 despite longerhistory30.87/
26.31h; meanmedianage10.93/8.30h, labelmean differs. Creatinine adverse156person
component has similar support/history/labelmean and noolder medianage than its
2463person broad counterpart: freshness not universalcause. Descriptive input
summaries, not matchedphysiology or causalproof. Three tests, no model/runtime/
promotion. Do not globallydeleteoldhistory; further relationaldiagnosisneeded.

Fixed outer group-shift decomposition COMPLETE:
`RECONSTRUCTION_CROSSFIT_GROUP_SHIFT_20260921.md`. All33targets/hash-bound
predictions, exact parent means and symmetric decomposition. Against process,
19/33 positive within-component selected/broad terms,15/33 composition terms
(descriptive,not significance). Totalprotein selected equal-component harm
is thin-support sensitive: four adverse components all<30,largest has1patient;
leave-one-component range crosses0. Creatinine still harms a156patient selected
component;glucose harms167/152patient selected and1890patient broad components.
Thus neither site composition nor smallN explains all failures. Three tests;
no exclusions/fitting/runtime/promotion. Next inspect exact same examples'
input support/ages and label distributions within components, not another head.

Fresh encoder crossfit TRAINING+OUTER EVALUATION COMPLETE:
`RECONSTRUCTION_ENCODER_CROSSFIT_FINDINGS_20260921.md`. All5folds/2arms/8epochs,
72executed8equivalent reused.247611pairs/10717people,247610eligible. Realmacro
all1.673005 vsprocess1.748309/population1.726672;selected1.874418vs1.964726/
1.954685,broad1.663383vs1.738083/1.716379. All/broad4descriptivescreens:
albumin,BUN,Hct,totalprotein;selected0/intersection0. Selected15mean-improving
butnonsignificant/18atleastonenonimprovingmean. Nouniversaltransferfix.
Albumin/totalprotein remainmutualinputs(notindependentorganevidence).
Freshencoder independencefixedforfoldoptimization,notuntouchedexternaltest;
crossfitoverlapmakesinferenceconditional/descriptive.15tests,allcheckpoint
integrityverified,no runtime/promotion/livejob. Older pendingcounts superseded.

Encoder crossfit FOLD3 TRAINING COMPLETE:
`RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Real/process8fixed
epochs283008updates each,6299fitpeople;predeclaredunsupportedaniongapmask
retained. Finalsampledfitloss.841241/.911205,noouterassessment. Finite/RNG/
median/dependency/checkpointsha checks pass.64/80statesavailable(56executed,
8reused). Fold4andfullouterevaluationpending. No runtime/promotion/livejob.

Encoder crossfit FOLD2 TRAINING COMPLETE:
`RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Bothreal/process8fixed
epochs283008updates each,5513fitpatients,differentfoldscalers. Finalfitloss
.491907/.545704,notholdoutaccuracy. Finite/RNG/median/bindings/checkpointsha
verified.48/80statesavailable (40executed8reused),folds3-4andallouterevaluation
pending. No runtime/promotion;no livejobatstageclose. Do notcomparelossacross
folds withtheir differentnormalization.

Encoder crossfit FOLD1 AVAILABLE via verified equivalence:
`RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Fold1real independently
ran8epochs/283008updates; entiremodel/optimizer/RNG/log bitwiseequalfold0.
Sameordered4969fitpatients/scalers;test andunusedinnerselectionexchange.
Fold1process explicitlyreusesfold0process withzeroadditionaloptimization,
hashbound provenance.32/80armepochstatesavailable,24executed8reused;folds2-4
plusallouterevaluationpending. Two newtests. Notindependentmodelreplication,
notaccuracy/promotion/runtimechange. No livejobatstageclose.

Encoder crossfit FOLD0 TRAINING COMPLETE (supersedes stage1 below):
`RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Bothfreshreal/process
finish8fixedepochs/283008updates each,zero skips. Finalsampledloss.922825/
.999242;notheldoutaccuracy. Finiteweights/bindings/checkpointsha/finalRNG
matched,normalexit.16/80armepochscomplete;folds1-4plusouterevaluationpending.
Noepochselection,noexternal/runtime/promotionclaim. No live jobatstageclose.

Independent encoder training STARTED/stage1 complete (not efficacy):
`RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Fixed5folds/2arms/8epochs,
fresh samearchitecture real/process numericcontrol. Fold0both epoch1complete,
35376updates each,loss1.119969/1.131107 innewfoldunits; noouterevaluation.
Releaselearnedweightsnotloaded;11tests includingmutatedcheckpointinitialization
invariance. FinalRNG/median/bindingsmatched. 2/80armepochscomplete;78remain,
thenfullouterassessment. Epochboundaryresumable,no live jobatstageclose.
No runtime/promotion; cannotinfergeneralizationfromtrainingloss.

Encoder fold input scaling/support replay COMPLETE (not training):
`RECONSTRUCTION_ENCODER_FOLD_INPUTS_20260920.md`. Fit-onlypatientcenter/IQR,
unclippedaffinereplay/full10737people; masks/age/time preserved, observedzero
distinct. Fold2myoglobin absentfit/102queryexamples,fold3aniongap absentfit/
7269inputexamples+404unsupportedlabels. Fit-onlyvocabularymask repairsinput
contract;247610/247611trainedtargetexamples eligible,oneemptyhistorystays
abstention. Nine tests. No encoder/runtime/promotion changed. Pooledsource
contains these variables: not proofoldfullmodel lackedthem or universalcause.

Encoder crossfit feasibility COMPLETE (not encoder training):
`RECONSTRUCTION_ENCODER_CROSSFIT_PREFLIGHT_20260920.md`. Full10737fitpatients,
raw complete-history crosswalk:14hospitals form8components/44multisitepeople.
Five outer folds patient/site safe, exactly-once test; inner selection only
1-2sites.31/33patient-supported allroles;fibrinogen24test in2folds,troponin
zero selection/test in1fold each. Thus naive nested encoder selection remains
thin-domain; fix epochs prospectively or complete inner encoder crossfit.
Three tests/no runtime/promotion. Fold assignments private/hashbound; old
bounded7/caches cannot provide encoder-independent outer predictions.

Robust matched raw/body probe COMPLETE:
`RECONSTRUCTION_ROBUST_RAW_BODY_PROBE_20260920.md`. All33x3fits,same64features/
alpha100/n,onlySmoothL1lossreplacesquadratic;99converged. Bodymacro3.197/2.708
->1.851/1.574,raw1.901/1.611,original1.798/1.560. Raw0gates;BUNpatientmean
rawgain reversesat equal-site(.02447/.00570harm). Glucose/PTTbodybetterboth.
Two tests/no runtime/promotion. Lossconfoundrepaired,nouniversalrawinfoadvantage;
do notkeepreadout/lossvariants asuntestedsolecauses.

Current probe tail/objective diagnosis COMPLETE:
`RECONSTRUCTION_PROBE_TAIL_OBJECTIVE_20260920.md`. TroponinI/CPK account77.92%/
80.40%ofpositivebody-ridgeregression; CPK1.507%weightedsource mass accounts
98.60%meansquaredloss. Fitmean/medianconstantcontrols clarifylossdependence,
notuniformcollapseproof/PTTcounterexample. No clipping/exclusions. Two tests,
no runtime/promotion. Robustraw-vs-bodycomparisondistinctfromalreadyfailed
body-onlySmoothL1refits; do notinferallrawsignalabsencefromridge.

Raw-probe scaling warning REPAIRED/VERIFIED:
`RECONSTRUCTION_RAW_PROBE_SCALING_20260920.md`. All33x3probes,32warnings/
288negativevars allsource-exactconstant,min-9.7e-35,originalscalesfinite.
Exactconstant-onlyhandling+standardscalerothers removeswarnings;prediction
maxdelta1.44e-11/MAE2.03e-13. Notfailurecause;priorraw/bodyfindingsunchanged.
Two tests,no runtime/promotion;do notrepeatwarningasasignal-loss hypothesis.

Current dimension-matched raw/body probe COMPLETE:
`RECONSTRUCTION_CURRENT_RAW_BODY_PROBE_20260920.md`. All14150sourcepatients,
33targets,exactmaskedhistories/currentbounded7. Raw/processPCA64vsbody64,
samefixedridge100;rawnumericbeatsbody14/33eachgroup,zerojointgates. Glucose/PTT
bodybetterboth;BUNsmallrawgainnotstable. Probesmacromuchworseoriginal;not
proofrawsignalabsence(PCA/summary/MSElimits). Two tests,no runtime/promotion.
Weightedscalerwarningsoccurred;allreportedMAEsfinite,originnotfullyaudited.

Exact MAE site-constrained readout COMPLETE:
`RECONSTRUCTION_MAE_SITE_READOUT_20260920.md`.66HiGHSfits/all33, fixed.1L1
anchor/matchedobjective. Every sourceMAEconstraint passes(max~8.7e-12);
17changed16effectivelyoriginal. Heldselected1.797984->1.791458,broad1.559771->
1.552942;matched1.793823/1.553108. Zeroincrementalgates. PTTregressionreduced
notreversed;BUNbroadgain/selectedharm persists. Metricmismatchrepairedin
candidate,nottransfer. Two tests/no runtime/promotion; no morepenaltysweeps.

Site-constraint metric reconciliation COMPLETE:
`RECONSTRUCTION_SITE_CONSTRAINT_METRICS_20260920.md`. All692SmoothL1constraints
reproduce,but94site-target-groupMAEincreases(across45/66groups). Notsolecause:
fiveheld-harminggroups have no source-siteMAEharm;PTTselectedall8sourceimprove
yetheldpatient+.07566/site+.09878. Two tests,no fitting/runtime/promotion.
ExactMAEconstraint notrun; scipyHiGHSavailable,requiresmatchedobjectivecontrol.

Source-site constrained readout COMPLETE:
`RECONSTRUCTION_SITE_CONSTRAINED_READOUT_20260920.md`. Samebody33linearheads,
fixedstrength1+pergroup/siteSmoothL1nonregression.33solves12-19iterations,
allconstraintswithin7e-11. Heldmacroselected1.797984->1.788583,broad1.559771->
1.550221;matchedunconstrained1.789484/1.551793.21numericgainsbothvsoriginal,
15vsmatched;zerojointincrementalgates. PTTstillharmsboth,glucoseselectedsite
meanharms. Two tests,no runtime/promotion; sourceconstraintnottransferguarantee.

Current latent-support diagnosis COMPLETE:
`RECONSTRUCTION_READOUT_LATENT_SUPPORT_20260920.md`. All33/groupfit-only
standardization+LedoitWolf, fixedfit99%distance. Medianoutside9.52%selected/
1.50%broad.14/17patient-meanregressingtarget-groups alsoinsideharm atN>=30;
glucosebroad3084/PTT1317inside stillharm. Descriptive,notCI/promotion;
ellipsoidmembershipnotmatchedphysiology. Distance-onlyguardnotfullrepair.
Two tests,no runtime; encoder-independent relation stillunresolved.

Conditional nested readout CV COMPLETE:
`RECONSTRUCTION_NESTED_READOUT_CV_20260920.md`. All33 same bounded7cachedstates,
5outer/4innerhospitalfolds, fixed4penalties+unchanged. Ordinary selected mean
delta-.022519->nested-.006126;14numericgains17harms2unchanged. Glucose/PTT
stillnestedgainbutpriorencoder-unseenhospitalharm: selectionoptimismnotsolecause.
Encoder/anchor already saw these fit hospitals; NOT independentencoderCV.
Two tests, no runtime/promotion. Independentheadselection already tested0917;
fullencoder crossfit distinct, not merely repeating old head confirmation.

Readout CV transfer diagnosis COMPLETE:
`RECONSTRUCTION_READOUT_CV_TRANSFER_20260920.md`. Matched group/site weighting:
24 changed heads,18 mean gains/6 harms,12 numeric gains both groups; expected
CV delta-.030964 versus held-source-.001430, rank correlation.224 descriptive.
No common incremental gates. Inner CV conditional on already-trained encoder
and original head; optimism, representation reuse and shift not separated.
One test/audit pass,no runtime/promotion. Next selection must isolate encoder
fit from head-selection roles; do not tune exclusions on these outcomes.

Source-CV anchored SmoothL1 readout COMPLETE:
`RECONSTRUCTION_ANCHORED_READOUT_20260920.md`. Currentstate, samecapacity,
sourcefit5hospitalfolds select.01/.1/1/10orunchanged(9targets). Anchoring
reducesunregularizedregression: selected1.896874->1.817772(still>1.797984),
broad1.601832->1.556647(<1.559771). Gatesselectednone,broadBUNonly.
Two tests,no runtime/promotion; partialoverfitrepairnotcommontransferfix.
InnerCVconditionalonencoderalreadyfitonsource, notindependentendtoend.

Frozen readout SVD diagnosis COMPLETE:
`RECONSTRUCTION_FROZEN_READOUT_SPECTRUM_20260920.md`. Fixedbottom16/64fitSVD,
medianvariance.655%, mediannewweightnormsquared70.95%. Removeonlycenterednew
weakcorrection:20/33numericgainsboth, macroselected1.896874->1.882583,broad
1.601832->1.601027, bothstillworseoriginal. Partialconditioningcause, not
noise/collapseprooforuniversalrankfix. One test,no runtime/promotion.

Current frozen-state linear readout refit COMPLETE:
`RECONSTRUCTION_FROZEN_READOUT_REFIT_20260920.md`. Full14150sourcepatients,
247611fit/77219selectionpairs exactpredictions. Same64dimlinearhead, weighted
SmoothL1L-BFGS,33converged,onlyreadoutweight/biaschanged. Fit1.258742->1.190608
butselectionselected1.797984->1.896874,broad1.559771->1.601832;0/33gatesboth.
BUNmeanimprovesbutfails. Two tests,no runtime/promotion. Optimizationgapnot
sufficientcause; currentbodyfeaturescachedforconditioning/shift diagnosis.

Bias-polish source-held-site selection COMPLETE:
`RECONSTRUCTION_BIAS_POLISH_SELECTION_20260920.md`. All33,90/3323sourcepeople,
2470/74749pairs, verifiedfit/tunepatient+siteisolation. Existingbothgroup
incrementalgate:selected0,broadBUNonly,intersection0. Savedselectedcandidate
alltensorsexactoriginal; no accuracybenefit/promotion. Three tests. Universal
biasrepairrejected; stop offset/slope/gatetuning, reviewrepresentationevidence.

All33 bias-polish site diagnosis COMPLETE:
`RECONSTRUCTION_BIAS_POLISH_SITES_20260920.md`. BUNselected9/11sitegains,allLOO
negative,onlysiteevidencefails; broad7/7gains. Bicarbbroad5/7sitesharm,
equal-sitepositive, notjustpower. Creatbroad5/7harms,allLOOpositive.
Weighting flips13selected/6broadtargets. No gate/site changes or promotion;
distinguish insufficient evidence from opposing effects before furtherrepair.

All33 source-fit bias polish COMPLETE:
`RECONSTRUCTION_BIAS_POLISH_20260920.md`. Full10737patients247611pairs;
expectedSmoothL1biasoptima, onlyreadout.biaschanged, allotherweightsidentical.
Fitmacro1.258742->1.246812. ReusedevaluationselectedMAE2.909265->2.895306,
broad1.280861->1.278737;22/18numericgains,12both. Incrementalpatient/site
passesselectedbicarb,broadBUN,noneboth; notfullpromotion. SelectedBUN176also
averageimprovesbutsitep.0967, unlike90selection. Three tests, no runtime.
Trainingreadoutgaprepairedincandidate; ranking/generalizationunresolved.

BUN bias stationarity COMPLETE:
`RECONSTRUCTION_BUN_BIAS_STATIONARITY_20260920.md`. Expectedboundedweights,
equalgroupSmoothL1scalar optimumshift-.259876; gradient.193945->0.
FitMAE.889197->.833171,SmoothL1.517784->.492410,so metrictradeoffnotsoleoffset
cause. Broadselection7.28%gain,selectedCIcrosszero. Nonstationaryreadoutdata
risknotglobaloptimizerfailure. Two tests,no runtime/promotion. Fixedalltarget
source-fitbiaspolish isdistinctnextrepair; mustretainindependentgates.

BUN signed site residual/contract audit COMPLETE:
`RECONSTRUCTION_BUN_SITE_RESIDUAL_20260920.md`. Selected90:71at449,effective
sites1.55; median+.157CI[-.410,.498], no provenoppositebias. Broad3086same4
sites allnegative medianCIs, effective2.60. Cohortinsulinrestrictiondistinct
fromrawmeasurementanchor; hospitalrolesnonrandom. Broadbiasnotonesiteonly,
selectedconditionalshiftunproven. Auditexit0,no weights/split/runtime/promotion.

BUN fit-defined conditional residual audit COMPLETE:
`RECONSTRUCTION_BUN_CONDITIONAL_RESIDUAL_20260920.md`. Four pooledfitprediction
bins, broadselection allmedianresidualCIsnegative. SelectedselectionN11/12/19/48,
onlylast>=30CI[-.488,.464]; insufficienttoestablishoppositeconditionalregime.
Highestbin53.3%selectionvs28.2%fitselected. Composition/supportremain; no
outcome-derivedrouting. Two tests,no runtime/promotion. Do notclaim identical
states requireoppositeanswers orfitcorrections on90selectionpatients.

BUN affine-calibration/rank diagnostic COMPLETE:
`RECONSTRUCTION_BUN_AFFINE_TRANSFER_20260920.md`. Samefit/selectioncachedpairs,
medianaffine source-only. Fullslopes1.042/.922; extra slope vs offset has
oppositegroupbenefits, no transferablefix. Spearman.440/.384, outputSDsmaller
thantruth but notproofmiscalibration (conditionalmedian). Two tests,no runtime/
promotion. Do not keep escalating scalarcalibration onthese reusedoutcomes.

BUN source-fit offset transfer COMPLETE:
`RECONSTRUCTION_BUN_OFFSET_TRANSFER_20260920.md`. Fitmedianintercepts529/9664,
applybothsets90/3086disjointsource-selectionpairs, exactbaseline. Broadfull
MAE.893612->.822632/.820202 (7.94/8.21%gain); selected worsensnumerically,
CIcrosszero. Aftercalibratingbotharms bodyoff stillbroadgain-.01139/-.00873,
selectedharm+.03998/+.04378. Offsetpartlyexplainsnoteliminates dynamicsissue;
notuniquephysiologicalcause. Two tests,no runtime/promotion. Signedpredictions
cached; next diagnose remaining conditional error without outcome-tuned routing.

BUN integration-resolution diagnostic COMPLETE:
`RECONSTRUCTION_BUN_INTEGRATION_RESOLUTION_20260920.md`. Same90/3086source
pairs, fixed bounded7; steps1/.5/.25h, originalerrors reproduce1e-7.
Broad quarter-step MAE .893612->.893357 (0.0285%), selected CIcrosszero.
Much smaller than body-off4.751%; coarse stepping insufficient explanation.
Field itself uses substep input, not pure solver convergence. Two tests,
no runtime/promotion; do not repeat finer-step tuning as an untested repair.

Protected dynamics finite-step replay COMPLETE:
`RECONSTRUCTION_DYNAMICS_FINITE_STEP_20260920.md`. Fixed.01,2directionsand
equal-L2unprotectedcontrols; all33originalsource-selectiontargets,exactbaseline.
TinyBUNchanges,onlyprobe0broadnominalCIgain; no consistentselectedbenefit.
ProtectednumericalSmoothL1harms12-16/33pergroup. Not significance/promotion;
no probe/stepselectedafterresults. Four tests,runtimeunchanged. Representative
lossprotection unresolved; no repeatedtiny-stepsearch ontheseoutcomes.

Body-dynamics protected-descent probe COMPLETE:
`RECONSTRUCTION_DYNAMICS_DESCENT_20260920.md`. All66group-targettasks,up-to64
disjointfitpatients/probe. Convexprojection feasible,norm.533/.688; BUNpositive
descent onotherprobe, but27/66and20/66other-taskdotsnegative(vs31/28unprotected).
Not actualstep/accuracy, not robustlossprotection or impossibility. Three tests,
no weights/runtime/promotion. Source-selection finite-step response untested.

BUN weighted-objective reconciliation COMPLETE:
`RECONSTRUCTION_BUN_WEIGHTED_OBJECTIVE_20260920.md`. Exactsampler7selected/
8finalRNG/sets,3752BUNdrawseach. BodyofflowerMAE/SmoothL1underpatient,expected
boundedweights,andactualdrawweightsbothgroups. ActualdrawSmoothL1delta-.003992/
-.013471. Not solelysamplerweightmismatch; BUNcomponentnotwholemultitargetloss,
not convergenceproof. No training/runtime/promotion. Common loss-preserving
repair remains unproven; global deletion already rejected.

BUN fitting loss/exposure diagnosis COMPLETE:
`RECONSTRUCTION_BUN_DYNAMICS_FIT_20260920.md`.529/9664fitpairs; exact8epoch
sampler counts/RNG/seen sets reconstruct selected7. Broad2934actuallyseen BUN
pairs bodyoff improvesMAE-.021316andSmoothL1-.014481, bothdescriptiveCInegative.
Selected527seen SmoothL1CIcrosszero. Not solely externalshift or MAEobjective
mismatch; harmfulbranch exists on trainedpairs. Not convergence/gradientconflict
proof or draww eightedobjective. Four tests,no fitting/runtime/promotion.

All33 dynamics branch diagnostic COMPLETE:
`RECONSTRUCTION_ALL_DYNAMICS_BRANCHES_20260920.md`. Original90+90source,
2470/2025pairs,fullpatienterrorsreproduce. Bodyoffmacro1.797984->1.811577selected,
1.787841->1.796712broad;24/33and21/33numeric harms. No nominalbodyoffCIgain
inbothtargetgroups. Organoffmixed. Universalbranchdeletion REJECTED; BUNharm
notglobaldynamicsfailure. Four tests,no fitting/runtime/promotion. Remaining
conditionaldrift/supervision problem notsolecausedetermined; sixheads pending.

BUN body/organ dynamics branches COMPLETE:
`RECONSTRUCTION_BUN_DYNAMICS_BRANCHES_20260920.md`.90/3086source,normalfullerror
exact. Broadoffbody-.042459(4.751%gain),offorgan-.013330(1.492%gain),bothCI
negative; selectedbothCIs crosszero. Offbodyequalsbothoff exactlybycurrent
readout graph, not organuselessness. Priorno-dynamics mismatch max1.192e-7,
recorded numericalcheck only; no statisticalgatechange. Two tests,no runtime/
promotion. All33sourcebranch replay needed before globaldynamics trainingrepair.

BUN history/dynamics factorial replay COMPLETE:
`RECONSTRUCTION_BUN_HISTORY_DYNAMICS_20260920.md`.90/3086sourcepairs,fullforward
exact. Allobservations/no between-eventdynamics broad.893612->.851153(-4.751%),
deltaCInegative;selected1.105465->1.136537CIcrosszero. Afterdynamicsdisabled,
deletingoldobservations additional-.001424broadCIcrosszero. Originalclock,
initialization/updategate delta retained. Implicates learneddynamicsforbroad,
not universaldeletionrepair; bodyvsorgan driftunisolated. Focusedreplaytest
passes,no fitting/runtime/promotion. Next distinguish body/organ contributions.

BUN site/support localization COMPLETE:
`RECONSTRUCTION_BUN_HISTORY_STRATA_20260920.md`. Exact90/3086source pairs and
sample-site metadata. All4broad sites favor latestcoverage; equal-site sign
stillselectedharm/broadgain. Within449site+33events signreversal persists but
selectedCIcrosszero. Broad benefit mainlylong/repeatedhistories, alsofreshlatest
values. No site-only/length-only universalrepair; content versus elapsed-time
state evolution remainsunisolated. Three tests,no fitting/runtime/promotion.

Full BUN latest-coverage source replication COMPLETE:
`RECONSTRUCTION_BUN_FULL_HISTORY_20260920.md`.90selected/3086broad, exactold
patient replay. Broad full.893612->latest.857813(-4.006%), deltaCI[-.04246,-.02914];
additional3003replicate. Selected1.105465->1.157538(+4.711%, borderlineunadjusted
CIpositive). Morehistory notuniversallybeneficial, not a universal truncation
fix. Initialization/age/frequency/casemix remainconfounded; site/pre-anchor
support diagnosis before retention gate. Five tests; no fitting/runtime/promotion.

Latest-variable coverage control COMPLETE:
`RECONSTRUCTION_LATEST_COVERAGE_20260920.md`. Original90+90source-selection,
2470/2025pairs,33targets; full/last1 exactly reproduce. Latest-per-variable
MAE1.810225selected/1.781377broad vs full1.797984/1.787841; most last1 aggregate
loss removed, so coverage loss confounded prior memory interpretation. Repeat
history deletion still harms20/33selected,14/33broad; BUN opposite group signs,
creatinine CIs crosszero. Whole-event retention/initialization/distribution
confounds remain. Seven tests; no fit/runtime/promotion. Replicate BUN source
reversal on larger existing source before inventing a history gate.

History suffix diagnostic COMPLETE:
`RECONSTRUCTION_HISTORY_SUFFIX_20260920.md`. Original90+90source-selection,
2470/2025pairs,33targets; exactfullscore replay. Last1 worsens bothaggregate
scores, last8 mixed. Not total history non-use. Truncation also removes latest
variable coverage and changes initialization; latest-per-variable control next.
Raw-summary/body ridge already tested20260917, not a new untested remedy.
Four tests; no runtime/weights/interval/promotion changes.

Full source creatinine update trace COMPLETE:
`RECONSTRUCTION_CREATININE_FULL_UPDATE_20260920.md`.90selected/3095broad,
exactfinalsource scores. Updatechange-.0177CI[-.1223,.0845]selected;
-.07537CI[-.08927,-.06162]broad;original83alsoimprove. Earlier8patientharm
DOESNOTreplicate; do not remove lastupdate. No >=30broadpresent-variable
stratum has positivemean, exploratory/overlappingnotcausal. No fit/runtime/
promotion; finalupdateusefulness differsfrommodel-to-modelgeneralization.

Final state-update trace COMPLETE:
`RECONSTRUCTION_FINAL_UPDATE_TRACE_20260920.md`.525sourceencoderexamples,<=8per
target/group,finaloutputs reproduce1e-7. Lastupdate macro-.0463/-.1535benefit,
lastadvance+.0012/+.00485,finalgap+.00044/+.00034. Creatlastupdateharm bothsmall
probes,other renal/metabolic mixed; notsignificance/skipupdatejustification.
Intermediateanchortruth errors retrospective,notearliertimeforecasts. No fit/
runtime/promotion; expand sourcepaired/content diagnostic before intervention.

Full gradient reachability COMPLETE:
`RECONSTRUCTION_GRADIENT_PATHS_20260920.md`.66sourceprobes/33targets,bounded7;
99/2245requiresgradtensors connected. Allbody/organupdate,dynamics,message,
readoutandmixers receive nonzerogradients. Unusedfutureheads expected in current
reconstruction;one-stepbelief recurrence not fullpersistent disconnect. No
blanket deadpath supported,notgeneralizationproof. No updates/runtime/promotion.

Bounded-site frozen EVALUATION COMPLETE:
`RECONSTRUCTION_BOUNDED_SITE_EVALUATION_20260920.md`. Broaduniform1.277458->
bounded1.280861(+.266%),selected2.918501->2.909265(-.316%);17/33,20/33numeric
gains. Creatbothimprove,BUN/lipasebothharm,glucosetradeoff. Zero pairedpatient/
sitepassesboth;0alldiagnosticcomparisons.11bindingsverified,exactpairs/reference
scores. Source-1.14%doesnottransfer uniformly; boundedsite sampling notrepair.
Newcontrolsnotfit,33trained6pending;no runtime/promotion. Do not repeat asnew.

Bounded-site matched TRAINING COMPLETE, evaluation pending:
`RECONSTRUCTION_BOUNDED_SITE_TRAINING_20260920.md`. Bothuniform/boundedreplacement
8epochs283008updates,selected7both. Source macro1.813590->1.792913(-1.14%),
20/33numericimprovements eachgroup. Creat/BUNselected improvebutbroad worsen;
tradeoffs remain. Seven tests; no evaluation/promotion/runtimechanges. Freeze
selectedcheckpoints for identical-pair evaluation; oldcontrolsnotnewpromotion.

Bounded site sampler IMPLEMENTED/PREFLIGHTED, not trained:
`RECONSTRUCTION_BOUNDED_SITE_PREFLIGHT_20260920.md`. Source-count-only weights
projected [.5,2],mean1; five tests. Median site-massTV.139selected/.276broad;
min effectivepatientfraction.672/.677. Bound expected, notrealized repeats.
Matched uniform with-replacement control required; previous queues not enough
to isolate sites. Protocolfixed, no outcome-tuned weights/runtime/promotion.

Complete source-site preflight COMPLETE:
`RECONSTRUCTION_SITE_BALANCE_PREFLIGHT_20260920.md`. Selected3multi-sitepeople/
98pairs uniquelyrawmatched; all335780sourceexamples mapped, roles/shards checked.
Naive equal-site weights inflate singleton selectedpatients48.73x; broadtrig
52.39x(4people),directbili80.27x(3,untrained). Not measured harm but major
repetition risk; do not switch blindly to site-uniform sampler. Six tests,
no training/runtime/promotion. Bothgroup provenance nowresolved; source-only
bounded-influence weighting plus matchedcontrol needed before efficacy claims.

Broad-source sample site recovery COMPLETE:
`RECONSTRUCTION_SAMPLE_SITE_RECOVERY_20260920.md`. All45ambiguouspatients,
1286pairs exact tensor/time/truth/query replay incl4legacy shared-RNGpatients.
All45have targets spanning sites. Private sidecar retains sample stay/site/
anchor and13479singleton site mappings; no samples changed/excluded. Fourtests.
Provenance omission repaired, not error-cause/accuracy proof. Original selected
group site provenance still needs checking for two-group balanced training.
No runtime/interval/weight/promotion changes.

Source-only site alignment COMPLETE:
`RECONSTRUCTION_SOURCE_SITE_ALIGNMENT_20260920.md`. Fit10201people/11sites,
tune3323/4sites; unambiguous tune largestsite50.44%, effective2.55sites.
35fit/10tune multi-site patients lack per-target stay/site in cached samples,
excluded only from diagnostic site summaries, never arbitrarily assigned.
Weighting reverses3/33(albumin/chloride/magnesium), notcreat/lipase; thus not
sole tradeoff explanation. BUN4/4source sites improve,glucose3/4, descriptive.
Five tests,no fitting/runtime/promotion. Recover exact sample provenance before
site-balanced training; do not equate morepeople with moreindependent hospitals.

Continued site-weighting diagnosis COMPLETE:
`RECONSTRUCTION_CONTINUED_SITE_TRANSFER_20260920.md`. Frozen latest evidence
reproduces. Broad vs old32/33 favorable pooled/site means,22fail site evidence;
vs previous expanded10pooled/12site mean failures,all33site evidence failures.
Weighting sign flips8/33broad incremental,11/33selected-vs-old. Objective
weighting differs, not proved site-balancing cure or pure-power explanation.
Three tests; aggregate-only artifact; no runtime/promotion. Next source-only
site support/loss check before any reweighting, never evaluation-derived weights.

Continued expanded-source replay COMPLETE:
`RECONSTRUCTION_CONTINUED_SELECTION_RESULTS_20260920.md`. All0-19epochs on
3323broadsource,original90scores exact. Allscopes/equalgroupstillselect16;
32/33fullbroadnumericallybeatprior,notpromotion. Creat/lipase7->16means turn
harm withlargersourcebutpaired95%CIs crosszero bothold/new: unsupportedtiny
oldgains,not statisticallyprovedhiddenharm. Glucose/BUNfullsourceCIsnegative,
postselection/unadjusted. No alternativecheckpointrepair or runtimepromotion;
sixheadsremainpending. Newepochsource replay no longeruntested.

Continuation transfer localization COMPLETE:
`RECONSTRUCTION_CONTINUATION_TRANSFER_20260920.md`. Selected7->16creatinine
source slightlyimprovesbothgroups butevaluation worsensboth; lipasesamesigns.
Not simply sacrificedsource score; smallsourcegains/noiseshift unseparated.
Source/evaluation signs disagree15/33broad12/33selected, descriptiveonly.
Troponinreversecase; protectingtiny sourcegainsnotguaranteedrepair.2tests,
bindings/logsexact. No runtime/promotion. Expandedsourceepoch9-19replay remains
distinctnextdiagnostic, not newexternaltest;33trained6pending retained.

Complete-exposure continuation TRAINED/EVALUATED:
`RECONSTRUCTION_COVERAGE_CONTINUATION_RESULTS_20260920.md`. All3arms8->19,
672144updates each, old8logs exact/fullfinalbroadexposureverified. Source
selectsreal16/process6/placebo18: selectedrealnotfullcoverage(8576draws).
Broadmacro1.269761->1.265322(-.350%),selected2.911144->2.893127(-.619%);
23/33and25/33numericgains, glucosebothimproves/creatininebothharms.
BroadcoreAST/bilirubin/calcium,selected0;fullincremental0both.7tests,
bindingschecked.33trained6pending; no runtime/interval/future/promotion.
Moreexposurehelpspartly,notuniquecause/retentionsolution. Newepochs expanded
source selection remains separate; avoidevaluation-oracle target checkpoints.

Expanded source selection replay COMPLETE:
`RECONSTRUCTION_EXPANDED_SELECTION_RESULTS_20260919.md`. Allsavedepochs0-8,
3323sourcepatients incl90original/3233additional; oldscores reproduce1e-7.
Equalgroup best7->8only.0312%gain. Fullbroad31/33numericallybeatprior(epoch8),
trig/tropstillharm; not31promotions. Glucoseold90harm becomesfullbroadgain,
but8vs7BUN/creatharmand16/33targets worsen. Selectionpopulation sensitivity
confirmed, not soleNcause/accuracyrepair.2tests; no eval/runtime/promotion.
Fulltrainingexposure andsixpendingheads remain; source re-evaluation is no
longer untested. Bestlast epoch doesnotproveconvergence.

Complete-source scope reconciliation COMPLETE:
`RECONSTRUCTION_COMPLETE_SOURCE_SCOPE_20260919.md`. Actualdense exposure3752
selected/4288final vs~9500-10100available; sparse6trainedtargets fullyvisited.
Sixuntrainedtargets allhaveexpandedfit/tunesupport, butoldselectedfit3AG/15LDH/
23amylase insufficient for naiveequalgroupdraws. Originalbroadtune6fib/7lipase/
12CPK/14trig/18trop versusexpanded514/455/751/488/811. Hashcounts/roleisolation
verified;2tests. Supportnotaccuracycauseproof. Next source-only fullselection
epoch replay before complete-source training; no runtime/interval/promotion.

Score-versus-local-gradient replay COMPLETE:
`RECONSTRUCTION_SCORE_GRADIENT_20260919.md`. Topfive51.19%sourceMAE contribute
22.12%/20.65%gradientnormmass,19.11%/11.73%signedmeanprojection in savedtwo
8patient probes. ExactGram-dot reconstruction verified. Not majority local
gradient domination or fullAdam/encoder causalityproof. Six tests; no fitting,
newweights/runtime/promotion. Do not set target weights from smallprobe ranks.

Source selection concentration audit COMPLETE:
`RECONSTRUCTION_SELECTION_CONCENTRATION_20260919.md`. Sixsource trajectories
reproduce; AST/CPK/ALT/bilirubin/lipase contribute50.20%-51.88%ofmacroerror.
But leave-one-target-out changes0/33epochs in5/6arms; relativeerror selection
stillchooses7expanded/5moment. Not score-rescaling-alone repair or proof of
training gradient domination. Five tests; no newselector/evaluation/fitting,
runtime or promotion. All33trained/sixpending retained.

Moment-clock training/evaluation COMPLETE:
`RECONSTRUCTION_MOMENT_CLOCK_RESULTS_20260919.md`. Same accumulated setup,
only betas changed to(.9**528,.999**528). Source selects5(score1.816181), but
broadmacro1.279049->1.285663 andselected2.893760->2.927830 worsen.13/33targets
numerically improve each,0reportedcomparisonpasses. Glucose slightly improves
both; renal/AST/troponin tradeoffs persist. This specific clock repair failed,
not proof of unique cause or alloptimizer futility.33trained6pending; reused
cohorts/no matchednewcontrols/future/interval/promotion; runtime unchanged.
Supersedes moment-clock-untested statements below. Do not repeat as untested.

Decay-only accumulation trial COMPLETE:
`RECONSTRUCTION_DECAY_RESULTS_20260919.md`.8epochs,same528blocks/sqrtLR/source
order; onlywd.01->.229722. Selected7(source1.820250vs1.817846). Broadmacro
1.279049->1.280008,selected2.893760->2.907537: noaggregate repair. Decayclock
difference exists but matchingnotmissing solution. No runtime/interval/promotion;
33trained6pending. Momentclock/stepcount remainunisolated,notprovedcauses.

Accumulation optimizer-clock audit COMPLETE:
`RECONSTRUCTION_OPTIMIZER_CLOCK_20260919.md`.ActualAdamW(.9,.999),wd.01;
momenthalf-life grows528xperexample; decay-only factor sequential.753514 vs
sameLR.999464/sqrtLR.987759.81tensors283008steps,18have276357;accum99all536.
2212no-state notprovenbug. Exactdecaymatchinghelper testedrealAdamWzero-grad;
notfullAdam equivalence or causalerrorproof.4tests,no training/runtimechange.
Next separate decay-clock intervention from moment-clock adjustment.

Balanced accumulation intervention COMPLETE:
`RECONSTRUCTION_ACCUMULATION_RESULTS_20260919.md`.2real8epoch fits,283008same
examples but536Adamsteps; sourcechoosessqrtLR epoch7. Broadmacro1.269761->
1.279049harm,selected2.911144->2.893760gain;9/33,16/33numericimprovements.
Glucosebothimproveandselectedbeatsprioragain;BUNbothharm.0reportedcomparison
passes;matchingaccumulatedplacebo/processnotfit,nopromotionclaim.4tests,
currentreconstructiononly;notpurevariancecausality(steps/LR/decaychanged).
33trained/6pending;runtimeunchanged. Do not repeat exactaccumulationasnew.

Gradient sample-size reliability COMPLETE:
`RECONSTRUCTION_GRADIENT_SAMPLE_SIZE_20260919.md`.Disjointnested8/16/32/64patient
sourceprobes; common63tasks cosine median.024/.075/.204/.372. Negativepairs
32/26/21/14(all supported). At64 no repeatedlynegative crossgroup target among
30supported;selectedCPK/lipase/tropunsupportedat64,notpassed. Localgateonly,
onecheckpoint/split;notfullencoder/Adamoraccuracyproof.4tests,no modelupdate.
Next distinctrepair: controlled balancedpatientgradientaccumulation,not small-
probe gradient surgery. Must control sampleexposure and optimizerstep differences.

Trained body-update gradient probe COMPLETE:
`RECONSTRUCTION_TRAINED_GRADIENTS_20260919.md`.8320bodygateparams,all33targets,
2groups,2disjoint8patientprobes.4crossgroupconflicts repeat; glucose notopposed,
troponin stronglyaligned. Same-task crossprobe median cosine.098,26/66negative:
smallprobe instability prevents universalgradientconflictclaim. LocalSGDgeometry
notAdam/fullencoder/intervention.2tests,no update/runtimechange. Need source-only
sample-size reliability before gradient surgery,not anotherunjustifiedarchitecture.

Expanded checkpoint feasibility COMPLETE:
`RECONSTRUCTION_EXPOSURE_SELECTION_20260919.md`.Onlyepoch0nonnumericallyharming
all66; expandedtrainedepochs harm12-17,small12-18. No other commoncheckpoint
dominates selected7expanded/8small.10expandedgroup-target transientgainslost;
2neverbeatprior. Selectionalonecannotretainallobservedgains; notimpossibility
or statisticalnoninferiorityproof. Prioruniform/selectedoutputretention already
traded electrolytebenefitforrenalharm.2tests,nooracleheads/runtimechange.

Paired expanded retention localization COMPLETE:
`RECONSTRUCTION_EXPOSURE_RETENTION_FIT_20260919.md`. Bothmodels same1252source
patients,132encoder scores exact. Expandedselectedglucosefit1.213390 losesprior
1.198364; notonly generalizationfailurefor thischeckpoint. Epoch2glucose gains
inbothsourcegroupslost atglobalselected7. OtherAST/troponinfitworsebuttunebetter,
BUNfitworsebutvalidbetter: no universalfit-retentioncriterion/gradientproof.
Broadoldfitsubsetnotfullnewexposure. Source-only target-retention repair remains
untested here; no oracleheads, evaluationretuning or runtimechange.

Paired exposure error decomposition COMPLETE:
`RECONSTRUCTION_EXPOSURE_INCREMENT_20260919.md`. All66pairedscores reproduce.
Broadtotal-.048489=prior-.004524+conditional-.043965;selected+.012255=
prior+.004615+conditional+.007640. Notcausal. Selectedglucoseprior improves
but conditionalerror+.105779 worsens; AST+.372038,troponin+.212687. Broad
troponin/lipase improved butstillloseprior; ALTgainmainlyprior unlikeAST.
Remainingtradeoff notuniformoffset; source-only fit/selection followup warranted.
No fitting, routing, gates or runtimechanged; all39scope retained.

Matched small/full source training + frozen evaluation COMPLETE:
`RECONSTRUCTION_SOURCE_EXPOSURE_RESULTS_20260919.md`. Six8epoch fits,
283008updates/arm; selected8/2/0small,7/6/8expanded. Same queues/budget and90+90
encoderpatients. Broadmacro1.318251small->1.269761expanded(-3.68%);selected
2.898888->2.911144(+.42%).26/33broad,16/33selected numeric improvementsvs small.
Original coregate passes ALT/AST/BUN onbroad,0selected; added small-model
incremental comparison yields0fullpassesboth. Report separately,not all0core.
33heads,6untrained; densecoverage<=4288patients.9tests; source diversity helps
averages but retention notsolved. Reusedcohorts/no external,future,interval or
runtimepromotion. Supersedes paired-training-pending entries below.

Expanded source loading and queue preflight COMPLETE:
`RECONSTRUCTION_SOURCE_STORE_20260918.md`. Real shard hashes/identities and nested
roles verified; expanded broad fit10201/234618pairs (original33targets), fixed
selected536fit90encoder and broad90encoder. Bounded16-shard LRU,7tests pass.
Initial new-loader embedded-role check corrected to authoritative nested split;
not evidence of old model failure or untouched evaluation. Queue replay sees
min(N,4288)patients/target. Paired training NOT started;runtime unchanged.

Full broad-source sample expansion COMPLETE:
`RECONSTRUCTION_FULL_SOURCE_SAMPLES_20260918.md`.10201fit/3323encoder patients,
240729/79001acceptedpairs;all39>=30bothroles. Lipase1239/455,CPK2605/751,
troponin2325/811. Old626shards unchanged;newhashseededonepair/target,no redraw.
Basic support only,notaccuracy/sitegate. Trainingpending,runtimeunchanged.

Joint exposure replay COMPLETE:
`RECONSTRUCTION_JOINT_EXPOSURE_20260918.md`.Exact283008draws,all66groups covered.
Broadlipase46pairs averaged93.22uses,glucose504averaged8.51;source diversity
differs despiteequaltarget effort. Notcausaloversamplingproof. Fullsource
sample expansion stillpending;no scope/gate/runtimechange.

Joint error geometry COMPLETE:
`RECONSTRUCTION_JOINT_ERROR_GEOMETRY_20260918.md`.Allsourceincrements reproduce.
SelectedCPK rank.714->-.110,lipase.862->-.080;smallN30/28. Centraltruthrange
alsoharms,not tail-only. Broadtroponin retainsrank.660(N18)despiteMAEharm:
notuniversalorderingfailure. Retrospectivetruthstrata notroutes;no runtimechange.

Joint fit/selection replay COMPLETE:
`RECONSTRUCTION_JOINT_FIT_GAP_20260918.md`.All66selection scores reproduce1e-7.
32/33fit targets beat population inbothgroups;all9selected/8broad selection
failures improvefit. Source generalization gap confirmed before externalgate,
not universal inabilitytofit. O2fitonlyexception improvesselection. Sparse
support/site shift confounduniquecause;no memorizationproof/runtimechange.

Joint constant-loss diagnostic COMPLETE:
`RECONSTRUCTION_CONSTANT_OBJECTIVE_20260918.md`. Equalgroup sourcefit median vs
SmoothL1constant,all33targets. Encoder macro Huber-minus-median-.010736selected,
+.011093broad (average+.000178). Mixed effects,notlargeuniformbias explanation;
conditional neural effects nottested.3tests,source-only,no runtimechanges.

Retention-selection feasibility COMPLETE:
`RECONSTRUCTION_RETENTION_FEASIBILITY_20260918.md`. Onlyepoch0numerically
nonharming all66group-targetpairs;everytrained epoch harms some. Selected
CPK/glucose/lipase/troponin andbroadCPK/temp/troponin neverbeatinitial across8.
Other5pergroup improve transiently then lose atselectedepoch6. Selectionalone
cannot repair observedtrajectory;notimpossibility/gradientcauseproof.3tests,
nooracleheads/gates/runtime changes.

Joint source-selection trajectory audited:
`RECONSTRUCTION_JOINT_SELECTION_20260918.md`. Exactepoch6aggregate replay;
9/33selected and8/33broad targets already lose epoch0prior on encoderselection,
despite macro gainsboth. Aggregate objective permits harm, not gradient-conflict
or solecauseproof. No eval-informed choice, pertargetoracle or runtimechange.

Joint gate localization COMPLETE:
`RECONSTRUCTION_JOINT_GATE_DIAGNOSIS_20260918.md`. Core gate(excluding additional
broad-only comparator)still0. Site evidence fails>=1reference33/33bothcohorts;
actual pooled mean failures9broad16selected,site-mean failures11/19.
22broad11selected source-supported targets have favorable meansallcore, not
significantpasses. Evidence limitation and true numeric harms coexist; no
solepower/heterogeneityclaim.3tests,no gates changed,runtime unchanged.

Joint-source training/evaluation COMPLETE:
`RECONSTRUCTION_JOINT_SOURCE_RESULTS_20260918.md`. Three8epoch arms selected6/1/0,
283008updates/arm. Broad macro1.365619old->1.305363joint,selected2.934232->2.912258.
30/33broad,19/33selected pooled improvements; average retention harm mitigated,
not per-target solved(glucose/bicarb/O2selectedstillharm).0reportedfullpasses.
Doublepatients/compute confounds isolated mixturecause; repeated/external/future
validation pending.5tests;all39statuses/runtime unchanged. Supersedes pendingbelow.

Joint-source manifest VERIFIED, training pending:
`RECONSTRUCTION_JOINT_SOURCE_PROTOCOL_20260917.md`. Disjoint selected+broad
536fit/90encoder each,1072fit180encoder total. Shard hashes/schema/identity and
exclusion of originalheldout/expandedconfirmation checked. Per-patient histories
unchanged; group tags sampler-only, no retrospective inference selector.
Fixed8epochs33targets/threecontrols and both-cohort evaluation predeclared.
Double source/compute not isolated mixture effect. No efficacy/runtime claim.

Broad-source old-cohort retention COMPLETE:
`RECONSTRUCTION_BROAD_RETENTION_20260917.md`. Fixednew weights,178oldheldout
patients4637pairs;28/33numerically worsen,macro2.934232->3.007420(+2.49%),
versus expandedconfirmation28/33improve(-4.18%). Measured distribution tradeoff,
not universalrepair; individual significance not implied. No overlap/retuning,
all39statuses/runtime unchanged. Retrospective cohort rule cannot be a router.

Broad-source prior/increment decomposition COMPLETE:
`RECONSTRUCTION_PRIOR_INCREMENT_20260917.md`. Exact83874pair identity and3tests.
Creatinine total-.07006:prior-.00474/increment-.06532;BUN-.05320:
prior-.01364/increment-.03956. Not just population-center improvement.
O2center improvesmorethanmodel;glucose excesserror shrinks but still losescenter.
Descriptive algebra, not causal attribution or gate. Fullpasses remain0;
all39statuses/runtime unchanged. Different residual failure patterns established.

Matched broad-source comparison COMPLETE:
`RECONSTRUCTION_MATCHED_BROAD_RESULTS_20260917.md`. Three8epoch arms select5/2/1;
83874identical pairs,28/33lower pooled MAE,macro1.365619->1.308580;0fullpasses.
ALT/AST/calcium/platelets beat non-old controls but notold at corrected gate.
Glucose improvesold but losespopulation/process/placebo. Cohort repair helps
averages, not solecause/universalrepair proof. Evaluator cached-model alias
fixed before scoring;oldMAEs reproduce. Retention/repetition/external pending.
Runtime unchanged. Supersedes matched-training-pending entries below.

Equal-patient broad source examples COMPLETE:
`RECONSTRUCTION_MATCHED_BROAD_SAMPLES_20260917.md`.536fit/90encoder patients,
same representative-site quotas via outcome-blind hashes;626shards,12441/2113
pairs(all39). Two matchingtests pass. Target support differs(lipase encoder7,
CPK12,trig14,troponin18,PTT27), preserve unsupportedstatuses. Counts/site quotas
not complete case-mix matching. Matched training pending; fullsource pool retained.
No runtime/interval/promotion change.

Broad source event extraction COMPLETE:
`RECONSTRUCTION_BROAD_SOURCE_EVENTS_20260917.md`. Treatment-independent extra
fit10201patients/5721709events;encoder3323/2487270events. Same rawlab/bedside
contracts as expandedconfirmation,15private hospitalshards,14tests pass. COPY
binding failure repaired before eventoutput; rerun complete. This repairs source
selection at extraction only; samples/training pending. Preserve equal-size
comparison to separate composition from N. No runtime/interval/promotion change.

Inherited cohort-selection mismatch VERIFIED:
`RECONSTRUCTION_COHORT_SELECTION_20260917.md`. Original rawrate replay:all569fit
stays/536people and all93readoutstays/89people have qualifying insulin-rate changes.
Expanded confirmation only125/4131stays,118/3433people. Old reconstruction lineage
inherits insulin/potassium-selectedstay universe; expandedconfirmation doesnot.
Last expansion confounds N and case selection, NOT isolated power experiment.
Concrete design mismatch, not soleerrorcause/allreleaseclaim. Need matched broad
source development (fit/encoderroles) under identical extraction, keep evaluated
labels out of tuning. No newtraining/runtime/promotion yet.

Expanded fixed-head confirmation SCORED:
`RECONSTRUCTION_EXPANDED_CONFIRMATION_RESULTS_20260917.md`.83874pairs/33trained
targets,28fixednominees,all39statuses.0confirm unchanged39family patient/site gate.
BUN0.97015->1.04924,bicarb0.95603->0.98252,WBC1.08922->1.13370:actualharm,
not just smallsample nonsignificance. O2sat0.94262->0.85451 improves but site
p0.00550fails correctedthreshold. Seven sites unchanged; expandedpopulation may
differ from oldqueryselectedcohort. No uniquecause, new fitting or promotion.
Sixhelpertests;exact sample coverage checked. Runtime unchanged.

Expanded confirmation samples COMPLETE:
`RECONSTRUCTION_EXPANDED_CONFIRMATION_SAMPLES_20260917.md`.3433patients,
88728proposed/88232acceptedpatient-targets;496rejected without redraw. All39
targets represented,36>=3sites>=30patients;amylase/fibrinogen2sites,LDH1.
Existing conflict/availability/48hhistory/blockmask rules,9tests pass. No model
scored or fit yet; existing nominees must stayfixed for confirmation. Original
fithead support still applies. No runtime, intervals or promotion change.

Expanded confirmation events EXTRACTED:
`RECONSTRUCTION_EXPANDED_CONFIRMATION_EVENTS_20260917.md`.3433additionalpatients/
4131stays,2154036canonical lab+nursing events in durable Gitignored cache.
Five bedside targets nowincluded; invalid nursing observations quality-marked.
14replay/bedside tests pass, input/code/cache hashes recorded. No target examples
or scoring yet: simultaneous conflicts, blockmask and independenthistory acceptance
pending. Confirmation source only, no evaluation/reserved fitting or promotion.

Expanded source-lab support preflight COMPLETE:
`RECONSTRUCTION_EXPANSION_LABS_20260917.md`. Additional role-contained patients,
existing name/unit/bounds and<=1h conservative availability delay:31/39targets
have>=3readoutsites with>=30screenedpatients;BUN/bicarb/phosphate6sites each.
Amylase/fibrinogen/LDH2sites;5bedside targets unscanned, NOT failures. Per-site
counts not uniquepatient totals. Full event conflict/mask/history/sample acceptance
still pending; no traininggain or promotion. Supports expanding confirmation
from rawsource without reassigning existing evaluation patients. Runtime unchanged.

Full raw patient inventory COMPLETE:
`RECONSTRUCTION_EXPANSION_INVENTORY_20260917.md`. Same-role full-table histories,
excluding currentcohort/currentancestral people+sites, yield additional fit10201,
encoder3323,readout3433,eval7178,reserved16372patients. Provenance-only upperbound,
not labelqualification or untoucheddata.49currentpatients have additional outside
role/unassigned raw admissions: prior isolation is extracted-history scope, not
allrawstays. No evidence those admissions entered model; no leakage claim.
Need explicit boundary and source-only event support before expansion. No runtime.

Confirmation support correction COMPLETE:
`RECONSTRUCTION_CONFIRMATION_SUPPORT_20260917.md`. Fixed nominees reproduce
all28confirmation pooled scores1e-10. BUN site sizes1/6/9/11/14/16/32; phosphate
andbicarb likewise only1site>=30. Equal-site sign reversal cannot establish
true hospital heterogeneity versus sampling noise. Singleton bootstrap bounds
are degenerate, NOT certainty. Independent role splitting leaves weak per-site
power; no causal failure attribution, threshold relaxation or promotion.
Earlier cross-site-instability wording must be read as descriptive, not proved.

Source-confirmed heads TESTED, no efficacy gain:
`RECONSTRUCTION_CONFIRMED_HEAD_FINDINGS_20260917.md`. Fit-site5fold nomination,
independent readout-group confirmation vs original/population, patient/site
tests with fixed39target Bonferroni:0accepted/28supported. Outputs unchanged,
not accuracyrepair. BUN pooled advantage reverses under equal-site means;
phosphate fit-CV alpha100 fails confirmation unlike prior chosen1000. Bicarb
favorable means but insufficient site evidence. Seven confirmation sites limit
power; no absence-of-signal claim.3tests, alloriginalscores reproduce. All39
statuses, runtime and promotion unchanged. Source robustness remains unresolved.

Nested training/head evaluation and margin diagnosis COMPLETE:
`RECONSTRUCTION_NESTED_TRAINING_FINDINGS_20260917.md`. Three8epoch arms select
identical8/4/1 weights; independent head selection28supported targets yields
macro1.797586->1.785098. AST/bicarb/phosphate nominal gains; BUN/RR/WBC harms.
22heads replaced,20selection bootstrap intervals crosszero, including all3harms.
Post-selection intervals descriptive, not independent gates; min-MAE replacement
is insufficient evidence, not proof of sole failurecause. Need independently
confirmed source head nomination, not evaluation-tuned thresholds. All39statuses
preserved,6tests pass; no runtime/interval/promotion change. Earlier pending
nested-training entries below are superseded by this completed outcome.

Ancestry identity crosswalk RESOLVED for current source:
`RECONSTRUCTION_ANCESTRY_IDENTITY_20260917.md`. Original eICUpatient table verifies
all source/query patient,stay,hospital identities. Current ancestral4559stays/
3291patients/12sites has ZERO patient/stay/site overlap with every reconstruction
role. Earlier zero-string caveat replaced by verified mapping. Historical source
counts/hash unresolved, not exact exposure proof. No evidence to assert overlap
as failurecause or discard initializer. Nested warm-start development may proceed
with caveat; no runtime/promotion.

Warm-start provenance audit COMPLETE, independence still UNCERTIFIED:
`RECONSTRUCTION_WARM_START_20260917.md`. Release bytes match canonical5000
checkpoint; historical563736rows/3286subjects vs current reconstructed561024/
3287. Historical source hash/membership absent; current IDs not directly aligned,
zero exact-string overlap NOT independence. Need verified ancestry exclusion or
source-fit-only initialization/scaler. No exact leakage/accuracy cause claim,
no runtime/promotion. Nested fine-tune split alone does not solve provenance.

Independent selection partition/support COMPLETE, retraining pending:
`RECONSTRUCTION_NESTED_SELECTION_20260917.md`. Preserve fit536/eval178/reserved870;
split oldtune whole sites into encoder90/4sites and readout89/7sites.28/39targets
>=30allfourroles; prior33 lose CPK/fibrinogen/lipase/triglycerides/troponin support.
Oldweights used both groups, cannot relabel as independent.3tests plus full
hash/support checks; no model efficacy, runtime or promotion. Keep all39scope.

Source original/probe selection TESTED, broad repair failed:
`RECONSTRUCTION_READOUT_SELECTION_FINDINGS_20260917.md`.33targets tune-only
original/median/5ridge; choices frozen before score.9original6median18ridge.
MacroMAE2.934232->2.966656; bicarb/phosphate gains, lipase median benefit;
BUN/platelets/troponin harm. Triglycerides selects original yet loses median.
Tune already selected encoder, not independent head gate. Need genuinely
separated selection training, not relabeling old data. No runtime/promotion.

Matched raw-history/body readout contrast COMPLETE:
`RECONSTRUCTION_RAW_STATE_FINDINGS_20260917.md`. Strict-site33targets same
masked rows, source-only ridge/median selection. Raw beats body0nominal targets.
Body improves original bicarb/phosphate (nonconstant) and lipase/triglycerides
(median fallback only); harms5other targets. Not universal readout repair or
encoder-information-loss proof; raw265/body64 dimensions differ. Reused data,
nominal CIs, no future/interval/promotion/runtime change.

Post-mask remaining support diagnostic COMPLETE:
`RECONSTRUCTION_REMAINING_SUPPORT_20260917.md`.4637strict-site pairs matched,
all33MAEs reproduce. Lipase61/62 and triglycerides29/30 have other events<=1h,
yet numeric increment harms; all-input staleness is not sufficient explanation.
Glucose>6h group17patients harms descriptively; no significance/source policy
validation, do not gate on this evaluated subgroup. Availability lag is not
specimen age. No runtime/promotion or discarded patients.

Source readout calibration repair TESTED, not established:
`RECONSTRUCTION_READOUT_CALIBRATION_FINDINGS_20260917.md`. Six earlierV2 splits,
tune-only offset/nonnegative affine L1, matched process control, sourceeval.
No target6/6 improves original real under either method. Creatinine affine2gain/
2harm despite lower median; triglyceride slopes zero2/6. Output calibration
alone not stable repair. Alltarget hashes/MAEs/role isolation checked,4tests.
Reused development, nominal CIs, no promotion/runtime/interval changes.

Source ranking localization COMPLETE:
`RECONSTRUCTION_SOURCE_RANKING_20260917.md`. Earlier six V2source replays:
triglyceride median fit/tune Spearman0.662/0.088, lipase0.637/0.357. Ranking
loss precedes external transfer, not stable negative mapping; no sign reversal.
Only0-1supported tune sites for triglycerides,1lipase, so broad sitecause
unproven. All source MAEs reproduce; descriptive, overlapping fits and selected
tune, not strict-site repetition. No runtime/promotion.

Error-shape localization COMPLETE:
`RECONSTRUCTION_SITE_ERROR_SHAPE_20260917.md`. All strict-site MAEs reproduce;
33supported/6unsupported retained. Lipase/triglycerides harm affects majority
and central patients, not tail-only; triglyceride Spearman-0.497 on30patients.
Offset/positive shrink cannot fix ranking (may reduce MAE). Glucose/bilirubin
central mean harm coexists with tail gain; truth strata retrospective, not
deployable selection. No fitting, new significance or runtime/promotion.

Source objective/gradient diagnostic COMPLETE:
`RECONSTRUCTION_SITE_OBJECTIVE_20260917.md`. Exact tune-score replay: AST/ALT
57.27%net improvement, but other31 still improve0.050679;8targets regress versus
initial median. Fit-MAD rescaling keeps aggregate ranking. At selected epoch8,
8fit patients/target show persistent-layer raw mean descent locally harms only
glucose/WBC; lipase/triglycerides gradients align despite heldout harm. One
endpoint/sample, not AdamW or encoder-wide proof. Common-score retention is
absent but universal gradient-conflict cause unsupported. No runtime/promotion.

Strict site comparison TRAINED/EVALUATED:
`RECONSTRUCTION_SITE_TRAINING_FINDINGS_20260917.md`. Three8epoch arms141504updates
each,33targets; real/process/placebo select8/4/1. Evaluation178patients4637pairs:
12nominal all-reference gains; lipase/triglycerides harm allreferences. Bilirubin,
O2/glucose process increment unestablished. Split correction preserves some
sharednumeric signal but not universalrepair. Single reused split, multiplicity,
pretraining and convergence unresolved; no formalpromotion/runtime. Eighttests.

Common strict development site partition FROZEN:
`RECONSTRUCTION_SITE_PARTITION_20260917.md`. Patient-linked components, no label
values/errors/eligibility in assignment.536fit/179tune/178eval,11sites each;
870reserved/21sites.33/39targets>=30allsource roles; remaining6explicit including
aniongap3fit vs143tune (site concentration). Three tests and fullhash/support
build pass. Fine-tune split contract only; warm-start provenance stilluncertified.
Matched joint numeric/process/placebo training pending; no runtime/promotion.

Strict site support preflight found a boundary limitation:
`RECONSTRUCTION_SITE_SUPPORT_20260917.md`. Original hospital-role patients touch
18cohort hospitals;3also source (646sourcepatients). Patient-disjoint is not strict
all-history site-disjoint; NOT allselectedanchors or allrelease audits invalid.
Patient-linked site components reserve oldhospital components, leaving893source
and quarantining676.33/39targets have per-target single-component tune/test
support; no common assignment/training yet. Future splits must enforce complete
patient/site histories. No accuracyrepair/runtime/promotion claim.

Source hospital increment audit COMPLETE:
`RECONSTRUCTION_SOURCE_HOSPITALS_20260917.md`. Six V2fit source-only saved rows,
pooled scores reproduce. Bilirubin only1source site>=30targetpatients eachsplit;
O2/glucose site signs mixed; creatinine single-site deletion keeps favorable
mean6/6. Random patient replication not broad site invariance. Newdiagnostic
local-query-index error caught before output;8multi-site patients kept unknown
and in pooled errors. Exact selectedstay for8 remains unresolved. No promotion,
training/runtime change. Source-site support preflight needed before robust fit.

Target-level source fit/tune replay COMPLETE across six V2 seeds:
`RECONSTRUCTION_SOURCE_FIT_FINDINGS_20260917.md`.207300pair evaluations (not
independentpatients), original hashes and12selected tune scores reproduce.
Triglycerides fitgain6/6->tunegain0/6; bilirubin fit/tune/sourceevalmean gain6/6
but hospital0/6; creatinine mean gains6/6 allroles but nominaleval4/hospital3.
Distinct source-generalization, hospital-transfer and weak-increment patterns,
not one universal random-fit cause. All39catalog retained, AGunsupported kept.
Ten focused tests, no new fitting/promotion/runtime. Unique causes still unproven;
next source-only hospital/support localization before a domain-robust repair.

Fixed-hospital seven-fit stability diagnostic COMPLETE:
`RECONSTRUCTION_FIT_STABILITY_20260917.md`. Saved prediction hashes and both-role
MAEs reproduce; same patient keys/truths across available fits.35supported
seven-fit targets: median between-fit share of MSE2.59%, descriptive mean-output
MAE reduction2.40%. Most error remains common across fits, not explained by
fit variability alone; NOT irreducible-noise or optimizer-only attribution.
Bilirubin/O2 mean numeric increment harms process7/7; creatinine improves mean
7/7 hospital but not full originalgate. Need target-level fit/tune localization,
not ensemble promotion. Five tests; all39catalog retained; runtime unchanged.

Seven-split reconstruction COMPLETE:
`RECONSTRUCTION_SEVEN_SPLIT_FINDINGS_20260917.md`. All7 matched eight-epoch
comparisons finished. Nominal combined gains: albumin AST totalprotein7/7,
ALT6/7, bicarbonate/HR4/7, creatinine2/7. Full39target summary saved. Samehospital
group, overlapping patients, multiplicity unresolved; NOT promotion/forecast.
Seed83 report recovered from saved predictions after exact-hash source restore;
keys/truths/fit medians checked. Root cause of variability still confounds patient
split with optimization randomness; convergence not established. Runtime unchanged.

Source loss REPAIRED and seed71 COMPLETE:
`RECONSTRUCTION_REPLICATION71_RESULTS_20260916.md`. Both rebuilt upstream event
files byte-identical to historical hashes; no bypass. Three eight-epoch arms,
287664updates each,14435heldout pairs. Six combined nominal gains; six-split
intersection albumin AST totalprotein. Lipase hospital harms process. New weights
backed up durably; no runtime/promotion. Only seed83 remains in fixed protocol.

Seed71 could not start: two upstream temporary event files missing during
hash checks, not a training/accuracy failure. See
`RECONSTRUCTION_SOURCE_RECOVERY_20260916.md`. Preserved2366 surviving files in
Git-ignored durable data backup; all1763 original shard hashes verify. No binding
checks bypassed. Extraction ancestry located; upstream reconstruction/hash
recovery still required before seeds71/83. Completed five-split results unchanged.

Seed59 V2 training/evaluation COMPLETE:
`RECONSTRUCTION_REPLICATION59_RESULTS_20260916.md`. Allthree8epochs,289408updates
each;38source targets (anion gap tune26 unsupported),13806heldout pairs. Five
combined nominal gains bothroles; five-split intersection albumin AST totalprotein.
Both real/process selectepoch8, so convergence remains unproven. Samehospital
group, nominal comparisons; no promotion/runtime changes. Seeds71/83 pending.

Seed43 V2 training/evaluation COMPLETE:
`RECONSTRUCTION_REPLICATION43_RESULTS_20260915.md`. Three eight-epoch arms,
285792updates each;39source targets,14530heldout pairs. Nine nominal combined
gains bothroles, but intersection across7/19/31/43 falls to albumin AST totalprotein.
Bicarbonate real/process incremental error is nearzero in this fit, not stable;
creatinine evaluation CI crosseszero. Same hospital patients, mixed split/init
changes, unadjusted tests. Seeds59/71/83 pending. No runtime/promotion.

Seed31 V2 training and heldout evaluation COMPLETE:
`RECONSTRUCTION_REPLICATION31_RESULTS_20260915.md`. All three arms eight epochs,
288192 updates each; hashes/checkpoints verified. Anion gap source-unsupported
(tune26); 38 trained targets and14758 heldout pairs. Real beats process on ten
targets in both roles; nine also beat population/placebo. Four common combined
gains across seeds7/19/31: albumin AST bicarbonate totalprotein. Hospital bilirubin
harms process comparator; hospital fibrinogen/LDH/TSH remain unsupported. Same
hospital cohort, nominal comparisons, not promotion or future forecasting.
Seeds43/59/71/83 remain pending. Runtime unchanged.

Seed19 V2 COMPLETE eight epochs and heldout evaluation:
`RECONSTRUCTION_REPLICATION19_RESULTS_20260914.md`. Three arms293904updates
each, bindings verified. Real/process/placebo tune1.772240/1.873076/1.914907.
Fresh role remapping and fit-only medians;14064heldout pairs. Real-over-process
nominal9targets both roles;7also beat population/placebo both. Four overlap
seed7 combined comparisons: albumin AST bicarbonate totalprotein. Not independent
replications (same hospitalpatients), multiplicity uncorrected, no promotion.
Creatinine improves onlythissplit; MAP evaluation harms population. Remaining
seeds pending; runtime unchanged. Supersedes first-epoch pending below.

Seed19 replication V2 first epoch COMPLETE:
`RECONSTRUCTION_REPLICATION19_PROGRESS_20260914.md`. Initialnewdriver process
fit failedtoneutralize whiletune did; invalidcontrol, not seed7issue. V2 shared
viewhelper/tests,rera nallarms; real/placebo logs exact. Correctedtune real1.845440,
process1.899023,placebo1.914907 vsinitial1.914917.26264pairs/36738updates.
Use train_reconstruction_replication_v2; sevenepochsremain, heldoutuninspected.

Repeated-split reconstruction support preflight COMPLETE:
`NATIVE_RECONSTRUCTION_SPLIT_SUPPORT_20260914.md`.Same1763patients/all7disjoint
roles; SAME194hospitalpatients, not independent studies.38/39source-supported
allseeds; aniongap tune26/26/28 in31/59/83,eval25 in19. Remapshardroles and
refit eachsplit; no new training/promotion yet. Do not replace unsupportedseeds.

Multiplicity sensitivity COMPLETE:
`NATIVE_RECONSTRUCTION_MULTIPLICITY_20260914.md`.156sign-flip/Bonferroni tests;
onlytotalprotein point-estimatepasses, but MCintervals cross.05bothroles.
Posthoc/exchangeability assumptions, not originalgate replacement or robust
promotion. Need independent complete-target replication, not threshold tuning.

Matched TRAINED process control COMPLETE8epochs:
`NATIVE_RECONSTRUCTION_PROCESS_FINDINGS_20260914.md`.Process selects5/tune2.038137;
real1.921813. Real beatsprocess12evaluation9hospital,7overlap (albumin ALT AST
bicarb HR platelets totalprotein), nominal only. Process beatspop15/9, so prior
population/permutation gains not exclusively numeric physiology. No real-process
significant harms; glucose/Hgb/creat incremental gains unestablished. No promotion.

Expanded frozen numeric-path diagnostic COMPLETE:
`NATIVE_RECONSTRUCTION_NUMERIC_PATH_20260914.md`.Neutralize numbers only;
original predictions reproduce.22evaluation/17hospital nominal harms;12of14
prior gain targets harmedboth. Not exclusively support-independent outputs,
but OOD intervention != trained process control or physiology proof. MAP/creat
NS; magnesium numerical dependence insufficientaccuracy. No runtime/promotion.

Expanded frozen patient-alignment diagnostic COMPLETE:
`NATIVE_RECONSTRUCTION_ALIGNMENT_20260914.md`.38/39perrole lower error under
correct versus expected other-patient predictions; all14nominal gains aligned.
Constants zero benefit, parent MAEs reproduced. No statistical/newpromotion;
creatinine/MAP alignment != population improvement. Workflow stillconfounded.

Expanded shared-body eight-epoch training AND evaluation COMPLETE:
`NATIVE_RECONSTRUCTION_EXPANDED_FINDINGS_20260914.md`. Real selects7/tune1.921813
vsinitial2.085252; placebo selects0.14targets nominal paired gains vs population
and selectedplacebo in bothroles. Not multiple-testing/seven-seed/promotion;
current masked reconstruction only. BNP evaluation and magnesium hospital harm.
Sparse hospital3unsupported. All hashes verified; runtime unchanged. Supersedes
first-epoch pending below. Need replication/process-controls, not empty-state claim.

Expanded shared-body training FIRST EPOCH COMPLETE:
`NATIVE_RECONSTRUCTION_EXPANDED_TRAINING_PROGRESS_20260913.md`.39targets,
25511fitpairs all consumed,35997updates each real/placebo. Source-tune
2.085252->2.011559real; placebo2.086760 selects0. Preliminary only; seven
epochs remain. Optimizer/RNG checkpoints saved; heldout errors uninspected.
No runtime/promotion. Resume same experiment through epoch8.

Expanded actual reconstruction cache COMPLETE:
`NATIVE_RECONSTRUCTION_EXPANDED_20260913.md`.1763patients49460pairs,
25828fit9380tune8808evaluation5444hospital. All shards/hashes/masks/times
verified.39source-eligible variables;36meet30patients everyrole. No redraw
after raw-anchor rejection; differs from old reservoir. Training still needed,
not accuracy/promotion; runtime unchanged.

Full existing-cohort raw support ceiling COMPLETE:
`NATIVE_RECONSTRUCTION_SUPPORT_CEILING_20260913.md`.923/335/311/194patients;
36of53variables have potential >=30patients everyrole. Upperbounds ONLY:
conflict-aware masked reconstruction eligibility not yet counted. Dense targets
are not limited to16availablepatients;17slots belowminimum at leastonerole.
MAP-derived qualified events, not all localdata; no training/runtime/promotion.

Full-coverage shared-body training COMPLETE:
`NATIVE_RECONSTRUCTION_FULL_COVERAGE_FINDINGS_20260913.md`. Two8960-update arms;
balanced selects7,tune1.790593 vsinitial1.811102; eventweighted selects0.
Heldout means1.761483->1.739840 and1.046497->1.029325 descriptive.
ALL CIs unsupported: each role<=16patients, bootstrap requires30. Pilot cannot
statistically pass; not a demonstrated null. Need larger raw-event inventory,
support preflight and controls, not a lower gate. Runtime unchanged.

Multi-target training exposure localized:
`NATIVE_RECONSTRUCTION_EXPOSURE_FINDINGS_20260913.md`. Selected event checkpoint
consumed124/879 fit pairs and omitted12 direct targets; balanced consumed115/879.
Not adequate full-target optimization evidence. Matched full-coverage retraining
completed under a fixed protocol; see full-coverage findings above. No promotion.

Frozen body linear probe COMPLETE:
`NATIVE_RECONSTRUCTION_LINEAR_FINDINGS_20260913.md`.2checkpoints35targets,
source-selected ridge vs derangedplacebo; original MAEs reproduce. No supported
gain vs population/original/placebo eitherrole. Smallpilot doesnot prove no
latentinfo; no encoderupdate/runtime/promotion. Readout-only remedy unestablished.

Selected multi-target numerical output variation diagnosed:
`NATIVE_RECONSTRUCTION_PATIENT_SIGNAL_20260913.md`. All heldout MAEs reproduce;
patient-permutation shifts median .0039/.0052event and .0015/.0017balanced
normalized units, small vs already-small median corrections. Output nearconstant,
not proven bodylatent collapse. No training/runtime/promotion.

Actual multi-target sampling training COMPLETE:
`NATIVE_RECONSTRUCTION_SAMPLING_FINDINGS_20260913.md`.35targets32fit16eachother
patients, two512update shared-body arms. Eventweighted selects4,tune1.808488;
balanced selects2,1.810791 vsinitial1.811102. Neither has supported heldout
populationgain; balanced doesnotbeat eventweighted. Privateweights/cache saved,
no runtime/promotion. Smallcurrent-reconstructionpilot, not wholebodyimpossibility.

Multi-target initial gradient audit COMPLETE:
`NATIVE_RECONSTRUCTION_GRADIENT_BALANCE_20260913.md`.232patient-target examples,
38targets/8fitpatients. Dense4vitals comprise64.394% rows. Initial gradient
cosines nearzero; negative48.65% not learnedconflict proof. Huberloss size not
gradient domination. Patient/target balanced actualtraining still needed;
no update/runtime/promotion.

Query-safe raw-group memoization verified:
`NATIVE_RECONSTRUCTION_CACHE_EQUIVALENCE_20260913.md`.64anchor byte-equivalence
against SHA-verified historicalbuilder;full1877anchor counts6249 reproduce,
35.07s cached build.15tests. Semantic scopeunchanged, no efficacy/promotion.
Representative gradient balance/fullcohort training remain pending.

Multi-target original-event to shared-body update connected:
`NATIVE_RECONSTRUCTION_JOINT_STEP_20260913.md`. Fixed missing6zero event channels
using existing append_events;15tests pass. Four realtarget losses(CPK/INR/PTT/HR)
give body gradients and one joint update. Inputpool8patients != consumedpatients;
only4examples used. No efficacy/fulltraining/savedweights/runtime/promotion.

Raw-event multi-target reconstruction builder verified:
`NATIVE_RECONSTRUCTION_BUILDER_20260913.md`. As-of original-event scheduler,
fresh anchor-arrival truth and full target/sibling masks.14tests pass;8fitpatient
smoke1877anchors/6249examples inclrenal/metabolic/vitals. Not neuraltraining,
independentpatientcount or promotion. Fullcohort cache/training remains needed.

Multi-target event reuse constraint located:
`NATIVE_MULTITARGET_CACHE_SCOPE_20260913.md`. Native histories are query-asof
revision/conflict views, not unionable event ledger. Same-time token differences
correctly stop new inventory; no training/support-count report produced.
New target anchors require raw-event scheduler rebuild, not merging later
queries. Existing single-query accuracy not invalidated by this finding.

Feature-level support guard COMPLETE:
`NATIVE_BODY_SUPPORTED_RANGE_FINDINGS_20260913.md`.212window-variable slots,
88zero/36with1-19 repeat-patient support (includes intentionallymasked slots).
Fixed20fitpatient numeric guard retains446/848 features; comparedrange-only
MAE11.911->11.902 and14.099->14.095, bothNS. Population/placebo stillNS.
Do not infer sample adequacy from totalN or keep threshold-sweeping heldouts.
No neural/runtime/promotion change.

Sparse slope extrapolation localized/mitigated:
`NATIVE_BODY_FIT_RANGE_FINDINGS_20260913.md`. CPK6h slope has2fit/0tune
patient supports; evaluation slope~675x fitmaximum. Uniform fit-only numeric
range bounding reduces evaluation MAE17.293->11.911/maxerror1618.84->72.00;
hospital14.139->14.099 NS. Still no population/placebo advantage or promotion.
Local raw-probe failure explained, not allJEPA causes; runtimeunchanged.

Multi-scale masked raw reconstruction COMPLETE:
`NATIVE_BODY_MULTISCALE_FINDINGS_20260913.md`. Fixed1/6/24/allhistory probe
does not pass. Evaluation MAE17.293 vsincumbent12.268; one error contributes
30.10% total, max1618.84mmHg. Tail attribution dominated byCPK6h slope.
Exact retraining reproduces allresults. Source temporal-support inspection
needed; not previousJEPA rootcause or permission to trimheldout. No runtimechange.

Baseline-independent masked raw control COMPLETE:
`NATIVE_BODY_UNRESTRICTED_FINDINGS_20260913.md`. Removed inherited nested
support/affine preprocessing; raw MAE12.2678/14.2745 versus population
12.0032/14.3093, both differences NS. Support-only exactly reproduces.
No demonstrated repair, neural retraining or promotion; raw-summary limits remain.

Original-history time validation repaired:
`NATIVE_BODY_TIME_VALIDATION_20260913.md`. Four masked-away invalid-time
counterexamples now rejected; 14 tests pass. Full266872-query cache has zero
invalid original streams, so this defect does not explain the MAP results.
Source revision explicitly recorded; historical artifact hashes unchanged.
No new training, accuracy claim or runtime change.

Expanded current-only masked raw probe COMPLETE:
`NATIVE_BODY_EXPANDED_FINDINGS_20260913.md`. Futurepair restriction removed,
onefreshanchor/patient. Heldout311/194patients raw/populationgain0bothroles;
rawMAE12.2774/14.2426 vs12.0032/14.3093. No raw/support/placeboincrement.
Not neural training/informationabsence/saturation proof; runtimeunchanged.

Matched masked raw-history control COMPLETE:
`NATIVE_BODY_RAW_CONTROL_FINDINGS_20260913.md`. Same53fit27tune50+50cohort.
Rawridge12.4076/13.6418 vs population12.58/11.70; neither significantgain.
Beats placebo onlynewsource, supportboth but supportharms hospital. No stable
rawsignal loss attributable uniquely to encoder. No runtime/promotion.

Native body masked training COMPLETE:
`NATIVE_BODY_MASKED_FINDINGS_20260913.md`. Fresh mask support53fit27tune,
50+50heldout. Real8epoch selects0,populationMAE12.58/11.70; placebo selects2,
no significant difference. Legalbody supervision not sufficient MAPrecovery
in smallpilot. Not forecast/wholebody impossibility, runtimeunchanged.

Native body masked supervision integrated:
`NATIVE_BODY_RECONSTRUCTION_20260913.md`. Same persistenttrajectory body
readout; target+sibling fullhistory masking, emptyevent removal, missingreject.
4tests pass; real freshMAP smoke-step bodyupdategrad.0382/dynamics.0772.
Structural supervision repair only, no efficacy/savedweights/runtime/promotion.

Native parameter update audit:
`NATIVE_ORGAN_PARAMETER_UPDATES_20260913.md`. Organ gates/dynamics move8-9%
L2 relative, body gates/dynamics exactlyunchanged. Confirms prior closedbody
bypass scope: not complete body-state learning experiment. Not sole forecast
cause; body supervision path required without unrestrictedforecastmixing.
No new training/runtime/promotion.

Owner event-mixture scope checked:
`NATIVE_OWNER_EVENT_MIXTURE_20260913.md`. MAP owner blood_vessels (notheart).
Lactate-only events22/64fit12/32tune patients, zero afterlastMAP all96queries.
FinalMAP overwrite not supported; earlier assimilation untested. No modelchange.

Nested baseline/residual COMPLETE:
`NATIVE_NESTED_RESIDUAL_FINDINGS_20260913.md`. Baseline32epochs excludes
residualfit/tune patients from optimization/selection; fullroles beatpersistence.
Residual64fit32tune8epochs bothselect1; state increment-.01213source/+.00256
hospital bothNS, no gain overanchor-only. Exposure correction insufficient
in thispilot, not multi-foldcrossfit or solecause proof. No runtime/calibration.

Native pretraining provenance checked:
`NATIVE_PRETRAINING_OVERLAP_20260913.md`. Recorded-source currentfile3291subjects
vsMAP1763, exactsubject/stay overlap0. No historicalinputhash/remapping proof;
not an untouchedpretraining claim. Confirmed native fine-tune exposure remains.
Nested excluded-patient baseline training pending, runtime unchanged.

Residual training exposure verified:
`NATIVE_RESIDUAL_TRAINING_EXPOSURE_20260913.md`. All64 residualfit patients
seen by baseline optimizer,22exactqueries; tune/eval/hospital0. Baseline tune
used for epochselection. Fit residual/current slope-.168 vs tune+.089,
descriptive only. In-sample vs unseen-error asymmetry, not proven solecause.
Patient-excluded baseline/features needed for clean stacking test; no runtime.

Native objective ablation COMPLETE:
`NATIVE_ORGAN_OBJECTIVE_FINDINGS_20260913.md`. Fixedteacher/centeredteacher/
value-only three8epoch fits allselect0. Tune8.34714->8.41928/8.42284/8.42367.
Latent shortcut not necessary for point failure; removingJEPA also fails.
No solecause/repair/promotion. Smallpilot, runtime/calibrators unchanged.

Native latent-objective shortcut diagnosed:
`NATIVE_ORGAN_LATENT_OBJECTIVE_20260913.md`. Fixedteacher tune loss.005629 to
.001140, but FIT-mean constant residual achieves.001168 (~99.36% of reduction).
Bothnone/sparse. EMA drift exists but fixedteacher improvement persists.
Most latentloss progress not patient-dynamics evidence in thispilot. Not an
independently trained constant model or demonstrated forecast repair. No runtime.

Conditional owner current+change pilot COMPLETE:
`NATIVE_ORGAN_CONDITIONAL_FINDINGS_20260913.md`.Three same-shape head arms,
64fit32tune8epochs. Conditional selects1, tune8.34492 vsbaseline8.34714;
change/anchor-only select0. Selected heldoutdelta+.00251/+ .00711 bothNS.
Current-state access alone not sufficient repair; four invarianttests pass.
Smallpilot, no runtime/calibration/promotion; no full-body impossibility claim.

Native state variation localized:
`NATIVE_ORGAN_STATE_VARIATION_20260913.md`.64fit/32tune fixed-checkpoint probe:
current owner-state varies (tune centeredRMS.2614), but delta read by head
RMS.00387 (1.48%); sparse1.87%. Initial no-edge4.03%. Mean delta grows while
between-patient delta spread shrinks. Not information-loss percentage or
useful-state proof. Conditional current-state+change readout remains untested
in this pilot; no new training/promotion/runtime changes.

Native state-input training control COMPLETE:
`NATIVE_ORGAN_STATE_INPUT_FINDINGS_20260913.md`. Fullstate eight epoch scores
exactly reproduce pilot; anchor-only head control also worsens tune8.34714 to
8.42216 versusfull8.41926. Both selectepoch0. Failure does not require state
input; no demonstrated extra benefit, not equivalence or solecause proof.
Fixed-head shuffling changes outputs~0.001mmHg. Small64fit32tune diagnostic;
no runtime/calibration/promotion. Direct bindings verified.

Native replay exact learning trace COMPLETE:
`NATIVE_ORGAN_REPLAY_TRACE_20260913.md`.All16pilot tune scores reproduced1e-8.
Fit7.42657->7.37558/7.37538 while tune8.34714->8.41926/8.41953. Learned
correction fails tune, not disconnected gradient. Finalepoch8 heldoutdifferences
NSbothroles, selectedepoch0 remains. Smallpilot, not solecause/convergence;
no runtime/calibration/promotion. Finalweights now retained for diagnostics.

Native anatomical replay paired learning pilot COMPLETE:
`NATIVE_ORGAN_REPLAY_PILOT_FINDINGS_20260913.md`.64fit32tune patients,8epochs,
none/sparse both selectepoch0. Tune8.34714->8.41926/8.41953, no increment.
64patient heldout eachrole equalsbaseexact; persistence CIs cross0 on thispilot.
Not fullcohort/7seed/no edgepromotion. Fitloss/finalweights notretained so no
memorization or unique rootcause claim.9bindings verified, runtimeunchanged.

Native organ future supervision integrated:
`NATIVE_ORGAN_FUTURE_SUPERVISION_20260913.md`. One actualsourcefit MAP pair,
two value+JEPA steps; zero-start exact, secondstep organupdategrad5.12e-5.
Owner-local statechange heads, sameanatomicaltrajectory, closededges. Newanchor
floatprecision bug repaired/tested. NOTaccuracy/fulltraining/promotion; body
utility and matched scoped/unscoped training remain pending. No weightssaved.

Native anatomical replay bridge smoke-tested on real source streams:
`NATIVE_ORGAN_REPLAY_BRIDGE_20260913.md`. SameJEPA weights -> pre-mixer local
organ proposals/sharedbody -> persistentreplay.8patients112events, finite
backprop to JEPA/updategates.14canonicalslots; no unvalidatededges/treatment.
State-norm smoke objective only, no futuretraining/MAE/promotion. Needs explicit
future-change readout and matched training; releaseunchanged.

Persistent chronological latent replay implemented:
`PHYSIOLOGY_OBSERVATION_REPLAY_20260913.md`. advance owns time integration;
update correction no longer doublecounts elapsed. replay_observations runs
one body/allorgans with scoped corrections and strict groupedtimestamps.
25tests pass incl two-organ sequence and gradients. Component only, not yet
raw-event JEPA training/accuracy/promotion. Existing future rollout untouched;
no weights/registry/calibrators changed. Not priorMAP failure attribution.

Persistent observation-mask defect repaired:
`PHYSIOLOGY_UPDATE_MASK_REPAIR_20260913.md`. All-organ-missing update changedbody
by98 in syntheticcounterexample; NaNs also pollutedorgans. Maskbeforeprojection,
body-presence correction and validitychecks now pass21tests. NOTrecentMAPcause:
nativefilter trials do not call thisupdate; inspected persistent/organexchange
forward paths initialize/rollout only. No weights/registry/calibrators changed;
full-body online assimilation remains unintegrated, no accuracy/promotion claim.

MAP predict-only event control verified:
`MAP_PREDICT_ONLY_CONTROL_20260913.md`. Seven frozen factorial checkpoints,
96sourcefit histories each;671matched endpoints maxforecastdiff7.63e-6 despite
46-54extraevents. One later-event endpoint excluded by inputtime. Separates
time decay from no-target-content innovation; structural bridge only, not
full-body repair/new MAEgain/training/promotion. Three tests; runtimeunchanged.

MAP projection-content control ALL7 COMPLETE:
`MAP_PROJECTION_CONTROL_FINDINGS_20260913.md`.84directbindings verified.
Projection-control/full gain0/harm0 bothroles; projection-own harm7/7both,
gain0. Hospital projection10.60673/control10.61140/own10.21067. Metadata removal
not sufficient repair; activation/clocks/query/optimization not isolated.
Supersedes sixpending below. No runtime/calibration/promotion.

MAP projection-content control seed7 COMPLETE32epochs:
`MAP_PROJECTION_CONTROL_FINDINGS_20260913.md`. Target-only history projection
with original full activation/query: versus support control no significant
change in either role; hospital10.54625->10.55518. Versus own significant harm
both (+.29146source/+.34450hospital). Not a sufficient metadata repair in this
split; cadence/optimization still unresolved. Six fixed seeds pending, no
runtime/calibration/promotion. Three focused tests include recurrence equality.

MAP clock retraining ALL7 COMPLETE21fits:
`MAP_UNCLIPPED_TRAINING_FINDINGS_20260913.md`.98directbindings verified.
Corrected-old gains0allarms/roles; full-own harms7source5hospital,gains0.
Hospital full10.57510->10.60183,own10.21067->10.26613. Clock correction not
sufficient repair under fixedpolicy. No runtime/calibration/promotion;
supersedes sixpending below. Remaining history mechanism not uniquely proven.

MAP unclipped-clock retraining seed7 COMPLETE3fits:
`MAP_UNCLIPPED_TRAINING_FINDINGS_20260913.md`. No significant improvement;
hospital own+0.16659/full+0.08959 significant harm,controlNS. Corrected full
still worseownboth. Oneof7 only, remaining6pending; not sufficient repair or
timing-irrelevance proof. No runtime/calibration/promotion.

MAP unbounded history clocks RECOVERED60shards:
`MAP_UNCLIPPED_CLOCK_20260913.md`.266872queries exactlyonce625722tokens,
234666clockchanges. Everyoldtoken/treatment/schedule exact, nonclock unchanged;
182directbindings verified. Newprivatecache only, matched training pending;
not accuracyrepair/runtime/promotion. Supersedes rawtime-recovery pending.

MAP clock scope: `MAP_NATIVE_CLOCK_SCOPE_20260913.md`. CurrentMAP shards still
clip historyhours0..168; previousKclockrepair not applied. Seed7 patient-equal
zero-clock-delta24.71%newsource18.35%hospital; fullysaturated11.57%/8.30%.
All7partitions audited. Not accuracycause proof; rawtime recovery/MAP matched
retraining pending. No runtime/calibration/promotion; priorKnegative retained.

MAP latest-observation carry probe: `MAP_NATIVE_CARRY_20260913.md`.
96sourcefit patients/seed, own/full/control, predictions reproduced. Median
post-lastMAP updates0 and direct(1-gate)carry1 allarms/seeds. Not routine
latestMAP overwrite; earlier-history/innovation/optimization remain open.
Two tests. No training/runtime/promotion or total-memory-retention claim.

MAP training-matched path factorial COMPLETE14fits:
`MAP_CONTROL_FACTORIAL_FINDINGS_20260913.md`. Controlhistory/ownquery harms
own7/7bothroles. Ownhistory/controlquery no significant gain/harm vsown;
improves control7/7source6/7hospital.31bindings verified. Gap follows expanded
history path in support control, not necessarily currentquery. Masks/ages/
cadence still combined; no unique rootcause, fullnumeric repair or promotion.

MAP support-control path replay: `MAP_CONTROL_PATHS_20260912.md`.
Seven source-tune frozen checkpoints exact originals. Own-query replacement
harm7/7(+.46694 median); own-history harm3/7(+.07694); neither improves.
Current query support matters too, not just update count. Off-distribution
input replay cannot assign training cause; no fitting/runtime/promotion.

MAP numeric control ALL7 trained: `MAP_NATIVE_NUMERIC_CONTROL_FINDINGS_20260912.md`.
Matched full support/cadence but removes non-MAP/event numbers. Control-own
gain0/harm7 bothroles; control-full gain0 both,harm1source0hospital. Hospital
MAE control10.61140/full10.57510/own10.21067.18direct bindings verified.
Extra numbers not necessary for gap; masks/age/query/update cadence still
confounded. No solecause, runtime, calibration or promotion. Supersedes prep.

Native MAP numerical-control preparation:
`MAP_NATIVE_NUMERIC_CONTROL_PROTOCOL_20260912.md`. New helper removes other
normalized numbers in history/query but keeps support/time/activation. Seven
unit tests pass. Not yet trained or accuracy-evaluated; runner/full-data checks
pending. No runtime/promotion. Do not report preparation as a successful repair.

New MAP active-update counts: `MAP_NATIVE_UPDATE_COUNTS_20260912.md`.
Same history indices do not mean same state updates. Seed7 patient-equal
own/full counts26.08/75.22 newsource,16.69/54.19hospital. All7 input-only
partitions audited; padding inactive, full preserves own updates. Quantifies
known content/cadence confound, not causal accuracy attribution or repair;
old inference-only scope replay already failed. No runtime/promotion.

MAP matched refinement ALL7 complete: `MAP_NATIVE_QUERY_REFINE_FINDINGS_20260912.md`.
14 additional fits, direct dependency hashes verified. Full-oldfull source
gain4/7,hospital0/7; full-own gain0 both,harm7/7source6/7hospital. Median
hospital full10.57510->10.55747,own10.21067->10.15199. Partial source improvement
not stable whole-context repair. Fresh AdamW, not uninterrupted64epochs or
convergence proof. No runtime/calibration/promotion; supersedes pending below.

New MAP matched refinement seed7: `MAP_NATIVE_QUERY_REFINE_FINDINGS_20260912.md`.
FreshAdamW32moreepochs: full-oldfull sourcegain-.09255,hospital-.01187NS;
full-own stillharmboth. Ownalsoimproves source. Oneof7 only, no stable repair
or uninterrupted-training claim; remainingseeds pending, runtimeunchanged.

New MAP shared-gradient probes: `MAP_NATIVE_QUERY_GRADIENTS_20260912.md`.
42sourcefit probes, predictions reproduced. FullJEPA/value normmedian.01517,
opposed9/21 but rawvalue-descentfactor>=.9949. No dominantopposition at selected
blocks; not Adam/earlypath/lossablation proof. No runtime/promotion.

New MAP value-objective check: `MAP_NATIVE_QUERY_OBJECTIVE_20260912.md`.
Exact normalizedSmoothL1 fullworse7/7 everyrole inclbothseen(.008631median).
Not MAE-only gap. Descriptive selected value loss, excludeslatentloss/optimizer;
no newfit/solecause/runtime/promotion.

New MAP sampled exposure: `MAP_NATIVE_QUERY_EXPOSURE_20260912.md`.
Exactsampler replay:32epoch10.07-10.77%fitquerycoverage. Both-seen full-own
harm6/7(+.22127),neither7/7(+.29949),gain0. Not only unvisitedquery issue;
MAE/selectedcheckpoint not totaltrainloss/convergence. No fit/runtime/promotion.

New MAP nativequery neural14fits COMPLETE: `MAP_NATIVE_QUERY_TRAINING_FINDINGS_20260912.md`.
Full-own harm7/7 newsource ANDhospital; bothpersistencegain7/7. MedianMAE
own8.84140/10.21067,full9.22298/10.57510. Fitpopulationalsoharm7/7(+.23255),
tune+.25346; selected predictions reproduce exact. Fit contains unsampledqueries,
so not actual-sampled fitting proof. Timingalone insufficient; no solecause,
calibration/runtime/promotion. Supersedes seed7-only/pending entries.

New MAP nativequery neural seed7 COMPLETE: `MAP_NATIVE_QUERY_TRAINING_FINDINGS_20260912.md`.
Own/full32epochs, selected25/23. Newpatient MAE8.80806/9.16712,hospital
10.21067/10.56384; full-own significantlyworseboth, bothbeatpersistence.
Only1/7split;9bindings7tests. Newtiming not sufficient in thissplit, not sole
cause/noinformation proof. Sixseeds pending; no runtime/calibration/promotion.

New MAP source transforms ready: `MAP_NATIVE_TRAINING_TRANSFORMS_20260912.md`.
All7seeds fit-only centers/scales/support/eventtransforms;46supportedvars each,
fit/otherpatient overlaps0. Compact own/full batches exact versusdense including
padding/events;3tests. Neuraltraining stillpending, no efficacy/runtime/promotion.

New MAP neural batch exercised: `MAP_NATIVE_TRAINING_BATCH_20260912.md`.
Ragged loader all266872queries625722tokens,max388; no truncation. Six tests,
real four-source-query AdamW/EMA step finite, historygradient1.19594/update
.011265. Integration only, no savedmodel/efficacy/promotion; matched training
and source transforms remain required. Runtime unchanged.

MAP newquery nativehistory COMPLETE: `MAP_NATIVE_QUERY_HISTORY_20260912.md`.
All60shards266872queries205723eligiblepairs; everyquery exactlyonce, all7461
oldcommonhistories exact. Full hash/assembly verification includes treatment
sequences and querycontexts. Newtask neuraltraining stillpending; no efficacy,
calibration/runtime/promotion. Supersedes shard0-only preparation below.

New MAP query nativehistory bridge: `MAP_NATIVE_QUERY_HISTORY_20260912.md`.
Shard0/60 built:1885queries4464groups, all19commonhistories exact including
treatmentvalues. Existing native_arrays reproduces all1885sequences and query
contexts; finite inputs. Four tests pass. Other59shards and matched neural
training remain pending; no accuracy/calibration/runtime/promotion claim.

MAP neural task scope: `MAP_NEURAL_TASK_SCOPE_20260912.md`. Oldnative33477 vs
newMAP205723eligible anchors overlap3905(1.898%). Recent tree fixes are not
JEPA retraining on nativeMAP task. New ragged histories/paired neural trial
remain needed; no accuracy/runtime/promotion claim.

MAP missing fitgap: `MAP_MISSING_FIT_GAP_20260912.md`. Full-own fit7/7gain,
full-support4/7; both0/7tune/newsource/hospital. Original scores/comparisons
exact. Numerical fit remains, not totalcapacityabsence; no solecause/runtime.

MAP derivedgap supported strata: `MAP_DERIVED_GAP_STRATA_20260912.md`.
Current-supported hospital147patients, derived-own/full/no-gap0/7gain;
full delta+.02617. No hidden complete-case stablebenefit. Preanchor masks only,
pooledexact; no fitting/routing/runtime/promotion.

MAP explicitgap21fits: `MAP_DERIVED_GAP_TRAINING_20260912.md`.
Derived-full/own0/7gainboth; no-gap/support1/7source0hospital. Hospital
full10.31455->derived10.25123, no-gap10.24189, own10.21838. Conventionconcern
not established principalcause/repair. Three tests; no runtime/calibration/promotion.

Derived anion stream ready: `MAP_DERIVED_ANION_SUPPORT_20260912.md`.
Explicit withoutK same-time/known-time components, invalidversions blockold;
48372versions27410valid. Fit40.02%hospital38.66%patient-equal current support.
No observedgap overwrite, no trained efficacy yet; two tests, runtimeunchanged.

MAP anion convention audit: `MAP_ANION_CONTRACT_20260912.md`. All5262rawvalues/
times exact. Same-time panels fit118/1798 matchwithK vs hospital228/848;
majoritywithoutKboth. Mixedconvention evidence, not specimen/method proof or
MAPerror cause. No recoding/training/runtime; explicitderived contract needed.

MAP fitted usage: `MAP_FEATURE_USAGE_20260912.md`. All7fullmodels rare<30fitpatient
variables zero splitgain. MAP45.85%,aniongap7.00%median gain; numbers82%includes
NaNrouting, not purephysiology. Veryrare dominance unsupported; combinations/
domain shifts unresolved. No training/runtime/promotion.

MAP native-missing support7fits: `MAP_MISSING_SUPPORT_20260912.md`.
Matched full-support0/7gain/harmboth; hospital10.31455vs10.31205,own10.21838.
Partial missingrepair not numerical-coupling evidence. Two tests, oldscoresexact;
no runtime/calibration/promotion or information-absence claim.

MAP native missing14fits: `MAP_NATIVE_MISSING_TRAINING_20260912.md`.
Maskednumbers NaN, masks/ages retained; observedzero preserved. Full-oldfull
gain0source2hospital; full-own0both. Hospital10.31455vsown10.21838. Partial
representation benefit, not exact oldcollision/solecause/promotion; runtimeunchanged.

MAP ordered increment diagnosis: `MAP_NATIVE_INCREMENT_DECOMPOSITION_20260912.md`.
Fit total-.29150, support-.24724, additionalnumbers-.03148; direct paired
support-vs-number magnitude difference7/7. Both lose heldoutgain. Much fitting
advantage exists withoutothernumbers; not causal attribution/workflow-only
proof. Original stats exact, no training/runtime/promotion.

MAP event-equal training21fits: `MAP_NATIVE_EVENT_WEIGHT_TRAINING_20260912.md`.
Only fitting weights changed, original held-out scores exact. Full-oldfull
gain0/7both, full-own0/7both; hospital10.37609 vs own10.23905. Repeatedfuture
fit weighting not sufficient repair. Eighttests; no runtime/calibration/promotion.

MAP-native partition recheck: `MAP_NATIVE_PARTITION_CHECK_20260912.md`.
All7roles patient/futureevent overlap0; duplicatequery0, positivehorizons.
Isolation confirmed, not remaining rootcause or upstream/pretraining proof.

MAP shrink support control: `MAP_NATIVE_SHRINK_SUPPORT_20260912.md`.
Same tune grid full/support; originalfull exact. Shrunkfull-shrunksupport
gain1/7source0hospital; hospital10.28760vs10.28528,own10.24211. Partialblend
gain not stable numerical attribution; no runtime/promotion.

MAP-native shrink partial repair: `MAP_NATIVE_INCREMENT_SHRINK_20260912.md`.
Source-tune alpha.1-.9; selected-own gain4/7source0hospital, selected-full
3/7source5/7hospital. Hospital10.37322->10.28760 still own10.24211better.
No complete externalincrement/calibration/runtime/promotion. Not solecauseproof.

MAP leaf patient support: `MAP_NATIVE_LEAF_PATIENTS_20260912.md`.
Seed7all200trees eacharm, nodecounts/predictions verified. Full only1/3000
leaf<5patients, tinyassignmentmass; no widespread near-single-patient-leaf
explanation. Not ensemble memorization exclusion or trained repair/promotion.

MAP sampling-gap accuracy: `MAP_NATIVE_SAMPLING_GAP_20260912.md`.
Preanchor repeateddraw-gap strata, full-own0/7gain with/without bothroles;
hospital no-gap harm remains. Not sole second-draw omission explanation,
not complete event-retention audit or retrained repair. Runtimeunchanged.

MAP-native raw/snapshot density: `MAP_NATIVE_RAW_DENSITY_20260912.md`.
Seed7 raw repeated6h creat6.72%vs snapshot6.60%,MAP93.27%vs85.41%fit.
Sparse labs mostly sparse in qualifiedcache; vitals lose some grid support.
No accuracy attribution/full-download inventory; six tests, runtimeunchanged.

MAP-native input density: `MAP_NATIVE_HISTORY_DENSITY_20260912.md` (V2).
Seed7 recent>=2draw proxies fit HR88.34%/MAP85.41%,creat6.60%,lactate1.60%.
Not fullraw event counts; recent lag != recentdraw corrected in auditV2.
Five tests; no accuracy causation, training or runtime/promotion.

MAP-native outcome/support audit: `MAP_NATIVE_OUTCOME_WEIGHTS_20260912.md`.
Event-then-patient weighting full-own0/7gain newpatients/hospitals; fit7/7.
Support-own fit7/7 also loses newpatient gain. Not numerical-only fit evidence
or repeated-outcome scoring repair; no workflow-only/causal/promotion claim.

MAP-native tree gap/stopping: `MAP_NATIVE_TREE_FIT_GAP_20260912.md`.
Full-own fit7/7gain becomes0/7tune/newsource/hospital. Source-only staged
selection0..200 full chooses200sixseeds; earlier stopping0/7repairbothroles.
Not external-only or proven overtraining/convergence; no runtime/promotion.

MAP-native raw-history21fits complete: `MAP_NATIVE_ANCHOR_TREE_20260912.md`.
Fixed15leaf HGB own/full/support, same newtask. Full-own0/7gain1/7harm both;
full-support0/0; persistence7/7both. Hospital185patients, full10.37322own10.24211.
41bindings verified. MAP-native timing not sufficient repair; not JEPA training,
information absence, old/new paired benefit, calibration or runtime/promotion.

MAP-native inputs completed: `MAP_NATIVE_ANCHOR_SUPPORT_20260912.md`.
266872queries/60private shards,205723eligiblepairs. All7461commonanchors AND
fixed-lag histories bitexact acrosssixfields; labelcurrent agrees. No fitting
yet, no accuracy/runtime/promotion. Supersedes input-pending note below.

MAP-native support: `MAP_NATIVE_ANCHOR_SUPPORT_20260912.md`. Queries were
insulin-rate-triggered. Same-stay MAP availability anchors yield205723pairs/
1702patients vs33477/1505; medianage10min vs60min. Old labels exactly reproduce.
146725distinct future events,22755overlap; changedtask, not accuracy gain.
Private labels/roles ready; matched inputs/training pending, no runtime/promotion.

MAP recordedchange strata: `MAP_HISTORY_CHANGE_SUBGROUPS_20260912.md`.
Historyrate-currentrate0/7gain/harm in preanchorchange group bothroles,
hospital38patients; mean signs mixed. SmallN not equivalence. No fitting,
routing, runtime or promotion; futureMAP not used for grouping.

MAP recorded history trial complete: `MAP_PRESSOR_HISTORY_TRAINING_20260912.md`.
Seven current-rate controls exact. History-rate vs current0/7gain allscopes;
oldsource/hospital1/7harm. Hospitalall10.14075->10.14504. Fixed1/3h snapshots
not full eventhistory; no stable repair/JEPA/calibration/runtime/promotion.

MAP recorded pressor history built: `MAP_PRESSOR_HISTORY_20260912.md`.
Past1/3hcutoffs, currentcache exact. Norepi changes2687queries/494patients1h,
4100/531at3h. Eighttests; same-record carry excluded. Recordedtimeonly, not
complete administrations or efficacy. Training pending; runtimeunchanged.

MAP pressor sourcevariation: `MAP_PRESSOR_VARIATION_20260912.md`. Eightobserved
channels presentall7fits; norepi209-227patients/91-102rates(weightnormalized),
vasopressin119-144patients/12-21rates. Not empty/constant principalchannels;
some otherchannels lowN. No fit, gating, runtime or promotion changes.

MAP pressor strata: `MAP_PRESSOR_SUBGROUPS_20260912.md`. Fixedprediction,
recordedpreanchor rate support; rate-baseline/support0/7gainbothroles with
support. Hospital45patients limits precision; not pooleddilution proof or
equivalence. No training/routing/runtime/promotion.

MAP recorded pressor training: `MAP_RECORDED_PRESSOR_TRAINING_20260912.md`.
Seven treebaseline/support/rate arms, baseline exactall7. Rate-baseline0/7gain
allscopes; hospitalall MAE10.21392->10.14075, rate-support1/7gain. Not stable
repair; snapshotonly/retrospectivetiming, no JEPA/conformal/runtime/promotion.

Recorded MAP pressors built: `MAP_RECORDED_PRESSORS_20260912.md`. All55468
queries,145510matchedrows; invalidlatest and conflictinglatest blockoldrates.
MAP norepi8125/vasopressin2082supported querycells. Eighttests; onlineeligible
false (recordedtimeonly). No fitting/accuracy/runtime/promotion.

MAP pressor capture: `MAP_PRESSOR_CAPTURE_20260912.md`. Currentnative excludes
pressors. Located units/min parser omission; exact60x aliases repaired,13tests.
Vasopressin compatible rows82->13922,6hqueries10->2101/231patients. No independent
infusion availability; inventory not efficacy or MAPgapcause. Parserchanged,
existing weights/caches/registry/calibrators unchanged; no promotion.

MAP update-scope replay: `MAP_UPDATE_SCOPE_20260912.md`. Seven expanded full
controls exact; MAP-only activeupdates retain fullquery. Versusoriginal gain1/7
source0hospital; versusown0both. No stable repair, content/cadence confounded,
not trainmatched or evidence skipping safe. One invariant test; runtimeunchanged.

MAP loss-gradient check: `MAP_LOSS_GRADIENTS_20260912.md`.42selected source-fit
probes; full13/21negativecosine, medianJEPA/value norm.0153. Rawcombined
value descent factor>=.9917; no dominant opposing gradient at inspected blocks.
Not Adam/update trajectory or loss-ablation proof. No runtime/promotion.

MAP value objective check: `MAP_VALUE_OBJECTIVE_20260912.md`. Same selected
predictions, exact normalizedSmoothL1: full worseown5/7fit,7/7tune numerically.
Not MAE-only failure; excludeslatentloss/gradients, not loss-ablation proof.
Three tests pass including formula equivalence; no runtime/promotion.

Expanded MAP fit-gap replay: `MAP_EXPANDED_FIT_GAP_20260912.md`.14selected
checkpoints reproduce oldpredictions/tune. Full-own fitgain2harm3, tunegain0harm2;
actually sampled-both rowsgain1harm3, sampled-neither gain0harm5. Not external-
shift-only or universally successful fit signal. Selectedepochs differ, unseen
rows may share outcomes, not full optimization diagnosis. No runtime/promotion.

Expanded MAP sampler/curve audit: `MAP_EXPANDED_EXPOSURE_20260912.md`.
Exact replay: epoch32 allpatients but36.9-39.3%rows; selectedfull20.6-35.4%rows.
Bestfull tune worseown7/7; latehalf best improves3/7full. Not universal late
improvement or proof undertraining; selected fit errors still unmeasured.
No model/runtime/promotion change.

Expanded native MAP14fits completed: `MAP_EXPANDED_NATIVE_20260912.md`.
Same old tune/scoring, expanded fitting only. Full-oldfull gain0/7source and
hospital; full-own gain0/7both, sourceharm4/7. Hospital median MAE full
10.43047->10.39056, own10.25143->10.03774; own expansion gain1/7hospital.
Full-persistence7/7source2/7hospital. All49 direct bindings verified. Selection
defect repaired but insufficient accuracy repair; no calibration/promotion or
runtime change. Supersedes native-training-pending entries below.

Expanded native MAP source transforms verified7/7 bitexact on oldfitpatients:
support/numeric/event centers/scales. map_expanded_transforms_20260912 artifact.
Cache reindexing not transformconfound; neuraltraining stillpending, no efficacy.

Expanded MAP native history complete: map_all_native_tokens_20260912 artifact.
55468queries,427180uniquegroups,median96/max385events; alloldtoken ANDtreatment
sequences exact. No truncation, samecontract. Readyforneuraltraining, not yet
trained/promoted; supersedes native-expansion pending. See MAP_LABEL_SELECTION.

MAP selection repair trained: `MAP_SELECTION_TRAINING_20260912.md`. Oldtune
fixed, oldcontrols exact7/7. Expandedfull improvesoldfull4/7source1/7hospital
on oldrows,5/7source1/7hospital allrows; full-own0/7both. Partial scope repair,
not solecause/neural/coupling proof. No calibration/runtime/promotion.

Expanded MAP fixed-lag inputs ready: map_all_query_inputs_20260912 artifact.
All55468queries,33477eligibleMAPpairs; oldquery AND history arrays exact.
No neural/native-event expansion or training yet. See MAP_LABEL_SELECTION.

MAP label selection scope repaired in new private cache:
`MAP_LABEL_SELECTION_20260912.md`. Old MAP queries required unrelated Kfuture
pair. Removing prerequisite restores33477pairs1505patients vs3925/1303;
oldlabels/IDs exact. Originalcohort stillrestricted, no MAP-native anchorclaim.
Expanded inputs/retraining pending; not accuracy or solecause proof/promotion.

MAP freshness: `MAP_FRESHNESS_20260912.md`. Currentmedianage1h, measuredgap7h.
Full-own gain0/7fresh/older bothroles; sourceharm4/3. Not stale-only explanation
or fresh-only retraining. No runtime/promotion, patients overlapstrata.

MAP cross-model comparison: `MAP_CROSS_MODEL_20260912.md`. Saved matchedlabels,
common20kCI. Treefullsummary vs ownJEPA0/7gainboth(sourceharm1). Hospital
medianpaired -.11373; not significant. Summaryrepair beatsoldtree partly,
not established betterJEPA. Inputs/capacity differ; no training/runtime/promotion.

MAP summary components: `MAP_SUMMARY_COMPONENTS_20260912.md`. Identicalfull
lag inputs, samewidth summaries; ownsummary gains3/7source/hospital overlag.
Other support thennumeric increments0/7both; fullcontrols exact7/7. Partial
repair includesownhistory, not proven crossorgan benefit. Sequential contrasts,
not equivalence/factorial; no neural/calibration/runtime/promotion.

MAP summary attribution: `MAP_SUMMARY_ATTRIBUTION_20260912.md`. Existing
support-vs-lagfull predictions gain2source4hospital, noharm; support-own0/0.
Prior summary improvement cannot be assigned to other numeric excursions;
own numerical summaries/other support/removal remain confounded. No fitting,
runtime or promotion; not workflow-only proof.

MAP event summaries tested: `MAP_EVENT_HISTORY_20260912.md`. Lagcontrols exact
7/7. Full vs lagfull gain2source4hospital, harm0; hospitalMAE10.59785->10.13827.
Full-own/support remains0/7gainboth; persistence7/7both. Partial history-
representation improvement, not cross-organ efficacy, JEPA retrain or promotion.
Quality-aware6/24h summaries, source-only selection; runtime unchanged.

Event summary quality repair: `EVENT_SUMMARY_QUALITY_20260912.md`. Invalid
finite spikes previously entered summaries; reproduced900vs70 then corrected.
Latest-known quality respected, validzero/prefix/invalidrevision tested.
Not cause of prior MAP trials (neither calls helper); no efficacy/promotion.
Rebuild summary caches before new use; historical hashes remain historical.

MAP non-neural raw control: `NATIVE_MAP_RAW_HISTORY_20260912.md`. Same roles/
labels, fixed-lag tree; full-own source0gain3harm,hospital0/0, full-support
source0gain2harm. Own median8.9998/10.3964 vs full9.4291/10.5978. Not solely
recurrent JEPA failure; not native-event equivalence or information absence.
No calibration/runtime/promotion, no separate insulin context in tree input.

Native MAP21fits completed: `NATIVE_MAP_FINDINGS_20260912.md`. Common20kdraws
x3streams: full-own source0gain5harm, hospital0/0; full-support0/0both.
Own persistence7/7source5/7hospital; full7/7source3/7hospital. Full worse
source-tune7/7. Same-method MAP subset retains sourceharm5/7; switching113/3925
not sufficient explanation. No intervals/external/runtime/promotion. Supersedes
MAP training pending; native diagnostic is not full anatomical architecture.

Bedside future labels: `BEDSIDE_FUTURE_LABELS_20260912.md`. Same5526anchors;
MAP3925pairs1303patients, HR5095/O24873/RR4814/temp3521. Future helper now
rejects nearest quality-invalid truth rather than skip;12tests. Not prior
glucose cause, no training/promotion. MAP own/full is now supported for testing.

Bedside history-support split: `BEDSIDE_HISTORY_SUPPORT_20260912.md`. Five
observations each of five vitals, pre-anchor definition; real-support0/7gain
both rich/poor groups and roles. Hospital rich70patients, not proof adequate
power or all-support failure. No trained routing/runtime/promotion.

Bedside CI numerical stability: `BEDSIDE_BOOTSTRAP_STABILITY_20260912.md`.
Common patientdraws20k x3streams all agree; real-support hospital0gain1harm,
supportlab1gain0harm. Names no longer alter draws in new comparison helper;
no gate weakening, model repair or promotion. Bound v2 report authoritative.

Bedside fit gap: `NATIVE_BEDSIDE_FIT_GAP_20260912.md`. Real-support gain3/7fit,
0tune,1newsource,0hospital; inconsistent numeric fit gain fails transfer already
inside source. New bootstrap salt flips one near-zero support-lab CI at1000
draws, same predictions; no ability change or promotion. No training/runtime.

Bedside revision check: `NATIVE_BEDSIDE_REVISION_CHECK_20260912.md`. Qualified
cache has zero duplicate source identities; invalid-revision synthetic test
preserves prior query and blocks stale restoration. Not observed failure cause,
no production repair/training/promotion.

Bedside raw-history tree: `NATIVE_BEDSIDE_RAW_HISTORY_20260911.md`. Same queries/
labels/roles, fixed-lag instead of nativeevents. Real-support0/7gainboth;
real-lab1source2hosp, persistence7both. Weak increment not exclusive to JEPA;
not no-information proof or promotion. Prior memory gradient already nonzero.

Bedside support14fits: `NATIVE_BEDSIDE_SUPPORT_20260911.md`. Same cadence,
numbers removed in train/test; controls exact7/7. Support-lab gain1source2hosp,
real-support gain1source0hosp(harm1hosp). Hospital MAE support41.658 vs
real42.328/lab42.080. No stable increment, not purely harmful cadence or proof
workflow-only physiology. No runtime/promotion.

Bedside fixed components: `NATIVE_BEDSIDE_COMPONENTS_20260911.md`. Original
replay exact7/7 after preserving float32 transforms. Neutralnumbers gain0/1
source/hospital; removal gain0/1 and sourceharm5/7. Context dependence differs
across roles; no universally helpful deletion or isolated cadence-cause proof.
No training/runtime/promotion.

Bedside rescale14fits: `NATIVE_BEDSIDE_RESCALE_20260911.md`. Source-only five
new coordinates, same cadence; controls exact7/7,44bindings. Rescaled vs lab
sourcegain1/harm0,hospital0/1; median34.618/42.328. Persistence7/7both.
Not stable increment; scale alone insufficient. No runtime/promotion.

Bedside14fits completed: `NATIVE_BEDSIDE_TRAINING_20260911.md`. Lab controls
bit-exact7/7 to previouscleanfull,44bindings verified. Bedside-lab sourcegain0/
harm0,hospitalgain1/harm3; median34.884->34.251source,42.080->42.881hospital.
Both persistence7/7. Added vitals not sufficient repair; cadence/numbers
confounded. No runtime, calibration or promotion. Supersedes trainingpending.

Expanded native bedside cache complete:207499uniquegroups, same5526queries,
median68/max385history length, no truncation. Query and history caches ready;
matched training with new source-fit support still pending. No runtime or
promotion. See NATIVE_BEDSIDE_EVENTS_20260911.md and native_bedside_tokens artifact.

Bedside quality/query rebuild complete: existing bounds flag2352events; do not
delete latest invalid evidence and carry old normal forward. Same5526queries,
all nonbedside current/mask/age exact against clean reference. Qualified support
HR5366/MAP4505/O25245/RR5195/temp4787. Expanded snapshot cache ready; native
history cache/training pending. See NATIVE_BEDSIDE_EVENTS_20260911.md. No promotion.

Bedside encoder wiring now accepts bedside_available alongside lab_available;
same-time conflicts cross sources, source identities remain namespaced.
Raw-to-token synthetic test verifies legal arrival and age. See bedside events
finding. No new neural cache/training/accuracy/promotion yet; historical code
hashes remain historical, not retroactively certified.

Bedside canonical events: `NATIVE_BEDSIDE_EVENTS_20260911.md`.1,347,546time-valid
events; same5526queries support HR5366/O25247/RR5233/MAP4507/temp4803 within6h.
11tests pass; zeros preserved, earlyentry rejected, MAP conflicts missing.
Non-temperature units are label conventions; rangequalification and native
training integration pending. No accuracy/runtime/promotion. Supersedes
canonical/anchor-support pending inventory entry below.

Bedside local inventory: `NATIVE_BEDSIDE_INVENTORY_20260911.md`.1916same
stays,3.536Mnumeric nursing rows private. HR/O2/RR/MAP/temp substantial; entry
time distinct (median5-12min), some entrybeforemeasurement. Canonicalunits,
legalanchor support and training pending. No accuracy/runtime/promotion.

Native scope verified: `NATIVE_INPUT_SCOPE_20260911.md`.41/53pooled lab slots
observed; MAP/HR/O2/RR/temp/urine absent. Only insulinrecord context, not full
administration/bedside stream. Known scope confirmed, not new rootcause or
proof additionhelps. No runtime/promotion.

Clean whole-state support: `CLEAN_STATE_SUPPORT_20260911.md`. Source-only
prototype distance near0/7full-owngainboth; farharm2source3hospital. Farrows
have extra harm but removing them not stable repair; geometry not physiology
equivalence. No trained routing/runtime/promotion.

Clean change-size audit: `CLEAN_GLUCOSE_CHANGE_20260911.md`. Larger-delta
full-own fitgain7/7 becomes1/7newsource0/7hospital. Smaller cases also harm.
Not solely stable dominance; no prospective regime claim or globalamplify
repair. Futuredefined descriptive groups only; no runtime/promotion.

Conflict actual update scope: `NATIVE_CONFLICT_UPDATE_SCOPE_20260911.md`.
All7clean checkpoints update history/mixers/state/sharedfuture/value head;
no frozen-context explanation or duplicateoptimizer parameters. Magnitude is
not causal efficacy; no runtime/promotion.

Clean future-event weighting: `CLEAN_FUTURE_EVENT_WEIGHTS_20260911.md`.
Hospital501queries490outcomes; equalizeevent thenpatient still cleanfull-own
0/7gainboth; persistence7/7. Repeatedoutcome weighting not sole explanation.
No gate replacement, training/runtime/promotion.

Clean context complementarity: `CLEAN_CONTEXT_COMPLEMENTARITY_20260911.md`.
Patientmean signederror correlation.952/.966; fullbetter~47%. Equalblend
gain0/7source1/7hospital, no stable repair. Outcomeoracle not deployable;
replicates prior portfolio pattern on corrected inputs. No runtime/promotion.

Native teacher semantics: `NATIVE_TEACHER_CONTRACT_20260911.md`. Existing
single-target equivalence extended to nonzeroanchors/translation/stopgradient;
5tests pass. No absolute-vs-delta mismatch in helper, not efficacy proof.
No training/runtime/promotion.

Conflict initial gradients: `NATIVE_CONFLICT_INITIAL_GRADIENTS_20260911.md`.
21sameinitial source probes, medianrelativechange.01333/cosine.99995. No typical
large initial gradient reversal found. Later optimization remains unmeasured;
not a proof of instability or accuracy repair. No runtime/promotion.

Conflict selection audit: `NATIVE_CONFLICT_SELECTION_20260911.md`.4/7full
old/clean select sameepoch; both hospitalharm cases inside those4. Allselected
indices match tune minima. Wrongepoch alone cannot explain; trainingpath
effect still unlocalized, no fit/runtime/promotion.

Conflict cross isolation: `NATIVE_CONFLICT_CROSS_20260911.md`.14diagonal
predictions exact. Fixed-input/preprocessing cleanweights hospital+.2903,
harm2/7; fixedweights replay effect~.002 no significanthospital gain/harm,
scale effect~2.4e-7. This repair regression follows trained/selected weights;
training trajectory versus selection still unresolved. No runtime/promotion.

Conflict exposure stratification: `NATIVE_CONFLICT_SUBGROUPS_20260911.md`.
Hospital149patients without changed scored inputs still cleanfull-oldfull
harm4/7(gain1), median+.926; changed40patients0/0. Harm not only edited
histories. Weights/selection/fitted event transforms remain confounded; fixed
cross-evaluation needed. No fitting/runtime/promotion or sole-cause claim.

Native conflict28fits complete: `NATIVE_CONFLICT_TRAINING_20260911.md`.
Same clean labels across old/clean own/full. Clean-full vs old-full gain0/7both,
harm1source2hospital; clean-full vs clean-own gain0/7both. Both persistence7/7.
54bindings verified. Correct replay not sufficient accuracy repair; no runtime,
calibration or promotion. Supersedes token/training-pending entries below.

Query replay repair implemented: `NATIVE_CONFLICT_REPLAY_20260911.md`.
5526private schedules rebuilt,817queries344patients change,3435obsolete/conflict
event occurrences removed. Prefix/resolution/zero/remeasurement tests pass.
Canonical table exact; no new query events. Tokens and matched neural training
still pending, no runtime/accuracy/promotion claim. Supersedes unimplemented below.

Actual native exposure: `NATIVE_SCHEDULE_CONFLICTS_20260911.md`. Hash-matched
schedule/tokens contain514/5526queries (238patients) with unmasked known-source
disagreement; glucose265,bicarb187,Mg1. This is input exposure, not error
attribution or final source-mask exposure. Legal invalidation/replay still
unimplemented; no training/runtime/promotion. Supersedes raw-only scope below.

Later native disagreement quantified: `NATIVE_CONFLICT_ARRIVALS_20260911.md`.
1551rawgroups have eligible-age single-valued later arrivals conflicting with
known prior sources. Per-arrival encoder cannot detect prior disagreement.
Not exact schedule exposure/error; requires legal invalidation/replay, cannot
delete earlier observations using future knowledge. No native/runtime repair yet.

Event variable identity checked: `EVENT_VARIABLE_IDENTITY_20260911.md`.
646120cache rows no collisions; silent same-ID variable change now rejected.
26tests. Ingestion validation, not observed forecast cause or accuracy repair.

Event summary conflict policy repaired: `EVENT_SUMMARY_CONFLICTS_20260911.md`.
As-of disagreeing events excluded from summaries; 274/648queries affected in
6/24h (Mg1).25tests pass. No accuracy or native-stream consistency claim;
old summary features need rebuilding for new training, runtime unchanged.

Joint conflict labels and accuracy: `GLUCOSE_CONFLICT_REPAIR_20260911.md`.
Input/label masks exact; total glucose92/bicarb118pairs rejected. Sameclean-test
tree refit: full-oldfull0/7gainboth, own-oldown1/7hospital; full-own0/7hospital.
Contract repaired, accuracy not stably repaired; not native JEPA retraining.
No runtime/promotion. Use combined label cache, not future-only intermediate.

Snapshot conflicts repaired: `SNAPSHOT_CONFLICTS_20260911.md`. As-of latest
source versions must agree at latest measured time; otherwise missing, no old
substitution. New cache retained values exact, bicarb210/glucose144 histories
affected, Mg1.22tests; no accuracy/promotion. Rebuild label current eligibility
before training; native event/summary paths not claimed fully harmonized.

Future-label conflict defect repaired: `FUTURE_LABEL_CONFLICTS_20260911.md`.
Same-time disagreeing sources no longer chosen by ID; nearest conflict rejected,
not skipped. New private cache glucose70/bicarb78 pairs removed, Mg0. Retained
truth/input exact;4tests pass. No retraining/accuracy claim/runtime/promotion.
Old helper hashes are historical; current snapshot conflict policy unresolved.

Mg loss-aligned offsets: `MAGNESIUM_LOSS_CALIBRATION_20260911.md`. Source-tune
L1/Huber, own/full matched; full-L1own-L1 gain0/7both. Hospital partialoffset
gain not stable repair; not end-to-end loss test. No runtime/promotion.

Mg actual auxiliary gradients: `MAGNESIUM_AUXILIARY_GRADIENT_20260911.md`.
21source-fit probes, weighted norm~.177primary, cosine~.011, opposed10/21;
not absent supervision nor evidence scaling repairs. Local selectedcheckpoint
geometry, not sole cause or optimizer repair. No runtime/promotion.

Mg auxiliary exposure: `MAGNESIUM_AUXILIARY_EXPOSURE_20260911.md`. Actual
sampler coefficients ~.009-.011/aux vs1primary, some sparse patient coverage
82-89%. Not gradient magnitude or Mg gap cause. Mg-primary run is not balanced
alltarget training; no runtime/promotion or reversal of earlier experiments.

Mg event-weight training: `MAGNESIUM_EVENT_WEIGHT_20260911.md`. Patient then
future-event equal weights, same query evaluation; old controls reproduce.
Full-oldfull0/7gainboth, full-own0/7gain1/7harmboth. Not a stable weighting
repair or neural-sampler test; no runtime/promotion.

Mg anchor overlap: `MAGNESIUM_ANCHOR_OVERLAP_20260911.md`. Common hospital
81events/49patients; new full-own and increment-change gain0/7bothroles. Broad
hospital mean sign survives leaveoneout6/7, not universal benefit. Timing repair
not established; smaller support and different training remain. No runtime.

Mg own-availability anchors: `MAGNESIUM_NATIVE_ANCHOR_20260911.md`.2510pairs,
809patients, current medianage0 vs3.167; future ID overlap636 only. Same original
patient roles. Tree full-own gain0/7source1/7hospital; full-persistence7/7both.
Not stable repair; composition/time confounded, no JEPA retrain/runtime/promotion.

Mg event summaries: `MAGNESIUM_EVENT_HISTORY_20260911.md`. Same roles plus
6/24h legal event summaries; lag baselines reproduce. Full-own gain0/7 both,
full-lagfull gain0/7 both. Both persistence7/7. These summaries do not repair
increment; not information impossibility or all event-encoder failure. No runtime.

Mg nonlinear raw tree: `MAGNESIUM_TREE_HISTORY_20260911.md`. Full-own0/7gain,
harm1/7source2/7hospital; both persistence7/7. Raw linear limitation alone not
repair; fixed-lag scope, no information-impossibility/runtime/promotion claim.

Mg raw ridge diagnostic: `MAGNESIUM_RAW_HISTORY_20260911.md`. Same roles,
fixed-lag raw no latent: full-own0/7gain,harm7/7source5/7hospital. Bypassing
JEPA compression not repair; not no-information proof/native-stream equivalent.
No runtime/promotion.

Mg rate-number ablation7fits: `MAGNESIUM_RATE_SUPPORT_FINDINGS_20260911.md`.
Full lab/presence retained; numeric removed. Vs own/numeric0/7gainboth,
hospitalharm1vsnumeric. Not rate-number removal repair; no runtime/promotion.

Mg fixed-partition14fits: `MAGNESIUM_SAMPLER_ORDER_FINDINGS_20260911.md`.
Seed7 predictions exact. Whole-own0/7gainboth; hospital variance ratio1.15x
vs mixed2.80x. Partition-linked component suggested, not variance attribution
or repair. No runtime/promotion; optimization repeats not seven patient splits.

Mg initialization correction: `MAGNESIUM_INITIAL_STATE_20260911.md`.
All2285initial tensors exact across7seeds. Ensemble variance is not random
initialization; partitions/sampling/source stats/selection remain confounded.
No fitting/runtime/promotion.

Mg ensemble variance: `MAGNESIUM_ENSEMBLE_VARIANCE_20260911.md`. Common103
hospitalpatients, whole prediction variance2.80xown. Averaging MSE improves,
MAEwhole .182428 vs own .181701 CIcross0. Variance part, not fullrepair/runtime.

Mg unit exposure resolved: `MAGNESIUM_UNIT_EXPOSURE_20260911.md`. Conflict
union1window/1patient, hospital0all7. Fit4seeds affected; indirect effect not
measured. Not sufficient universal gap explanation; no conversion/runtime change.

Mg raw unit audit: `MAGNESIUM_RAW_UNITS_20260911.md`.20906values exactraw;
13system/interface unit conflicts (1.5-1.9),1future/1current pair touched.
Real ambiguity, not proven conversion error; history/accuracy impact pending.
No guessed conversion or runtime change.

Mg future IDs checked: `MAGNESIUM_LABEL_IDENTITY_20260911.md`.2366future
values/times match raw;18%equal values, draw spacing median9.017h vs anchor
horizon6.033h. No timestamp-reuse defect established or repair; age supplied.

Mg context dropout7fits: `MAGNESIUM_CONTEXT_DROPOUT_FINDINGS_20260911.md`.
Fixed50%own-only batches, inferencefull; source .216822,hospital .183796.
Vs matchedwhole/own0/7gain0/7harmboth. No stable repair/runtime/promotion.

Mg history depth: `MAGNESIUM_HISTORY_DEPTH_20260911.md`. Median3events,
31.95%<=2; fixed1-2/3-4/5+ bins all0/7whole-own gain bothroles. No supported
shallow-history rescue/routing claim; no fit/runtime/promotion.

Mg native support contract: `MAGNESIUM_SUPPORT_CONTRACT_20260911.md`.
41observed variables,35with30fitpatients; native and snapshot >0 masks exact
all7. No erroneous channel exclusion found. No training/runtime/promotion.

Mg patient influence: `MAGNESIUM_PATIENT_INFLUENCE_20260911.md`. Whole benefit
~44-54%patients, shrunk hospital~51-57%; some mean signs single-patient fragile.
Not universal benefit masked solely by outliers. Descriptive, no exclusions,
new gate, runtime or promotion.

Mg increment shrink: `MAGNESIUM_INCREMENT_SHRINK_20260911.md`. Source-tune
alpha.1-.8; blend vs own0/7gainboth, vs whole2/7source0/7hospital.
Partial attenuation benefit, not stable extra-context repair. No runtime/promotion.

Own event isolation checked: `NATIVE_OWN_EVENT_ISOLATION_20260911.md`.
Inactive nonzero events/timestamps leave state and uncertainty exactly unchanged;
mask wiring verified,2tests pass. No cadence contamination at this filter path.
Not cohort-independence/forecast repair; no training/runtime changes.

Mg auxiliary14fits: `MAGNESIUM_AUXILIARY_FINDINGS_20260911.md`.
Fixed.25real auxiliary vs primary-only gain0/7source/hospital; versus shuffled
source1gain1harm, hospital0/0. Mg-primary auxiliary is not stable repair.
No runtime/calibration/promotion; other-target retention not measured here.

Mg sampler audited: `MAGNESIUM_EXPOSURE_20260911.md`. Fit437-453patients;
whole selected99.3-100%patient exposure, epoch32~99%rows/events. No broad
sampler-omission explanation; independent diversity/subgroup support unresolved.
No fit/runtime/promotion; glucose2250experiment not identical Mg cohort.

Mg matched gap localized: `MAGNESIUM_MATCHED_GAP_20260911.md`.
Whole-own fit gain6/7 (.17924 vs .20035), tune/evaluation/hospital0/7.
Scores and patient independence verified. Extra-context advantage disappears
already on new source patients, not only cross-hospital. No sole-cause proof,
refitting, runtime, calibration or promotion.

Matched magnesium-primary7fits: `NATIVE_MAGNESIUM_MATCHED_FINDINGS_20260911.md`.
Same objective/selection as own; whole .218283/.184960 vs own .217013/.182700.
Whole-own gain0/7harm0/7 both; both persistence7/7. Task-priority mismatch is
not sufficient explanation. No equivalence/no-information claim, runtime or
promotion. Incremental extra-context generalization remains unresolved.

Magnesium own7fits: `NATIVE_MAGNESIUM_OWN_FINDINGS_20260911.md`.
Whole-own gain0/7source/hospital, source harm2/7; both persistence7/7.
Own .2170/.1827 vs whole .2234/.1814. Different task-loss/selection explicit;
supervision repair not proven extra-context gain. No runtime/promotion.

Native auxiliary constant control: `NATIVE_AUXILIARY_CONSTANT_20260911.md`.
Glucose/Mg/K beat source-fit constant delta7/7source/hospital. BUN0/7source,
6/7hospital; creat0/7both. Beyond fixed drift, not yet own-history incremental
coupling. No neural training/runtime/calibration/promotion.

Native truth auxiliary14fits: `NATIVE_TRUTH_AUXILIARY_FINDINGS_20260911.md`.
Other-target actual supervision improves glucose-only heads7/7 for listed labs;
hospital Mg/K persistence7/7, shuffled5/7,6/7; BUN6/7,5/7 respectively.
Glucose shuffled0/7, several other targets still fail. Partial supervised repair,
not established cross-organ increment or promotion; no runtime/calibration.

Source-tune curve replay: `NATIVE_RETENTION_CURVE_20260911.md`.
Seven32epoch trajectories reproduce primary logs exactly. No positive epoch
numerically preserves all17-20supported targets; not a statistical harm claim.
Selection-only strict repair unsupported; no runtime/calibration/promotion.

Selection-scope diagnostic: `NATIVE_SELECTION_SCOPE_20260911.md`.
Glucose-only source-tune improves primary but target-equal scaled score worsens
5/7; selected retention4/7, uniform1/7. Descriptive17-20targets, not clinical
gate. Per-epoch other-target curves missing, so better epoch unproved. No runtime.

Frozen sharedfuture14fits: `NATIVE_FROZEN_FUTURE_FINDINGS_20260911.md`.
Whole source34.8408->33.9582 but hospital42.4007->42.7113; no stable matched
own gain. Other-target effects mixed. Rollback sensitivity did not predict
trained repair; shared weights exact, encoder adapted. No runtime/promotion.

Update block localization: `NATIVE_UPDATE_BLOCKS_20260911.md`.
Source-only seven saved checkpoints; sharedfuture restoration Mg/Ca gain7/7,
phosphate6/7, glucose0gain/0harm but BUN harm6/7. Not solely encoder drift;
coadapted-weight sensitivity, not trained repair. No runtime/promotion.

Private update14fits: `NATIVE_PRIVATE_UPDATE_FINDINGS_20260911.md`.
All52nonprimary outputs exact initial, but whole beats originalown0/7both;
privateown hospital harms4/7. Whole-privateown hospital2/7 partly worsened
comparator. Isolation works, generalization not repaired.105bindings verified;
no runtime, calibration or promotion. Frozen event projection caveat explicit.

Same-weight retention control: `RETENTION_SAME_WEIGHT_20260911.md`.
Changing only mask reduces opposition12->9/21 at uniform weights and8->6/21
at selected weights.42old probes reproduce exactly. Reference selection changes
local conflict but is not a demonstrated generalization repair; no runtime.

Retention gradient diagnosis: `RETENTION_CONFLICT_DIAGNOSIS_20260911.md`.
Source-fit42 probes: uniform opposing12/21, selected6/21; median cosine
-.1878/+.2070. Local conflict exists but reduced conflict did not fix the
prior whole-own accuracy gap. Separates primary-only multi-target retention
failure from unresolved source generalization. No training/runtime/promotion.

Selected retention7fits: `NATIVE_SELECTED_RETENTION_FINDINGS_20260911.md`.
Source-qualified references improve BUN/creatinine versus uniform7/7hospital,
but remain worse than glucose-only partly; calcium/phosphate lose preservation.
Glucose whole-own still unstable.82bindings verified; no universalrepair,
runtime/calibration/promotion. Supersedes selection-training-pending below.

Reference quality selection complete: `NATIVE_RETENTION_SELECTION_20260911.md`.
Source-tune only pairedCI versus persistence selects bicarbonate/K7/7, Mg6/7;
no BUN/creatinine. Fixed policy, not hospital-picked exclusion. Selected-retention
retraining remains pending; no runtime/promotion or accuracy-repair claim.

Retention repair7fits: `NATIVE_RETENTION_REPAIR_FINDINGS_20260911.md`.
Source-only fixed output retention reverses hospital Mg/phosphate harm7/7,
Hct6/7, glucose gain1/7no significantharm; BUN/creatinine harm7/7 by preserving
poor initial outputs. Real interference-control benefit but not universalrepair
or stable whole-own gain.63bindings verified; no runtime/promotion.

Native other-target retention: `NATIVE_GLUCOSE_RETENTION_20260911.md`.
Glucose-only shared updates leave208other value-head tensors exactlyunchanged,
but hospital Mg/phosphate harm7/7, Ca/Hct6/7. BUN/creat improve yet remain above
persistence. Candidate multi-target retention failure, NOT proof of glucose
incremental root cause or release regression. V2scores reproduce; no runtime.

Initial-versus-final native glucose: `NATIVE_GLUCOSE_INITIALIZATION_20260911.md`.
Initialwhole beats initialown hospital7/7, but not source. Both fine-tuned
models improve hospital (whole6/7, own7/7); relative ordering reverses as own
improves more. Not simple absolute training damage or universally good initial
context. Fit-gap remains, no from-scratch/causal/runtime/promotion claim.

Student training-mode14fits complete: `NATIVE_GLUCOSE_MODE_FINDINGS_20260911.md`.
Existing dropout0.1 restored online, EMA eval retained. Whole-old gain1/7both
roles, hospital harm1/7; whole-own0/7both(newsource harm3/7).68bindings and
original predictions verified. Not a sufficient regularization repair; no runtime.

Source-aware history28fits complete: `NATIVE_GLUCOSE_SOURCE_TRAIN_FINDINGS_20260911.md`.
Whole source vs zero gain1/7source,2/7hospital with harm1/7each; versus matched
own0/7both. Hospital42.8721->42.6263, ownsource42.1301.86bindings verified.
Source-name preservation alone does not repair context generalization; no
runtime/calibration/promotion. Supersedes preparation-only entry below.

Source-aware glucose history preparation: `NATIVE_GLUCOSE_SOURCE_INPUT_20260911.md`.
Raw-source indicators aligned to native group IDs, without changing old tokens
or future labels. Six unittest checks pass. Source-aware training is NOT yet
completed; current-source offset failure does not test this representation.
No runtime/calibration/promotion change.

Initialization provenance checked: `NATIVE_PRETRAINING_PROVENANCE_20260911.md`.
Checkpoint-referenced CURRENT file patient/stay/hospital overlap0, native
identity mapping verified. Current580980/3291 vs metadata563736/3286 lacks
historical hash, so actual pretraining membership still uncertified, not proven
leakage. Reused evaluation remains separate. No runtime/promotion change.

Delayed-arrival accuracy stratified: `NATIVE_OUT_OF_SEQUENCE_ACCURACY_20260911.md`.
Whole-own gain0/7 both groups/roles. Hospital delayed numerical benefit6/7;
no-delayed harm5/7. Failure not concentrated in suspected delayed group, so no
principal-cause/deletion rationale. Association only, no runtime/promotion.

Delayed-measurement regime quantified: `NATIVE_OUT_OF_SEQUENCE_FINDINGS_20260911.md`.
Window-exact V2:2,392/5,526histories any older arrival;2,032glucose. Legitimate
delayed evidence, not duplicate/error proof. Encoder has age but same update
mechanism; prediction effect untested. No reordering/deletion/runtime change.

Known-source offset accuracy tested: `GLUCOSE_SOURCE_OFFSET_FINDINGS_20260911.md`.
Current-source-only median residual correction vs constant/shuffled0/7source
and hospital. Whole hospital42.4007->42.5476, matchedown41.3413. No global
source-bias repair, source-aware history remains distinct. No runtime change.

Glucose source provenance audited: `GLUCOSE_SOURCE_SWITCH_FINDINGS_20260911.md`.
1,798/5,438pairs switch bedside glucose/glucose labels. Hospital switch-group
whole-own numerical harm7/7(significant1), same-source gain0/7; source same
also harm3/7, so not solecause. Future eICU extractor retains source_test_name;
no input/cache/weight overwrite, no future-source routing or promotion.

Native glucose source-support strata: `GLUCOSE_NATIVE_SUPPORT_FINDINGS_20260911.md`.
Source-defined current/age/recorded-rate joint bins, >=30 distinctfitpatients:
whole-own gain0/7source/hospital even in supported bins. Rare combinations of
these3features alone do not explain failure; high-dimensional support unresolved.
Not a routing rule, delivered-treatment claim or runtime/promotion change.

Native projection scale diagnostic: `NATIVE_PROJECTION_COMPONENTS_FINDINGS_20260911.md`.
Source-fit first-linear lab centered RMS1.547 vs missing-age.326; missing-age
uncentered2.833 mostly constant. No panel-dependent scalar normalization or
proof of numerical drowning. Descriptive, no forecast/causal/runtime repair.

Native glucose update sensitivity: `GLUCOSE_NATIVE_CADENCE_FINDINGS_20260911.md`.
Frozen whole checkpoints, only observed-glucose groups update state; final
context unchanged. Original predictions reproduced. Skip-vs-original gain0/7
all roles; new-source MAE34.8408->36.0343,hospital42.4007->42.7597.
Not a repair; removes event content as well as cadence, so no isolated cadence
cause or train-matched conclusion. No runtime/calibration/promotion change.

Native glucose scale repair tested: `GLUCOSE_NATIVE_CALIBRATION_FINDINGS_20260911.md`.
Source-tune patient-weighted offset and affine delta calibration, both own/whole.
Whole versus equivalently corrected own gain0/7source/hospital for both methods.
Whole hospital MAE42.4007->42.5011offset/42.5062affine. Whole mean bias is already
smaller than own despite worse MAE; no uniform bias/scale explanation or repair.
Original bindings/scores reproduced. No neural/runtime/interval/promotion change.

Native glucose failure localized: `GLUCOSE_NATIVE_FIT_GAP_FINDINGS_20260911.md`.
Frozen 21-fit predictions: whole-own fit gain6/7, new-source numerical harm7/7
(significant2/7), hospital gain0/7. All45 original bindings and held-out/tune
scores verified; roles patient-disjoint. Extra treatment numbers fail stable
increment even in fit (gain1/harm2). Earliest measured failure is source
generalization, not solely external shift. This is not a unique biological
root-cause proof. No new training, runtime, intervals or promotion.

Glucose native21fits complete: `RATE_GLUCOSE_FINDINGS_20260911.md`.
5,438pairs/1,740patients; hospital own41.4025 vs whole-events42.4007 vs
persistence77.3038. Real-support gain0/7source1/7hospital; real-own0/7both,
source harm2/7. Both persistence7/7; no stable incremental repair.45bindings
verified. No runtime/calibration/promotion; not solely potassium evidence now.

Additional real outcomes at fixed rate queries built:
`RATE_MULTITARGET_SUPPORT_20260911.md`. Glucose5,438pairs/1,740patients,
bicarbonate4,539/1,483; old potassium labels reproduce. No new training yet;
enables target comparison rather than extrapolating K-only failure to allbody.
Same potassium-selected queries/roles; no runtime/promotion.

Native rate scaler audit: `NATIVE_RATE_SCALE_FINDINGS_20260911.md`. Normalized
IQR .60-.85fit/.86-1.15hospital, not nearzero numeric collapse. Hospital >5sigma
tails .86-3.00% vs .32-.66%fit; shift exists but no error-causation proof.
No fitting/runtime change or posthoc clipping adopted.

Real-label projection14fits complete: `NATIVE_REAL_PROJECTION_FINDINGS_20260911.md`.
Fast projection vs original native gain0/7allgroups; harm3/7source both and
1/7hospital bracket. Hospital .389022->.394157. Real-support increment still
unstable. Synthetic success did not transfer to real accuracy; do not deploy
fast LR as a repair.48bindings verified, original scores reproduce. No runtime.

Known-signal optimization repair: `NATIVE_PROJECTION_FINDINGS_20260911.md`.
Observation projection lr1e-3 vs rest1e-4; synthetic real-support/whole/persistence
gain7/7all source/hospital groups, hospital MAE .402314vs support.580522.
34bindings verified. Demonstrates synthetic-path learnability and optimization
contribution, NOT real treatment accuracy. Real-label matched refit remains
next; no runtime/calibration/promotion changes.

Synthetic latent probes complete: `SYNTHETIC_LATENT_FINDINGS_20260911.md`.
Source-fit fixed Ridge probes: new-source sign correlation .1606belief/.0100
context/-.0229future. Weak linear recovery already in state/fusion, not only
value head; no information-absence proof. No neural/runtime update or promotion.

Same-path synthetic14fits complete: `NATIVE_SYNTHETIC_FINDINGS_20260911.md`.
Known0.5event increment real-support gain3/7source,4/7hospital all, tiny median
MAE .580313vs.581786hospital; no gain over frozen whole0/7.33bindings verified.
Known-signal uptake itself limited, so cannot blame real signal absence alone.
Not biological accuracy/promotion; runtime unchanged. Optimization/path cause
not yet uniquely localized. Supersedes positive-control-pending notes below.

Native treatment fit-gap: `NATIVE_FIT_GAP_FINDINGS_20260911.md`. Six/seven
fit paired real-support effects <.0001absolute; selected real/support fit MAE
.373898/.373873. Not purely hospital transport failure; pathway learnability
versus weak conditional information unresolved. Old scores reproduced, no fit
or runtime change. Same-native-JEPA injected-signal control remains untested.

Native memory gradient probe complete: `NATIVE_MEMORY_GRADIENT_FINDINGS_20260911.md`.
32source-fit patients/seed; older>=10active updates contribute nonzero observed
gradients (K48.9-75.9%, insulin17.9-43.8% total). Not literal old-history erasure,
not proof of useful memory or causal sensitivity. No training/runtime change.

Clock14-fit accuracy completed: `NATIVE_CLOCK_TRAIN_FINDINGS_20260911.md`.
Corrected real vs original native hospital gain0/7both groups; MAE .389022->
.387887all/.426119->.425551bracket. Source gain1/7 with all-group harm1/7.
No stable repair; time collision alone not principal-cause evidence.50bindings
verified; old checkpoints reproduce. Supersedes clock-retraining-pending notes.
Runtime/calibration/promotion unchanged.

Clock-corrected native cache built:36,192/128,236tokens change only two clock
columns; signed pre-ICU offsets retained, three tests pass. Original cache
unchanged. `NATIVE_CLOCK_FINDINGS_20260911.md`. Both lower/upper clipping matter;
prior aggregate collision count is not exclusively post168h. Retraining pending.

Native clock collision quantified: `NATIVE_CLOCK_FINDINGS_20260911.md`.
69,164/263,668 active transitions lose positive elapsed time in encoded clock;
3,824/5,526queries affected.12h late-event collision reproduced. This is a
representation limit, not established error cause; matched unbounded-time
retraining pending. No runtime/calibration/promotion change.

Native14-fit accuracy complete: `NATIVE_RATE_JEPA_TRAIN_FINDINGS_20260911.md`.
Native-real improves snapshot whole0/7everygroup; hospital bracket harm1/7.
Hospital MAE .389022 vs snapshot .386973; treatment-number versus support
gain0/7allgroups. Both retain persistence benefit7/7, no full repair.32bindings
verified. Native training is no longer pending; older readiness notes below
are superseded. No runtime/calibration/promotion change.

Native training runner wired with --native-events; comparison frozen in
`NATIVE_RATE_JEPA_TRAIN_PROTOCOL_20260911.md`. Same source channel eligibility,
patient roles and labels; real/support numbers; separate private output paths.
Seven focused tests and CLI import pass. Training not yet launched; next action
is the full14-fit contrast. No accuracy claim or runtime change.

Native encoder integration ready: `NATIVE_RATE_JEPA_ENCODER_READY_20260911.md`.
Same filter activates rate-only events; padding inert, query context exact.
Seven focused tests; eight-query real forward finite, max165steps/178features.
Not accuracy validation; native training still pending, runtime unchanged.

Native event tokens built: `NATIVE_RATE_JEPA_TOKENS_FINDINGS_20260911.md`.
128,236unique groups,75,552lab,54,198rate,52,402rate-only,282inactive. New-event
masks, not LOCF repeats; six focused tests pass. Native neural training pending;
must allow rate-only steps explicitly, no runtime/calibration/promotion yet.

Native schedule V2 materialized and round-trip verified: private event table
retains units/measured timestamps, ragged group indices saved and hash-bound.
Counts unchanged; `NATIVE_RATE_JEPA_SCHEDULE_FINDINGS_20260911.md`. No trained
native sequence yet. Rate record time remains distinct from availability.

Native rate-query schedule built/tested:
`NATIVE_RATE_JEPA_SCHEDULE_FINDINGS_20260911.md`. Median42native event groups
per48h history,p9090,max165;5,296/5,526queries exceed10groups. Four contract
tests pass, no training yet. Extra events not automatically extra predictive
information; lab availability and insulin record-time provenance differ.

Event-only step-mask preflight: `RATE_JEPA_EVENT_MASK_SUPPORT_20260911.md`.
220/38,431known-rate history positions skipped;133queries, only7hospital
patients/10queries. Not sufficient for subgroup efficacy. This is fixed-grid
mask support, not native event replay. No training/mask/runtime/promotion.

Frozen event-stage localization: `RATE_JEPA_EVENT_STAGES_FINDINGS_20260911.md`.
Numerical-event removal relative RMS shift hospital belief .0124,context .0075,
future latent .0071. Already small at belief, not a large state signal lost only
at final decoder. Descriptive coordinates, no biological information fraction,
no demonstrated repair/runtime/promotion.

Frozen event-path check complete: `RATE_JEPA_EVENT_PATH_FINDINGS_20260911.md`.
New event weight norms .028-.133 (not frozen); numbers-off hospital forecast
shift ~.001, all-off ~.0022. Exact baseline MAE reproduction. Weak end-to-end
use located, not proof of bottleneck cause or irrelevant treatment. No fitting,
runtime/calibration/promotion. Do not blindly amplify event weights.

Recorded event-history JEPA14fits completed:
`RATE_JEPA_EVENTS_FINDINGS_20260911.md`. Real vs original gain1/7source,0/7
hospital; real vs support0/7all and hospital harm1/7. MAEs nearly unchanged.
Same lab-supported update times, no treatment-only steps or complete dose
semantics. Event path utility unresolved, no runtime/calibration/promotion.

Separate rate-JEPA event history built:
`RATE_JEPA_EVENT_HISTORY_FINDINGS_20260911.md`.46,657exact history snapshots,
30,993rate-supported including1,939zeros;5,525/5,526query support. Labs unchanged,
rate/count/age remain separate. Not yet encoder training; infusion availability
unverified, no runtime/calibration/promotion. Private hash-bound event cache.

Fixed rate-JEPA seven-seed mean tested: `RATE_JEPA_ENSEMBLE_FINDINGS_20260911.md`.
Common hospital-only evaluation avoids source cross-seed leakage. Whole MAE
.381275all/.409151bracket vs own .399965/.434379; whole-own CI upper .002784/
.000706, both cross0. Persistence beaten, no full incremental repair or
runtime/calibration/promotion. Do not redefine the gate around near misses.

Current rate-query comparison input integrity verified:
`RATE_JEPA_INPUT_INTEGRITY_20260911.md`. Body/query/label/seven partition hashes
match; current role patient overlap zero, duplicate paired queries zero.
Inherited checkpoint pretraining independence remains uncertified. No repair
of forecast error, fitting, runtime, or promotion is implied.

Rate-query source sampling replay: `RATE_JEPA_SAMPLING_FINDINGS_20260911.md`.
Rows86.95-90.47%, future events90.52-92.30% seen at32epochs; whole selected
epochs4-29, none32. Not a convergence proof, but no low-coverage/late-boundary
pattern warranting automatic budget extension. No fitting/runtime/promotion.

Rate-JEPA fit-gap diagnosis complete: `RATE_JEPA_FIT_GAP_FINDINGS_20260911.md`.
Whole-vs-support gain5/7fit (harm1/7),1/7new source,2/7hospital. Numerical
increment attenuates already within source, but no wholesale median reversal.
Support-only hospital gain0/7 remains a separate transport limitation.
Exact evaluation reproduction; no fitting/runtime/calibration/promotion.

Train-matched rate-JEPA support control7fits complete:
`RATE_ANCHOR_JEPA_SUPPORT_FINDINGS_20260911.md`. Support-only beats own5/7source,
0/7hospital; whole beats support1/7source,2/7hospital all,0/7hospital bracket.
Whole/support hospital MAE .386973/.400350. Source shared gain is not uniquely
numerical physiology; frozen sensitivity does not establish trained increment.
No information-absence claim, runtime/calibration/promotion change.

Frozen rate-JEPA component diagnosis completed:
`RATE_JEPA_COMPONENT_FINDINGS_20260911.md`. Neutralizing other lab numbers
harms5/7source and3/7hospital; exact-age standardization has tiny effects.
Whole hospital MAE .386973 becomes .407274 without numerical context. This
is off-manifold sensitivity, not train-matched proof or causal physiology.
Source incremental gain versus cross-hospital consistency remains the gap;
no runtime, calibration, training or promotion change.

Frozen neural hospital audit: `RATE_JEPA_HOSPITAL_FINDINGS_20260911.md`.
193patients/15hospitals, only1site>=30patients. Whole-own mean gain7/7 but
single-hospital removal reverses2/7. Unsupported sites not efficacy failures.
No fitting/promotion; source increment versus broad transport still unresolved.

Rate-anchor JEPA14end-to-end fits complete:
`RATE_ANCHOR_JEPA_FINDINGS_20260911.md`. Whole-vs-own source gain5/7all,6/7bracket;
hospital1/7both, no harm. Whole hospital .386973all/.413968bracket; persistence
7/7both arms allgroups. Shared-lab source increment, transfer incomplete;
no treatment input, runtime/calibration/promotion change.

Shared encoder inputs at rate queries rebuilt:
`RATE_ANCHOR_BODY_INPUTS_20260911.md`. Same5526pairs/1763patients,40supported
current lab variables in53-slot schema; potassium labels reproduce. No neural
training yet; treatment separate, bedside missing. No runtime/promotion.

Snapshot repair accuracy21refits complete: `SNAPSHOT_REPAIR_ACCURACY_20260911.md`.
Old/own controls reproduce; significant gain/harm0/7 every group. Hospital
bracket MAE .406434->.407396. Keep correctness repair, not main accuracy cause.
No runtime or promotion change.

Simultaneous snapshot defect repaired: `SIMULTANEOUS_RATE_FINDINGS_20260911.md`.
Conflicting equal-time rates no longer picked by source_id; now unknown.
Only1/5526native-rate query windows affected, so not presumed principal cause.
Candidate extractor changes require fresh fit; runtime unchanged.

Frozen rate-anchor influence: `RATE_ANCHORED_INFLUENCE_20260911.md`.
Hospital bracket-after mean benefit7/7,51.4-59.6%patients benefit, no single
patient can reverse mean. Prior significance2/7 unchanged. Not sole outlier,
still heterogeneous small increment; no fitting/runtime/promotion.

Record-anchor prediction21fits complete:
`RATE_ANCHORED_PREDICTION_FINDINGS_20260911.md`. Hospital bracket-after146patients,
real MAE .406434vsown .422643; gain2/7own,1/7support,7/7persistence. Source
gain1/7own. Partial increment not stable repair, no JEPA/runtime promotion.

Rate-triggered labels rebuilt: `RATE_ANCHORED_SUPPORT_20260911.md`.
5526paired/1763patients;4516bracket-after/1526patients. Hospital strict146patients
versus9 lab-query group. Same patient roles, new query/label task. No accuracy
test yet; retrospective record timing not verified knowledge-time. Runtime unchanged.

Rate-change timing ambiguity located: `RATE_CHANGE_BRACKET_FINDINGS_20260911.md`.
439/554post-draw recorded differences straddle current draw; only115windows/
98patients have unequal pairs entirely after draw. Hospital10windows/9patients.
Prior40patient group is record-based, not verified new intervention. No training,
runtime change or efficacy conclusion; explicit bracket classification tested.

Frozen post-draw subgroup: `POSTDRAW_RATE_FINDINGS_20260911.md`.
40hospital patients/43rows, source62-85patients. Real vs own numerical hospital
gain6/7 but significant0/7; original0/7; support1/7. Hospital MAE .450256real,
.463533own,.535persistence. No fitting/promotion; timing dilution unproved.

Timing support audit: `RATE_MEASUREMENT_ORDER_FINDINGS_20260911.md`.
Only554/3373sequence windows have observed rate change after current potassium
draw; hospital43/358. Most tested context is historical, not a new intervention
after latest draw. Not proof of absorption/causality; subgroup patient support
not yet counted. No training/runtime change.

Positive sensitivity control14fits: `INCREMENT_SENSITIVITY_FINDINGS_20260911.md`.
Artificial +/-0.5 increment detected7/7source/hospital versus nuisance. Only
current/age base, not full-history/JEPA or small-effect power. Gross learning
and scoring work; no real accuracy repair or runtime promotion.

Exact treatment support control completed7fits:
`INSULIN_SUPPORT_CONTROL_FINDINGS_20260911.md`. Numeric vs same-mask control
gain0/7source/hospital all/sequence; hospital harm1/7. Sequence hospital
MAE .343361numeric/.337547support/.336886own. Not donor-mismatch explanation
alone, not equivalence or no-information proof. No runtime change.

Transition accuracy completed14fits: `TRANSITION_ACCURACY_FINDINGS_20260911.md`.
Frozen scores reproduce; adding observed changes/reversals/last-change timing
improves original0/7 all role/groups. Hospital sequence .341000->.343361.
Representation collision real but repair not predictive gain. No runtime change.

Rate-summary collision verified: `RATE_TRANSITION_COLLISION_20260911.md`.
Identical old snapshot/range/change/span can hide different internal reversals
and last-change age. Added transition features and synthetic regression; not
yet outcome-tested, not established as principal cause. No runtime change.

Expanded sequence diagnostic completed21fits:
`EXPANDED_INSULIN_POTASSIUM_FINDINGS_20260911.md`. Patient/hospital sequence
support269-281/184. Real-vs-own/source gain1/7; hospital0/7, sequence numerical
harm7/7. Hospital sequence MAE .336886own/.341000real/.400376persistence.
Support-only explanation not sufficient; no shared-JEPA or runtime promotion.
Fixed roles and private models preserved, no new external lockbox.

Expanded pair support built: `INSULIN_POTASSIUM_PAIR_SUPPORT_20260911.md`.
3373pairs/1562patients/54hospitals have prior insulin sequence plus potassium
5-7h future/current pair;12354pairs total in rate-recorded stays. Source roles
not yet assigned, no training or promotion. Treatment availability remains
unverified; private caches retain identity for later patient-disjoint splitting.

Full eICU sequence inventory:5152patients with insulin rate sequences,
4909with changes,67/66hospitals. `EICU_INSULIN_SEQUENCE_INVENTORY_20260911.md`.
Existing diagnostic stays121patients have sequences somewhere in stay, not
necessarily at scored queries. Label-aligned expansion remains unverified;
this is not a database-wide scarcity ceiling or new lockbox. No training/runtime.

Frozen rate subgroup audit: `POTASSIUM_RATE_SUBGROUP_FINDINGS_20260911.md`.
Insulin sequence evaluation support only1-10patients/seed, all paired CIs
unsupported despite341pooled windows. Rate inputs vary and own trees use them
7/7. Not dead path, not a valid subgroup no-effect conclusion. No fitting/runtime.

Rate history follow-up completed28fits: `POTASSIUM_RATE_HISTORY_FINDINGS_20260911.md`.
341insulin/16diuretic sequences; own gain1/7vsbaseline and1/7vsplacebo,
full0/7both. Own .369221, full .369971. No stable repair, dose integration,
external evaluation or runtime changes. Other drug-rate sequences unsupported.

All-action support V2: only insulin/diuretic exposure can form pure tree leaves
in7/7; bicarbonate/Mg/K/RRT0/7. Fit exposed patients insulin110-129,diuretics10-14,
K2-5,Mg1-2. Training report now records row/patient support and empty evaluation
separately. `TREATMENT_LEAF_SUPPORT_FINDINGS_20260911.md`. No prediction change.

Rare infusion testability defect located: `TREATMENT_LEAF_SUPPORT_FINDINGS_20260911.md`.
Potassium fit8-19windows < minleaf30 all7; evalzero2/7. Synthetic perfect rare
signal blocked at30, recovered at10. No-split does not falsify potassium signal.
Pooled negative result stands; this is comparator support, not universal cause.
No real training/runtime/promotion change; rare-action claims need preflight.

Order issuance context now tested: `POTASSIUM_ORDER_CONTEXT_FINDINGS_20260911.md`.
28source fits; real vs original/placebo0/7gain and0/7harm own/full. Own MAE
.373470->.370643,full .369971->.370717. No final cancellation/dose/start/stop
inputs; eventually canceled orders retained. Retrospective order-time only,
not execution/availability evidence. No external/runtime/calibration changes.

Potassium medication scope audit: `POTASSIUM_MEDICATION_COVERAGE_FINDINGS_20260911.md`.
Two brand aliases repaired140 missed medication orders, recognized windows727
to739. Infusion remains65events/20windows: aliases do not explain prior negative
infusion fit. Orders are not administrations; cancellation finalization and
available-time semantics remain unresolved. No training/runtime changes.

Latest treatment-context accuracy test completed28 source fits:
`POTASSIUM_TREATMENT_CONTEXT_FINDINGS_20260911.md`. Real vs baseline own1/7gain
and1/7harm, full0/7; vs shuffled0/7both. Full median .369971 unchanged;
three full trees never split on treatment, others use few insulin features.
Snapshot append does not repair incremental context. Retrospective event-time,
not independently known-time; no external/runtime/calibration/promotion change.

Treatment coverage V2 distinguishes solution flow from drug rate:
`POTASSIUM_TREATMENT_SUPPORT_FINDINGS_20260911.md`. Explicit mL/h recovered
for potassium20,bicarbonate36,magnesium13 query windows, without asserting
drug delivery. Prior rate parser worked; drug-dimension restriction discarded
volume from the audit. Separate rate_channels now retains both meanings.
No concentration/availability inference, model training or runtime promotion.

Latest treatment coverage: `POTASSIUM_TREATMENT_SUPPORT_FINDINGS_20260911.md`.
Exact6174queries/1309patients infusion audit: insulin event/rate windows1081/497,
potassium20/0,bicarbonate36/0. Current lab-only contrasts do not test complete
treatment state. AKI/glucose helpers are action-scoped; infusionoffset lacks
independent availability time. Counts are retrospective support, not known-time
guarantees or absent-treatment proof. No training/runtime/promotion changes.

Latest scalar amplitude diagnostic: `POTASSIUM_EVENT_GAIN_FINDINGS_20260911.md`.
Frozen event-window trees; patient-equal nonnegative LAD gain fit on tune only,
with amplification allowed. Full gain>1 in6/7 but scaled-vs-original improvement
0/7 all arms; full MAE .369971->.370997. Large-change attenuation does not
justify global amplification. No external reads, interval/runtime/promotion.

Latest frozen change decomposition: `POTASSIUM_EVENT_CHANGE_FINDINGS_20260911.md`.
V2 fixes float boundary ties without changing pooled predictions/scores.
Large-change full MAE .717920 vs persistence1.090041, direction .906;
predicted movement .400635 vs actual1.090041. Full-own gain0/7 large,
5/7 smaller; full-persistence7/7 large,0/7 smaller. Retrospective groups
cannot establish prospective routing or clinical stability. Not a sole root
cause or forecast repair. No new fitting/external/runtime/promotion changes.

Latest within-window history: `POTASSIUM_EVENT_WINDOWS_FINDINGS_20260911.md`.
21source-only fits with pre-anchor event summaries on unchanged6174pairs.
Lag extrema omit about4.31%/20.63% supported6h/24h entries after float tolerance.
Augmented full improves original1/7, own3/7, support5/7. Partial numeric evidence,
not stable repair or principal-cause proof. No external/runtime/promotion changes.

Latest native potassium anchors: `POTASSIUM_NATIVE_ANCHOR_FINDINGS_20260911.md`.
21fixed-capacity source refits; same patient roles, potassium availability queries.
Native full-own4/7source but0/7hospital/MIMIC. Same-native-row full refit-vs-old
gain0/7all roles; own source harm2/7. Not a repair or proof glucose query gating
was the principal cause. Support shrinks source/hospital; no runtime promotion.

Latest frozen nonlinear transfer: `POTASSIUM_TREE_TRANSFER_FINDINGS_20260911.md`.
21selected source refits reproduce exactly. Full vs own0/7gain in both207patient
hospital and859patient MIMIC; both persistence7/7. Source partial tree gain does
not establish external increment; not uniquely a JEPA-compression failure.
See `ROOT_CAUSE_EVIDENCE_20260911.md` for verified contributors versus unresolved
causes. No calibration, runtime or promotion changes.

Latest nonlinear raw history: `POTASSIUM_TREE_HISTORY_FINDINGS_20260911.md`.
63source-only histogram-tree fits. Full vs own3/7gain0/7harm; support2/7gain;
JEPA2/7gain. Full MAE .370529, own .377185; persistence7/7both. Unlike ridge,
partial nonlinear incremental context recovered. Not universal information
absence or a unique encoder defect; no all-seed/external/calibration promotion.
Runtime unchanged; preserve partial positive evidence without selecting on it.

Latest matched raw history: `POTASSIUM_RAW_HISTORY_FINDINGS_20260911.md`.
105source-only ridge fits, same three-way rows; float64 authoritative (float32
solver warnings repaired before interpretation). Full vs own0/7gain3/7harm;
vs support0/7gain2/7harm; vs JEPA0/7gain2/7harm. Own/full MAE .376129/.392061.
Bypassing encoder with this linear readout does not recover stable context gain.
Not absence-of-information proof; boundary alpha4/7, finite model/grid. No
external/calibration/runtime/promotion change.

Latest train-matched support-only control: `POTASSIUM_SUPPORT_ONLY_FINDINGS_20260911.md`.
14fits complete; own controls exact. Support-only vs full-numeric gain0/7,harm0/7;
vs own gain0/7,harm1/7; persistence7/7. Median .380441 vs full .374244.
Not equivalence or proof workflow-only; removing numbers does not repair context
gain. No external/calibration/runtime promotion. Both support and numeric
increment remain unestablished beyond own history under this trained contract.

Latest potassium component diagnosis: `POTASSIUM_COMPONENT_FINDINGS_20260911.md`.
Seven frozen three-way weights reproduce. Non-target block removal harms5/7;
numeric neutralization harms1/7, exact age prediction shift median .000633.
No intervention improves full/own stably. Used pathway is not proof numerical
context adds transferable physiology; support/value interaction remains.
Need train-matched support-only control to distinguish off-manifold dependence.
No runtime, calibration, external read or promotion change.

Latest independent source partition: `POTASSIUM_THREE_WAY_FINDINGS_20260911.md`.
14fits complete; per seed785fit/262tune/262evaluation patients across whole source.
Evaluation never selects checkpoints. Both persistence7/7; whole-own gain1/7,
harm1/7, numerical gain4/7. Median own/whole/persistence .377129/.374244/.437907.
Source incremental-context instability persists after diagnostic split repair.
Not a new lockbox or sole-cause proof; no external/calibration/runtime promotion.

Latest patient-influence diagnosis: `POTASSIUM_PATIENT_INFLUENCE_FINDINGS_20260911.md`.
All21 comparisons/42 MAEs reproduce. No single-patient mean-sign reversal;
original whole benefits50-59%patients. Seven tune splits have500appearances but
194unique patients, all from original cohort; expansions fit-only. Metrics are
patient-equal, not row-weight mismatch. Next source diagnostic should independently
partition the whole source cohort into fit/tune/evaluation, without selecting
on evaluation. No new external read, unique cause, runtime or promotion claim.

Latest sampling intervention: `POTASSIUM_EVENT_BALANCED_FINDINGS_20260911.md`.
All14 matched64epoch fits complete. Uniform patient/event/query sampling retains
all labels. Significant gain over original0/7own,0/7whole; whole-own0/7.
Whole-persistence7/7 does not establish improvement over original6/7. Median
own/whole MAE .364404/.355709. Repeated outcome weights are not established as
the principal cause. No external reads, calibration, runtime or promotion.

Latest label provenance repair: `AUXILIARY_EVENT_REUSE_FINDINGS_20260911.md`.
Future source IDs retained; all53 old numerical label columns exactly reproduce.
K6174pairs/4660outcomes (24.52% excess); glucose13.97%. Legitimate different
anchors can share truth, not leakage or effective independent N. Event-balanced
sampling has now been tested above. No future ID inputs/runtime/calibration change.

Latest controlled loss repair: `POTASSIUM_MAE_FINDINGS_20260911.md`.
14matched64epoch L1 fits vs prior SmoothL1. Same JEPA/inputs/sampling. Significant
gain0/7own,0/7whole; harm0/7both. Whole-own remains0/7, persistence6/7both.
Numerical MAE alignment alone is not a demonstrated fix. No external sweep,
runtime/calibration/promotion change; retain this negative result.

Latest loss diagnosis: `OBJECTIVE_ALIGNMENT_FINDINGS_20260911.md`.
63 real-source local gradient probes across21 selected checkpoints. JEPA/value
gradient norm median .0114glucosewhole/.0245Kown/.0206Kwhole; combined direction
cosine>.9998 medians. Some opposing gradients but no strong immediate total
cancellation. Teacher delta contract matches native encoder. Not a training
trajectory/Adam analysis or proof upweighting fixes anything; prior balancing
already failed full glucose repair. No runtime/calibration/promotion changes.

Latest freshness localization: `POTASSIUM_FRESHNESS_FINDINGS_20260911.md`.
Fixed14weights, pooled reproduction exact. All3 pre-anchor age bins whole-own
0/7both domains. MIMIC3-6h old values persistence0/7, fresher bins7/7. Observation
age differs across units, but fresh-only does not repair incremental context
gain. No posthoc cutoff, weights, intervals or runtime promotion changed.

Latest frozen potassium transfer: `POTASSIUM_EXTERNAL_FINDINGS_20260911.md`.
207hospital/859MIMIC patients, same measured contract. Whole persistence7/7
both, whole-own0/7both; MIMIC harm2/7. MAE whole/own .42972/.43031hospital,
.38250/.37945MIMIC. CVICU incremental5/7 but MICU persistence0/7. Tune advantage
does not transfer generally. No posthoc unit routing, calibration or promotion.

Latest equal-capacity diagnosis: `POTASSIUM_MATCHED_ENCODER_FINDINGS_20260911.md`.
14 completed64epoch own/whole potassium-supervised JEPA fits. Same capacity,
same pairs, potassium-selected checkpoints. Whole numerical gain7/7 but
significant0/7; both persistence6/7. Not the glucose directional reversal;
do not claim all whole-context learning harms. Source-tune only63-78patients,
overlapping seeds, selection bias; no external/calibration/promotion. This
narrows target heterogeneity, not sole root cause or complete repair.

Latest own-history diagnostic: `AUXILIARY_OWN_HISTORY_FINDINGS_20260911.md`.
Fixed auxiliary checkpoints vs target-only ridge on identical source pairs:
potassium significant1/7 (versus persistence6/7), glucose0/7, creatinine2/7;
no target7/7. This narrows incremental whole-body evidence, not proof that all
cross-variable information is absent. Different model capacities; glucose-tune
checkpoint selection, no final external test. No runtime/calibration/promotion.

Latest dynamics diagnosis: `AUXILIARY_DYNAMICS_FINDINGS_20260911.md`.
Source-only constant-change control: potassium beats6/7, while bicarbonate/BUN/
calcium/hematocrit beat0/7 despite7/7 improvement over unused glucose-only heads.
Potassium retrospective large-change MAE0.647 vs persistence1.010; remaining
0.306 vs0.281 (numerical harm6/7). Future-defined groups cannot select deployment
routing. Some conditional signal exists, not cross-organ attribution or a unique
root cause. No runtime, weights, calibration or promotion change.

Latest longer supervision result: `AUXILIARY_ENCODER_BUDGET_FINDINGS_20260911.md`.
All21 seven-seed64epoch source-only fits completed with exact baseline/pilot
log reproduction. Auxiliary improves primary glucose single-target0/7 and
shuffled0/7; seed31 worse than shuffled. Other targets do learn: potassium
beats persistence6/7, shuffled4/7 on source-tune; no promotion or external test.
Local shared-context gradient probes oppose5/21 times, weighted norm ratio
median0.2544, not universal gradient conflict. Auxiliary learning is not stable
transfer to glucose. No persistent organ-state, calibration or runtime change.

Latest source-only supervision test: `AUXILIARY_ENCODER_FINDINGS_20260910.md`.
Fourteen auxiliary variables have measured current/future support at unchanged
glucose queries (49,823rows/2386development patients;2250fit/136tune per seed).
Nine matched8epoch fits: real multi-target loss improves glucose-only0/3 and
shuffled auxiliary0/3 significantly. Glucose-only logs exactly reproduce the
prior2250/64first8logs. Selected primary/real checkpoints all at epoch8, so
this is not convergence or a full rejection of joint supervision. No external
evaluation, runtime/calibration/promotion change.309tests pass.

Latest structural diagnosis: `ENCODER_PATH_DIAGNOSIS_20260910.md`.
All14 actual-time identity checkpoints have active value/JEPA gradients through
cross-variable inputs and context mixers on synthetic probes. The loaded class
is the base world-model predictor, not a persistent organ-state subclass:
separate organ arguments have exactlyzero effect with shared context fixed.
Organ information still enters context. These glucose-only supervised trials
do not test jointly supervised organ-state dynamics. This corrects scope, not
accuracy or promotion; the source fit/new-patient reversal remains the earliest
measured extra-context failure. No unique root cause has yet been established.

This file is the mandatory entry point for project status. Exact experiment
numbers remain in the linked findings and machine-readable reports.

## Active Runtime

The current release is a factual, research-only whole-body forecasting stack.
It uses a frozen shared JEPA representation, target-specific validated sources,
source-bound uncertainty calibration, and a persistence safety anchor.

The authoritative validation command is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m scripts.validate_release
```

Latest verified result for the bundled example patient:

| Item | Count |
|---|---:|
| Catalog target-horizon cells | 41 |
| Patient-level runtime cells allowed to move | 27 |
| Patient-level persistence/abstention cells | 14 |
| Missing runtime cells | 0 |
| Derived variables | 2 |

The frozen global source selector contains 28 eligible move cells and 13 global
persistence cells. The strict base runtime registry contains 27 move cells. A
patient-level runtime can activate fewer than the selector maximum when an
eligible source abstains for missing, stale, or insufficient personal input.
Therefore `validate_release` reports the actual bundled patient output, not a
contradictory global catalog. If prose disagrees about the same scope, rerun the
validator and inspect both registries before updating the claim.

## What Is Established

- The runtime can make selective target-specific forecasts across the 41-cell
  catalog and abstains through persistence where a source has not passed.
- The shared current body state contains externally reproducible information,
  strongest in acid-base, mineral, renal-metabolic, perfusion, blood-count, and
  partial liver/synthesis relationships.
- eICU-to-MIMIC transfer and NWICU evaluation show real cross-database signal,
  but source rankings vary by database and care environment.
- Personal target history is a very strong predictor. Extra whole-body context
  adds value only for selected targets and states.
- Treatment context can improve factual prediction where administration timing,
  dose, route, and rate are captured. This is not a causal treatment-effect claim.
- Prediction intervals are inseparable from their point source. A new point
  source may not reuse an old calibrator.

## What Is Not Established

- There is no universal whole-body JEPA checkpoint that outperforms the active
  target router for every organ and horizon.
- Cross-organ coupling is not generally promoted. Only individually validated
  sparse edges may influence a target.
- Sparse hormones, troponin, INR, tumor markers, and other irregular targets do
  not have reliable fixed-horizon future forecasts. Many remain observed-only,
  nowcast-only, event-risk candidates, or persistence.
- The model cannot reliably predict future clinical decisions or unforeseen
  interventions from physiology that exists before those events.
- MIMIC-IV and NWICU are no longer pristine lockboxes for newly tuned methods.
  A frozen final policy requires another untouched database such as HiRID or
  AmsterdamUMCdb.
- The system makes factual forecasts only. It does not answer counterfactual
  treatment questions and must keep `causal_claim_allowed: false`.

## Current Failure Location

The whole-catalog cross-variable-history screen found failure of its tested
raw readouts before external transfer. After a
patient's own target history is known, adding current and 1-24 hour
measurement-pure histories from the other 52 variables did not earn source
eligibility for any of 37 sufficiently supported continuous release cells;
`platelets@1h/@3h` lacked the predeclared hospital support. Corrected placebos
preserved the patient's own current physiology, and target-specific
support-balanced hospital folds avoided the earlier fixed-fold support defect.

Possible contributors remain; these are not an exhaustive causal diagnosis:

1. The frozen runtime representation conflates some input semantics because missing,
   measured-at-median, fresh, and stale observations are not fully distinct in
   the encoder contract.
2. Even raw observation-aware cross-variable inputs and causal lagged histories
   did not demonstrate reliable incremental future prediction beyond the
   target's own history for the two tested 3-hour targets. Finite readout failure
   does not establish information absence.

Appending masks and ages to a frozen readout, increasing quantile count, changing
clustering, widening latent size, or relaxing acceptance rules has not fixed this.

Combining the four closest heterogeneous near-miss sources also does not repair
the failure. A tune-frozen convex portfolio and a state-dependent expected-error
router each passed 0/4 cells. Outcome-aware source oracles passed 4/4, but source
errors were highly correlated and legal pre-anchor routing agreed with the
oracle only about 19-27% of the time. This locates the gap between retrospective
choice and prospective identifiability: after seeing the future, one candidate
is often better, but the current tabular patient state cannot reliably identify
which candidate that will be.

## Current Research Frontier

The original aligned `creatinine@6h` interpretation was too strong and has been
corrected. Its raw placebo shuffled only temporal history while retaining
current cross-state, its JEPA arm included raw values, and nominal 6h accepted
actual 2-10h labels. It did not isolate AKI-onset alignment or rule out encoder
improvement. Do not repeat the earlier claim that all extra information is merely
population correction or absent.

The corrected diagnostic separates current-state, temporal, full-cross-block,
pure latent and release-bundle readouts, with primary actual 5-7h labels. Pure
latent beats its swapped-block placebo in pooled eICU patient/hospital and MIMIC
comparisons, but does not establish improvement over own history in eICU and
loses to persistence in MIMIC. Raw temporal features also do not stably improve
over current state: they worsen eICU patient MAE and fail the MIMIC temporal
placebo comparison. Care-unit gains remain heterogeneous. This is partial
readout evidence, not a promoted cross-organ mechanism; latent includes own
target history, so cross-organ attribution still requires own-target ablation.
See `ANCHOR_ALIGNED_SIGNAL_CORRECTED_FINDINGS_20260908.md`.

The follow-up measurement-anchor diagnostic used fixed patient support and
actual creatinine measurement anchors, then compared equal-capacity target-only
and whole-body frozen-JEPA heads. Fresh anchors reduced error for every method,
including persistence, so task freshness cannot be credited to the encoder.
Target-only JEPA had weak eICU benefit but was significantly worse than
persistence in MIMIC. Adding non-target state was worse than target-only in all
seven eICU fresh patient resamples and only directionally, not significantly,
better in MIMIC, where it still lost to persistence. The first isolated failure
is therefore conditional cross-database instability of the non-target latent
correction, not stale timing alone. See
`MEASUREMENT_ANCHOR_ENCODER_FINDINGS_20260909.md`.

Stale anchors, target-only state and non-target state have now been separated on
fixed patient support. Another readout sweep on the same frozen latent is not
the next experiment. A new encoder trial would need event-time observation
semantics and cross-database invariance trained end to end, a prespecified
source-side benefit, and a new untouched lockbox. Prior negative portfolio and
catalog readout results remain valid within their tested protocols; they do not
prove a universal information ceiling. High-frequency pre-event signals and
richer treatment intent remain distinct, unproven source contracts.

Exact physical store supervision remains unavailable because the current eICU
ledger has urine output but no complete unit-normalized intake/net-balance
flux. This gap should not be disguised as a model-capacity experiment.

For event prediction, evaluate point forecast, prospective change risk, and
interval calibration as separate capabilities. A success in one is not a
promotion of the others.

A prequential continual-learning screen has also been completed for exact
`creatinine@6h` measurement anchors. Naive online updates forgot prior domains;
short/long and uniform replay reduced that harm and improved the frozen adapter
on all seven patient orders. Neither replay method beat persistence, and both
still caused significant MIMIC harm under some arrival orders. This establishes
a source-side adaptation signal and the value of replay, not a deployable
self-learning model. See `CONTINUAL_SELF_LEARNING_FINDINGS_20260909.md`.

A stronger prequential test then used the already validated `glucose@6h`
release source. The frozen source beat persistence on the complete strict eICU
and MIMIC co-measurement slices. On repeat eICU patient predictions, a frozen
model plus a patient-scoped running residual state improved the release source
by 2.27 mg/dL in all seven arrival orders without harming unrelated patients.
It did not significantly beat persistence on that smaller repeat-only slice,
so it was not promoted. Naive, short/long replay and uniform replay updates to
the global 311-feature adapter all worsened the stream and caused significant
retention, hospital and MIMIC harm in all seven orders. Corrections were too
large rather than merely clipped. Current evidence supports low-dimensional
patient-state updates with frozen population weights, not global online
retraining. See `CONTINUAL_VALIDATED_CELL_FINDINGS_20260909.md`.

That patient-state hypothesis has now been replayed independently inside every
eligible MIMIC patient with a stricter primary definition: at least one earlier
forecast outcome had to be resolved before a prediction counted as
personalized. A time-aware residual level/trend selected on eICU did not
transfer. It was significantly worse than the simpler patient mean on eICU
hospital-heldout and significantly worse than both the frozen release and the
simple state in MIMIC. Its main reproducible behavior was lower false movement
on stable patients, accompanied by worse direction accuracy and MAE when
glucose truly changed. Patient-conditioned conformal scaling narrowed the
MIMIC interval by 25.75 mg/dL versus the frozen interval, but row/patient
coverage of 0.933/0.934 narrowly exceeded the allowed upper bound and point
forecast failed. Personal error history currently supports uncertainty
conditioning more strongly than transferable trajectory correction. See
`PATIENT_STATE_EXTERNAL_FINDINGS_20260909.md`.

A corrected failure-localization audit supersedes that experiment's placebo
interpretation. The original report mixed full-role and personalized-only donor
supports; under one causal relative-time placebo contract, no patient-state
method passed all seven external placebos. Component ablation showed that the
plain patient residual level already loses transfer significance, while
exponential decay adds significant hospital and MIMIC harm; slope recovers only
part of that loss. With the point source frozen, personal scales narrowed
intervals but were less consistently calibrated than a constant scale after
MIMIC patient-disjoint recalibration. Raw cross-variable history and frozen
JEPA latent both contain weak patient information, but neither stably improves
over target history across eICU hospital-heldout and MIMIC. The next repair is
not another unconditional trajectory extrapolator: it is an anchor-known gate
for whether a previously estimated personal bias remains valid, evaluated
separately from fixed-point uncertainty calibration. See
`PATIENT_STATE_DIAGNOSTIC_FINDINGS_20260909.md`.

The prespecified follow-up kept the same frozen `glucose@6h` JEPA source and
compared no personal update, unconditional resolved-error state, and an
anchor-known event-aware gate. The gate received current measurement values,
observed masks, ages, changes since the previous anchor, and harmonized
pre-anchor insulin, dextrose, fluid, steroid and vasopressor history. Across
seven source-defined fits it significantly beat the frozen source in only 1/7
eICU hospital comparisons and 0/7 MIMIC comparisons. It never beat a
state-only equal-capacity gate and did not beat shuffled event context. On
eICU treatment-transition rows its direction accuracy improved, but that
pattern did not transfer to MIMIC. Wrong-patient state remained harmful,
showing patient identity is informative; the current event representation
does not identify when the old signed correction remains valid. No runtime
point, interval or selector changed. See
`EVENT_AWARE_PATIENT_STATE_FINDINGS_20260909.md`.

An anchor-summary recurrent correction loop has now been tested. A
patient-scoped recurrent state consumed anchor-known measurement semantics,
within-patient changes, harmonized pre-anchor treatment and only already
resolved forecast errors, then directly decoded a signed residual. It beat
persistence in all seven eICU hospital comparisons but none in MIMIC, and it
beat the frozen JEPA source in 0/7 comparisons in both roles. Its small eICU
gain on genuinely changing rows came with stable-row harm; both effects became
harm in MIMIC. Reset-state and shuffled-event recurrent controls remained
competitive, and treatment-transition direction did not improve. The present
summary contract did not identify a transferable signed state update.
No runtime point, interval or selector
changed. See `RECURRENT_EVENT_STATE_LOOP_FINDINGS_20260909.md`.

The 2026-09-10 CodeGraph audit corrects the earlier claim of complete event-loop
validation. Update streams depend on future-label eligibility; input centering
uses the evaluation cohort; the new-measurement flag misses repeated observed
measurements; presence/count treatment encoding loses rate-only changes.
Three synthetic input counterexamples were reproduced. Feedback is frozen-model
error, not candidate-output error. Their contribution to MAE remains unmeasured.
Repair the event contract before another architecture sweep; see
`EVENT_STATE_CONTRACT_DIAGNOSIS_20260910.md`. Runtime is unchanged.

The repair has now been implemented and tested: source-frozen centering,
unclipped float64 measurement age, repeat-event detection, unit-safe treatment
snapshots with known masks/provenance, label-independent cached-anchor updates,
resolved candidate-error feedback, and zero-checkpoint selection. Two final
seven-seed runs still beat frozen JEPA in 0/7 comparisons in both eICU hospital
and MIMIC. MIMIC MAE is 30.985 for full cached-anchor recurrent state and 30.940
for own-error feedback, versus 28.944 frozen; scored-only repaired replay is
30.239/30.190. Software invariants are repaired, but transferable signed updates
are not established. Full raw-event/available-time replay remains unfinished;
the cache is not every bedside event. No runtime change. Final evidence is
`EVENT_STATE_REPAIR_FINDINGS_20260910.md`; intermediate repair runs are not the
current code's validation.

The fixed-weight update-invariance diagnostic is complete. Re-consuming the
same encoded event without new truth changed later forecasts; 4-copy MIMIC mean
absolute shift was 1.520 mg/dL, and 5/7 seeds had significant MAE harm. This
establishes non-idempotent ingestion sensitivity, not actual duplicate frequency
or the sole cause of the transfer gap. In contrast, withholding just-arriving
error feedback slightly worsened MIMIC MAE (30.940 ->30.963), with no significant
benefit in any seed. Newly arriving old errors are not supported as the principal
immediate cause; accumulated hidden-state effects remain possible. Next verify
raw event identities/availability and separate re-documentation from true new
measurements. Do not repeat a blanket error-removal experiment as an untested
solution. See `UPDATE_INVARIANCE_FINDINGS_20260910.md`. Runtime unchanged.

The raw-event follow-up found an earlier availability-contract failure in the
glucose@6h cache. Every old scored current/future value matched native lab
events, but current results were entered AFTER the old anchor in644/846 eICU
development,614/797 hospital, and2349/2349 MIMIC rows (median delay49/47/64min).
eICU uses conservative final-revision entry time; MIMIC uses storetime. Exact
source-version dedup removed zero inspected role rows; distinct equal-valued
remeasurements were retained. The original-anchor accuracy trial correctly did
not start because MIMIC has zero strict pairs. Raw-event/query separation is
implemented and structurally tested, not trained or promoted. Next rebuild
availability-time anchors AND all input/label times, not just deduplicate or
shift old labels. Old retrospective scores on this slice do not establish
availability-safe online performance; this is not a blanket rejection of all
JEPA evidence. See `RAW_LAB_IDENTITY_FINDINGS_20260910.md`. Runtime unchanged.

## Data Roles

| Data | Current role |
|---|---|
| eICU | Primary development, selection, and internal hospital-heldout evidence |
| MIMIC-IV | Repeated external/validation evidence; no longer untouched |
| NWICU | Repeated external/validation evidence; no longer untouched |
| MIMIC-IV Waveform / VitalDB | Waveform research; no promoted forecast cells |
| NHANES | Cross-sectional nowcast evidence, not longitudinal forecasting |
| HiRID / AmsterdamUMCdb | Candidate future untouched lockbox if imported and frozen before use |

## Safety And Release Boundary

- Raw row-level clinical data stays local and out of Git.
- A physical zero is an observed value, not missingness.
- Missing values remain missing unless a separately validated nowcast exists.
- Unsupported cells use persistence or abstain; they do not emit invented
  precision.
- Candidate artifacts do not change release weights, registry, intervals, or
  serving behavior unless the complete promotion gate passes.

See `EXPERIMENT_INDEX.md` for completed experiments and negative results.
# Availability-Anchored Glucose Update (2026-09-10)

See `AVAILABILITY_ANCHOR_FINDINGS_20260910.md`. Rebuilt glucose queries at raw
result availability and rebuilt both history and 5-7h future labels; old feature
vectors/predictions were discarded. MIMIC support restored to6,616 pairs/889
patients. Seven source fit/tune seeds: history ridge beats persistence7/7 on
held-out eICU hospitals and MIMIC. Median MAE38.874 vs45.878 and34.900 vs37.449.
Increment beyond current-only ridge is significant7/7 eICU but3/7 MIMIC.
This is a glucose-only non-neural diagnostic, not full JEPA or interval promotion;
runtime unchanged. Full multi-variable availability-safe encoder inputs remain
unfinished; old and new task MAEs cannot be compared as a paired repair benefit.

## Availability-Safe JEPA Input Follow-up

`AVAILABILITY_JEPA_FINDINGS_20260910.md`: full53-slot JEPA schema now rebuilt
from42 eICU/40 MIMIC time/unit-qualified laboratory variables. Bedside/treatment
channels without verified availability remain missing, not fully reconstructed.
Same new glucose anchors: frozen full-lab truthful-mask head MAE41.611/34.926
beats persistence45.878/37.449; whole vs own incremental benefit fails MIMIC.
Source-refitted311-feature full readout harms transfer (median43.891/56.979),
while own-only is37.181/33.467. Full readout wins0/7 vs own or frozen in both
domains. Truthful mask change under frozen weights also worsens old encoding;
this is an input-distribution compatibility concern, not permission to fake masks.
Feature-scale/support shifts observed in pH, LDH, bilirubin, ionized calcium;
single root cause not established. V2 repaired unit aliases; v1 preserved.
289 tests pass. No end-to-end retraining, calibration, promotion or runtime change.

## Availability Encoder Training Pilot

The previously untested encoder step has now been tried, not merely proposed.
See `AVAILABILITY_ENCODER_FINDINGS_20260910.md`. Three seeds x own/full inputs
x frozen/trainable context, same availability-safe glucose labels. Real EMA
target-delta JEPA plus direct value loss; six end-to-end checkpoints each change
77 context tensors. Whole end-to-end improves whole frozen significantly only
1/3 in either domain, beats own0/3, and beats persistence3/3 hospital but2/3
MIMIC. Median whole MAE42.386/36.091, own42.112/34.536. This is direct neural
value decoding, not the earlier311-feature head. Warm-start8-epoch pilot only;
no general conclusion that encoder adaptation is impossible. No calibration,
seven-seed promotion, other-target retention or runtime change.

## Direct Decoder Diagnosis

`ENCODER_BOTTLENECK_FINDINGS_20260910.md` isolates a proven amplitude limit:
direct glucose residual cannot exceed30.023mg/dL;35%-46% of validation labels
exceed that range. This is not the external shared readout and not a universal
release diagnosis. Whole MIMIC predictions also saturate more while correlating
less well than own, so removing the cap alone may amplify errors. Identical
whole weights with own-only inputs help only1/3 MIMIC seeds. JEPA gradients are
active but relatively small in sampled source batches. These separate a hard
decoder constraint from unresolved context generalization; do not call either
the sole cause of all past failures. No runtime or promotion changed.

## Controlled Amplitude Repair

`ENCODER_REPAIR_FINDINGS_20260910.md`:24 matched fits completed. Replacing tanh
with identity in candidates and retraining own-history JEPA improves glucose
MAE42.112->38.273 hospital and34.536->32.070 MIMIC, paired improvement3/3 seeds
in each. Whole unbounded improves hospital3/3 but MIMIC1/3. Gradient balancing
helps whole MIMIC modestly; combined MAE38.485/34.105, still loses to own.
All modes fail to establish whole-vs-own benefit. The amplitude cap is an
experimentally supported contributor, not the sole transfer cause. Serving
retains tanh and all prior weights; candidates require activation metadata.
No seven-seed/interval/retention promotion.293 tests pass. Goal remains open.

## Source-Support Contract Repair

`ENCODER_SOURCE_SUPPORT_FINDINGS_20260910.md`: source-only support masking with
fixed repaired weights isolates ionized_calcium. Source fits observe zero such
events after mg/dL rejection; MIMIC mmol/L values enter an inherited median3.8,
scale0.178 scaler (median z about-15). The source-seen intervention changes
only that MIMIC channel and no eICU channels. MIMIC MAE35.461->33.323 unbounded,
34.105->33.324 balanced, significant3/3 each. Identical eICU outputs.
Candidate checkpoints now carry source support/activation metadata. No guessed
unit conversion: raw system/interface units conflict in some records. This is a
localized input-contract contributor, not proof of old scaler provenance or
complete repair; whole remains worse than own32.070. Runtime unchanged.

## Source Fit Gap (Latest)

`ENCODER_FIT_GAP_FINDINGS_20260910.md`:14 selected original32-epoch identity
checkpoints reproduce source-tune scores. Whole beats own significantly6/7
on fitting patients but loses numerically7/7 (significantly6/7) on source-tune
patients. Each split is only250 fit/136tune patients. This directly locates
most extra-context failure before external transfer: in-sample advantage does
not generalize even within eICU. Do not call MIMIC harmonization the only
remaining problem. The subsequent source expansion is recorded below. Small N
alone is not established as the sole cause.

## Source Expansion (Latest Completed)

`ENCODER_SOURCE1000_FINDINGS_20260910.md`:28 matched32-epoch fits completed
with1,000fit/136unchangedtune patients per seed, versus250-fit history-mask
reference. Whole identity MAE hospital38.070->37.116, MIMIC33.497->32.286;
paired improvement7/7 and6/7. Own36.617/31.211 remains better; whole-vs-own0/7
both. Whole source-tune remains worse7/7. Source breadth/support helps but does
not solve incremental whole-context generalization. Count/composition and
fixed-compute exposure are confounded; no sole-N or convergence claim.
The2,250-patient stage is now complete (see below). No runtime/calibration/promotion changes;
302 tests pass. Raw patient data stays private.

`ENCODER_SOURCE2250_FINDINGS_20260910.md`:28 additional matched fits completed.
Whole identity MAE37.108hospital/32.223MIMIC, versus37.116/32.286 at1000fit;
own36.998/31.214. Whole-vs-own remains0/7both, source-tune worse7/7. Whole
improves250-fit reference6/7each, but no full repair. TSICU0/7both persists.
This fixedcompute curve does not establish convergence or sole-N cause;
expanded-cohort optimizer exposure remains untested. No runtime/calibration
or promotion change. The source-only budget diagnosis and follow-up are now
complete, as recorded below.

## Expanded-Cohort Budget (Latest)

`ENCODER_EXPANDED_BUDGET_FINDINGS_20260910.md`: exact source sampler replay
shows2250fit supervised-row coverage21.60%, future-event coverage24.07%,
median7 draws/patient at32epochs. Late whole source-tune improves7/7, so a
fixed64epoch experiment was run.28fits complete; first32logs identical.
Whole hospital37.108->36.796,MIMIC32.223->32.037; significant improvement2/7
both. Own36.310/31.209; whole-vs-own0/7both, source-tune worse6/7. Additional
optimization helps partly, not complete repair; no convergence/solecause
claim. No runtime/calibration/promotion change.303tests pass.

## Requested Time Diagnostic

`ENCODER_QUERY_TIME_FINDINGS_20260910.md`: builder nominal6h versus actual
5-7h label tolerance confirmed. Added explicit query horizon argument without
changing default inputs. Four-time fixed-weight14checkpoint source-tune
diagnostic reproduces nominal scores; actual-time query improves0/7own and
0/7whole. MAE changes only~0.005/~0.012mg/dL. No inference-only repair; actual
elapsed-time training remains untested. Retrospective measurement-time query
is not prospective fixed6h forecasting. No runtime/promotion;303tests.

The subsequent actual-time training is now complete:
`ENCODER_ACTUAL_TIME_FINDINGS_20260910.md`.28matched2250/64fits, actual5-7h
source supervision/tune selection; primary validation remains fixed6h query.
Whole36.799hospital/32.067MIMIC versus36.796/32.037nominal-trained, paired
improvement0/7both, harm1/7hospital5/7MIMIC. Whole-vs-own0/7both. Actual-time
training within this narrow tolerance does not repair context generalization.
Separate retrospective actual-query scores descriptive only. No runtime,
calibration or promotion;303tests. Do not repeat as an untested timing repair.

## Empty History Mask

`ENCODER_HISTORY_MASK_FINDINGS_20260910.md`: located omitted step-mask wiring
at the belief filter call. Synthetic leading-empty state change3.75717 becomes
exactly0 with its existing mask.28 matched32epoch fits completed; whole identity
hospital38.202->38.070,MIMIC33.368->33.497. Whole-vs-own remains0/7. Own identity
has significant harm3/7hospital,4/7MIMIC. Therefore the invariant correction
does not establish forecast repair or sole root cause. No runtime change;
300 tests and release validator41/27/14 pass. Candidate requires explicit
observed_steps metadata. Repeated real events are not deduplicated by value.

## Context Dropout

`ENCODER_CONTEXT_DROPOUT_FINDINGS_20260910.md`:28 matched32-epoch fits with
training-only non-target history block dropout. All own-control logs and
predictions exact. Whole identity hospital38.202->37.479,MIMIC33.368->33.128;
paired improvements3/7 both, MIMIC harm1/7. Whole-vs-own remains0/7. Thus this
fixed augmentation is not a stable solution, despite modest median gains.
TSICU and2014-2016 remain0/7 vs persistence.299 tests; no runtime, calibration
or promotion changed. Do not repeat this as an untested dropout hypothesis.

## Source Numerical Rescale

Teacher follow-up: `ENCODER_TEACHER_DIAGNOSTIC_20260910.md` tests14 budget
checkpoints on a fixed glucose delta grid. All teachers distinguish direction
and amplitude. Literal constant future-target encoding is not supported;
this does not rule out weak supervision on real data. Remaining context
transfer failure has no experimentally established single root cause.

`ENCODER_RESCALE_FINDINGS_20260910.md`:28 matched32-epoch fits with source-only
non-target median/std normalization; all own-control logs/predictions exactly
reproduce. Whole identity hospital38.202->37.708, MIMIC33.368->33.572;
paired improve3/7 both, MIMIC harm1/7. Whole-vs-own remains0/7. Simple inherited
affine scale mismatch alone does not repair/explain the remaining transfer gap.
MIMIC TSICU0/7,CCU1/7 vs persistence remain. No new runtime/calibration/promotion;
298 tests. Candidate weights require numeric transform metadata; do not change
release input scaling without compatible weights and calibrators.

## Numeric Versus Age Localization

`ENCODER_COMPONENT_FINDINGS_20260910.md`: fixed32-epoch whole weights, value
neutralization retaining mask/age/presence improves MIMIC7/7 (MAE33.368->32.437),
but not hospital (38.202->38.485, no significant change). Exact-age neutralization
has tiny effect. Prior heme block harm does NOT prove heme numbers help: heme
value neutralization improves MIMIC4/7 while block removal harmed5/7. Numeric
response and observation support interact; no sole cause or workflow-only claim.
Do not deploy artificial centers. Own trained MAE31.238 remains better. No
weights/runtime/promotion changes;297 tests. Source-trained numeric contract
repair remains to be tested without external feature selection.

## Context Group Localization

`ENCODER_GROUP_FINDINGS_20260910.md`: seven32-epoch whole identity weights,
12 input-group removals, fixed predictions reproduced. Acid-base removal
harms4/7 hospital but helps3/7 MIMIC; blood/coagulation removal harms5/7 both.
No group deletion helps7/7 both. All-non-target removal also fails. This
localizes mixed group/domain dependencies, not a causal organ or sole cause.
Masking changes support/age too; value-vs-process dependence is unresolved.
No runtime/calibration/promotion. Lab-only respiratory group is a no-op, not
proof respiratory biology has no predictive value.

## Extended Training Budget

`ENCODER_BUDGET_FINDINGS_20260910.md`:28 fits extended8->32 epochs, all first8
logs exactly reproduced. Own identity improves hospital7/7,MIMIC5/7 versus8;
whole identity improves1/7 each and significantly harms MIMIC4/7. MAE own
36.559/31.238,whole38.202/33.368. Whole-vs-own remains0/7. Thus short training
alone does not explain/repair the full-context gap. No convergence claim:
own selected epochs remain29-32. MIMIC TSICU both0/7 vs persistence; whole
other care-unit/time heterogeneity persists. No runtime/calibration/promotion;
296 tests pass. Source-tune gains do not establish transferable physiology.

## Seven-Seed Decoder Confirmation

`ENCODER_CONFIRM_FINDINGS_20260910.md`:28 matched source-supported own/whole x
bounded/identity fits complete, corrected BNP inputs. Both identity arms
improve bounded7/7 and persistence7/7 in pooled hospital and MIMIC. Own median
MAE38.091/31.812, whole38.331/33.087. Whole-vs-own incremental gain remains0/7.
MIMIC TSICU own vs persistence1/7, whole0/7; whole CCU1/7 and2014-2016 period1/7.
These are descriptive subgroup audits, not forward-time holdouts. Cap repair
is seven-seed supported but insufficient for deployment. No calibration,
retention, new lockbox, or runtime change;296 tests pass.

## Analyte Identity Repair

`BNP_IDENTITY_FINDINGS_20260910.md`: MIMIC50963 was incorrectly mapped to BNP;
local dictionary verifies NTproBNP. Renamed to nt_pro_bnp, no alias/conversion.
Standard availability cache corrected; original archived for prior reports.
Fixed supported candidates show no significant glucose gain (0/3 both modes,
median33.323->33.337 and33.324->33.339). Confirmed semantic defect, not principal
residual forecast cause. Units/method shifts must not be "fixed" by guessing
from medians.296 tests pass; release weights and registry unchanged.
