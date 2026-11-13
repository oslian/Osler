# WholeBody-JEPA Experiment Index

2026-10-05 E44 COMPLETE/VERIFIED, fixed diagnostic ablation:
RECONSTRUCTION_TANGENT_SPECTRUM_PROTOCOL_20261005.md;
RECONSTRUCTION_TANGENT_SPECTRUM_NUMERICS_20261005.md;
RECONSTRUCTION_TANGENT_SPECTRUM_FINDINGS_20261005.md;
scripts/audit_tangent_spectrum.py; scripts/verify_tangent_spectrum.py;
tests/test_tangent_spectrum.py; tests/test_tangent_spectrum_verification.py.
Fixed-weight strong/weak Jacobian-direction decomposition of saved E43 correction;
original lambda and accepted step, no refit or query-selected threshold.
artifacts/native_tangent_spectrum_20261005.json and
artifacts/native_tangent_spectrum_verification_20261005.json are complete.
Strong-only fit/selection/same-source MAE .746151/.892978/.787198 versus zero
.753682/.876270/.781413. Removing weak directions is not a sufficient repair.
Fivefolds/20480queries independently replay0,scalar3.553e-15;39/30bindings,
34tests per saved findings. Direct-SVD numerical repair and failed sources retained.
No new formal efficacy gate, forecast, self-learning or promotion; fold0/2 repeated,
development data reused. Close fixed branch, no cutoff search. Runtime unchanged.
Supersedes earlier pending-E44 notes; documentation sync does not rerun research.

2026-10-05 E43 COMPLETE/VERIFIED: full-fit local solve plus nonlinear transfer.
RECONSTRUCTION_TANGENT_BRIDGE_PROTOCOL_20261005.md;
RECONSTRUCTION_TANGENT_NUMERICS_20261005.md;
scripts/fit_tangent_bridge.py; scripts/verify_tangent_bridge.py.
Full fit prediction Jacobian for the single frozen-backbone bridge, followed by
fit-only finite-step selection and actual recurrent replay. Distinct from E36's
derivative along an existing direction and E42's fixed-readout rank audit.
RECONSTRUCTION_TANGENT_BRIDGE_FINDINGS_20261005.md;
artifacts/native_tangent_bridge_20261005.json / native_tangent_bridge_verification_20261005.json.
All5folds,2048 retained fit directions,half-step accepted; actual MAE fit .745849,
selection .893038,same-source .784977 versus zero .753682/.876270/.781413.
Linear diagnostic .407425/1.332393/1.163262; all11screens/full0. Query mean
reversals23/33 matching fit improvements in each role. Saved5models/20480queries
replay0; scalar/paired6.661e-15;45/33bindings;23tests. Initial failed solver and
numerical amendment disclosed. Close fixed branch; no promotion/runtime change.
Not a pure optimizer comparison, future forecast, or new external lockbox.

2026-10-05 E42 COMPLETE: fixed-readout algebraic reachability audit.
RECONSTRUCTION_READOUT_REACHABILITY_PROTOCOL_20261005.md / FINDINGS;
scripts/audit_readout_reachability.py; tests/test_readout_reachability.py;
artifacts/native_readout_reachability_20261005.json. Saved5folds,15bindings,
7unittests, NumPy/torch rank and pseudoinverse agreement. Six zero rows are
unscored; active47/scored46/supported27 subsets full-row-rank. Supported
float32 unit-output replay max3.5763e-7. No hard affine readout dependency
explaining27supported failures; recurrent reachability/learning remains open.
Retrospective inspection disclosed, no training/runtime/promotion or new lockbox.

2026-10-05 E41 COMPLETE: E40 full-inventory screen failure decomposition.
RECONSTRUCTION_SCREEN_DIAGNOSIS_PROTOCOL_20261005.md / FINDINGS;
scripts/diagnose_screen_failures.py; tests/test_screen_failures.py;
artifacts/native_screen_failures_20261005.json. All53targets,5120selectionrows,
460comparisons,81bindings,11tests; independent scalar mean/p maxdiff3.916e-13.
7unsupported/19under-supported/27supported mean-conjunction failures; no target
blocked ONLY by significance across all ten controls. Raw23mean-fail/4CI-fail.
No new test selection, changed alpha, training, runtime or promotion. Not an
external replication or solution of representation/conditional-transfer gaps.

2026-10-05 E40 COMPLETE/VERIFIED: actual Adam step projected after PCGrad.
RECONSTRUCTION_ACTUAL_STEP_PROTOCOL_20261005.md / FINDINGS;
scripts/train_actual_step.py; scripts/verify_actual_step.py;
tests/test_actual_step.py; tests/test_actual_step_verification.py;
artifacts/native_actual_step_20261005.json / native_actual_step_verification_20261005.json.
5models/20480queries,explicitreplay0,scalar/paired3.918e-13,58/69bindings,28tests.
Cross-targetraw1814localharm->actual0,441roundingveto;finite489tinyincreases.
Fit/selection/same MAE .749089/.876494/.777282;ten comparisons/fullscreen0.
32/47fit targets fail to improve over PCGrad. Fixed branch closed, not a full
repair, no sweep/runtime/promotion. Fold0/2identical, not independentreplicates.

2026-10-05 E39 COMPLETE/VERIFIED: same-step Adam stage accounting, no repair.
RECONSTRUCTION_ADAM_STAGES_PROTOCOL_20261005.md; findings in ROOT_CAUSE_LEDGER.md section6.
scripts/diagnose_adam_stages.py; scripts/verify_adam_stages.py;
tests/test_adam_stages.py; tests/test_adam_stages_verification.py;
artifacts/native_adam_stages_20261005.json / native_adam_stages_verification_20261005.json.
Ten E38 original tensors/logs exact;40960 training-example steps. PCGrad1921
actual locally harmful: first proposal43,momentum1759,preconditioning119.
Finite loss/MAE exact reconstruction; scalar roundoff2.711e-20. Local conditional
accounting, not unseen-patient efficacy or optimizer-removal causality.
No runtime/weights/interval/promotion. Full repair unresolved; do not rerun as untested.

2026-10-05 E38 COMPLETE/VERIFIED: paired mean versus PCGrad, one shared bridge.
RECONSTRUCTION_PAIR_GRADIENT_PROTOCOL_20261005.md / FINDINGS;
scripts/train_pair_gradient.py, scripts/verify_pair_gradient.py;
tests/test_pair_gradient.py, tests/test_pair_gradient_verification.py;
artifacts/native_pair_gradient_20261005.json / native_pair_gradient_verification_20261005.json.
10models/20480queries, replay0, scalar/statistics1.833e-13;24tests,69/74bindings.
Proposed local harmful counts670->67 but actual Adam2197->1921; selection
.876553->.876528, same-source .777108->.777131; nine comparisons/full0screen.
Fixed budget/initialization/support, no runtime/promotion. Fold0/2 identical.
Branch closed: local gradient surgery insufficient, no parameter sweep.

Historical entry (superseded by completed E38 evidence above):
2026-10-05 E38 PREPARED ONLY, not a completed experiment:
RECONSTRUCTION_PAIR_GRADIENT_PROTOCOL_20261005.md;
scripts/train_pair_gradient.py; tests/test_pair_gradient.py.
Matched paired mean versus PCGrad on the fixed shared bridge. No completed
test/training/assessment/verification evidence yet; no promotion or runtime change.
The latest documentation request does not authorize running this pending plan.

2026-10-05 E37 COMPLETE: original E34 immediate Adam momentum-origin replay.
RECONSTRUCTION_ADAM_ORIGIN_PROTOCOL_20261005.md / FINDINGS; scripts/diagnose_adam_origins.py;
tests/test_adam_origins.py; artifacts/native_adam_origins_20261005.json.
5fold original tensors/logs exact,20480queries,53origin inventory+decay+rounding,
52bindings,15roles preserved;25tests. Vector/tangent max1.111e-16,zero parity0.
Selection18 local harmful totals=8 own helpful/other harmful+10 own harmful;
fit7=7+0,same-source11=9+2. Independent scalar total check6.939e-18.
Joint denominator/path conditional; not deletion causality or finite-step
repair. Fold0/2 vectors identical,not independentreplicates. Existing small
gradient cosine/accumulation experiments not repeated. No runtime/promotion.

2026-10-05 E36 COMPLETE: differentiate saved E34 immediate projection ray atzero.
scripts/diagnose_parameter_path.py; tests/test_parameter_path.py; protocol/findings
RECONSTRUCTION_PARAMETER_PATH_*_20261005.md; artifacts/native_parameter_path_20261005.json.
All5fold/3roles/20480queries/53inventory, nofit/scale choice. Float32 endpoint parity0;
float64 maxdrift1.264e-6;670prespecified two-step FD checks max1.036e-8, scalar/fsum
max1.777e-15.21tests,42bindings, originalcontrol rows/identities/support preserved.
Selection22MAEmean harms:18initial parameter direction harmful,4beneficial;
BUN/temperature initial/chord disagree. Same15:11/4; componentselection25:20/5.
Parameter path nonconvex: local harmful derivative does not forbid allfinite scales.
Notoptimizer training trajectory, significance, future, causal orpromotion.
Time dtype bug in NEWdiagnostic fixed before successful fullreplay; original models
unchanged. Rootcause of conditional direction transfer unresolved, no runtime change.

2026-10-05 E35 COMPLETE: E34 output-direction/curvature and objective audit.
scripts/diagnose_bridge_geometry.py; tests/test_bridge_geometry.py;
RECONSTRUCTION_BRIDGE_GEOMETRY_PROTOCOL_20261005.md and FINDINGS;
artifacts/native_bridge_geometry_20261005.json. Fixed53/5fold/3channels/3roles,
20480queries/61440outputs. All rows scalar/fsum match,max3.553e-15;20tests,
30bindings,original MAEs/support reproduced. Selection immediate22mean harms:
20positive MAE right derivatives,2crossing-driven;4SmoothL1 gains/MAE harms.
17/18prior reversals have positive derivatives; same-source11/15mean harms.
Descriptive,not significance or physiological direction. Macro right derivative
negative despite20per-target positives; cannot claim no aggregate shrinkage
opportunity. Limits fixed-output positive scaling,not recurrent parameter path.
No oracle scaling,routing,new fitting/promotion or external validation.
Prior teacher geometry/E9 and exact-MAE readout20260920 are not untried repairs.

2026-10-05 E34 COMPLETE/VERIFIED (supersedes prepared status below).
Only bridge.projection.weight trains from identical E29 epoch10 start;
5folds/3channels/3roles,15models/20480queries,28,672trainable parameters per arm.
No main-model updates, all saved explicit-replay parity0; controls unchanged.
Independent scalar MAE/paired bootstrap/screens verified,maxdiff8.882e-15;
20tests. Seven comparisons and full screen0. Relative matched bound_start,
39/47fit means improve,18selection/12same-source reversals; coadaptation of
backbone is not necessary, but condition/support/budget responsibilities remain.
Same-source macro improvement is descriptive, not promotion or fresh external.
Close fixed branch; no runtime/interval change. Protocol/findings:
RECONSTRUCTION_FROZEN_BRIDGE_*_20261005.md; scripts/train_frozen_bridge.py,
scripts/verify_frozen_bridge.py; artifacts/native_frozen_bridge_assessment_20261005.json,
artifacts/native_frozen_bridge_verification_20261005.json and five fold reports.

Historical entry (superseded by the completed E34 evidence above):
2026-10-05 E34 PREPARED, NOT VALIDATED: fixed-backbone bridge versus E29 joint
adaptation at the same start. RECONSTRUCTION_FROZEN_BRIDGE_PROTOCOL_20261005.md,
scripts/train_frozen_bridge.py and tests/test_frozen_bridge.py exist.
Tests/training/scoring/independent verification are not completed; no findings
or promotion. Latest request is documentation only, not execution. E33 below
remains the latest completed evidence; consult ROOT_CAUSE_LEDGER.md first.

2026-10-05 E33 COMPLETE: fixed trained bridge on/off, no fit. All5folds/3channels/
3roles,20480queries. Only projection zeroed in copy; originals unchanged.
All saved on and off/native parity0; independent aggregate verification
maxroundoff3.553e-15,17tests. Immediate bridge mean-harm26/46selection and
26/44same-source;25/24 fit-helpful then harmful. Selection off macro .875705
vs on .876522/unbridged .875949, still above raw .853856. Hgb/renal counterexamples;
not universal removal or promotion. Training coadaptation vs conditional
instability still unresolved. Protocol/findings RECONSTRUCTION_BRIDGE_ABLATION_*;
scripts/audit_trained_bridge.py,scripts/verify_trained_bridge.py;
artifacts/native_trained_bridge_ablation_20261005.json and verification.
Do not repeat trained on/off as untested. No runtime/interval/weight change.

2026-10-05 E32 COMPLETE: fixed E29fit/E31third-role common-source midpoint
decomposition,53targets/5folds/4controls; no new predictions or training.
Percontrol206decomposed,49missing-role,10no-common-source. Primary170within>0,
185abs(within)>abs(composition); descriptive erosion, not absolute failure.
All206include a component-role under30; no exclusions/gate. Existing group-shift
helper reused; independentSQL824checks maxdiff1.388e-17;31bindings,13tests.
Protocol/findings RECONSTRUCTION_ASSIMILATION_COMPOSITION_*_20261005.md;
artifact native_assimilation_composition_20261005.json;
scripts/diagnose_assimilation_composition.py,tests/test_assimilation_composition.py.
No runtime or promotion. This is new role evidence, not a new decomposition
algorithm or complete root-cause identification.

2026-10-05 E31 VERIFIED (supersedes pending below). Independent selection,
native/explicit replay for5120queries across5folds and5arms:prediction error0;
summary/support/means verified,max roundoff8.882e-16.16tests pass. Original
22selection reversals split13same-source non-gains/8mean gains/1unsupported.
1613people,44supported;5105query occurrences exposed in epoch10 training.
No new gate or promotion; smallsupport/mixture remain. Findings:
RECONSTRUCTION_ASSIMILATION_TRANSFER_FINDINGS_20261005.md; verification:
artifacts/native_assimilation_transfer_verification_20261005.json.
Do not repeat third-role inference or equate descriptive mean with root cause.

2026-10-05 E31 RAW INFERENCE COMPLETE, verification pending. Existing session
29146 exited0; artifacts/native_assimilation_transfer_20261005.json saved.
Same-source adaptation-unseen role, NOT necessarily encoder-unseen patients.
scripts/verify_assimilation_transfer.py is written but not executed/validated.
Latest request is documentation only; no further execution or promotion.
E30's unexecuted third-role statement below is historical, not current status.

2026-10-05 E30 COMPLETE: E29 same-model role identifiability, no fitting.
Fivefold source/patient intersections0;219target-folds both roles without
common source,46missing role support. Pooled5common sources are NOT same-model
overlap.23bindings/independentSQLchecked,9tests;no runtime/promotion.
Prior E13/conditional-offset diagnostics do not supply missing E29 same-source
unseen-patient control. That third role remains unexecuted; original encoder
exposure must be audited. Protocol/findings:
RECONSTRUCTION_ASSIMILATION_ROLES_{PROTOCOL,FINDINGS}_20261005.md;
artifact native_organ_assimilation_roles_20261005.json;
scripts/audit_assimilation_roles.py,tests/test_assimilation_roles.py.

2026-10-05 E29 COMPLETE/VERIFIED, no promotion. Same-body
observation assimilation policy from E28 reachability evidence; immediate organ
innovation/delayed innovation/body innovation three equal-parameter arms.
Same E27 roles/proposals/2passes, unbridged and fullraw references; all53 inventory.
22tests and real fold0 input-contract preflight pass. New evaluator missing-ID
guard repaired before training, not evidence of old data corruption. Fixed
protocol RECONSTRUCTION_ORGAN_ASSIMILATION_PROTOCOL_20261004.md; scripts
organ_assimilation.py, train_organ_assimilation.py, verify_organ_assimilation.py.
Original session49649 exit0; fivefold15models complete,2048people/5120selection
occurrences/46scored targets. Immediate .876522 vs unbridged .875949/raw .853856
patient-macro normalized MAE; all five comparison screens and full screen0.
Saved assessment: artifacts/native_organ_assimilation_assessment_20261004.json.
Independent verifier13535 exit0; all15models/bothroles exact prediction replay,
baseline/hash/statistics match; maximum mean roundoff1.776e-15. Original assessment
verification=false preserved as historical snapshot; separate verification complete.
Diagnosis:46fit improvements vs unbridged,22selection reversals;28fit non-gains
vs body control. No near-tanh saturation in saved hooks;26regression tests pass.
Findings RECONSTRUCTION_ORGAN_ASSIMILATION_FINDINGS_20261005.md;
artifacts/native_organ_assimilation_{verification_20261004,diagnosis_20261005}.json.
Fixed branch closed; no runtime/interval changes, complete repair unproven.

2026-10-04 E28 COMPLETE: frozen last-organ reachability audit,15E27 models.
Same-time body-only outputs lack organ-proposal gradient path; positive-time
and body-bypass controls live. All15360context timestamps inventoried:
fit9403/10240 andselection4634/5120zero final gaps. No accuracy/causal or
information-loss percentage claim; overlapping fold contexts. Two tests,
no training/runtime/weight/interval change. Protocol/findings:
RECONSTRUCTION_ORGAN_REACHABILITY_{PROTOCOL,FINDINGS}_20261004.md;
artifact native_organ_reachability_20261004.json; script audit_organ_reachability.py.

2026-10-04 E27 COMPLETE: fixed paired organ-binding intervention,15 models verified.
Native vs pre-pool binding vs post-pool nonlinear control, frozen event JEPA,
same E19/E21 contexts; original full-fit raw baseline retained. Selection5120
occurrences/2048people/46targets; original assessment0 complete screens.
All saved fit/selection predictions reproduced exactly. Initial verifier40442
failure located to143MAE roundoffs from DataFrame layout; max1.776e-15. Existing
64eps applies only to MAE, paired stats/support/screens exact. Full31688exit0,
30tests; raw/statistics/summary verified.23/46 fit not improved over native,
9fit improvements reverse. No runtime change/promotion. Protocol:
RECONSTRUCTION_ORGAN_BINDING_PROTOCOL_20261004.md; assessment:
artifacts/native_organ_binding_assessment_20261004.json; verifier:
scripts/verify_organ_binding.py. Completed findings:
RECONSTRUCTION_ORGAN_BINDING_FINDINGS_20261004.md; verification/fit_gap/
statistics_diagnosis artifacts share native_organ_binding_*_20261004 names.

2026-10-04 E26: within-organ numeric/identity assignment collision identified.
Original additive pre-mixer pooling mathematically invariant to exchanging
observed tuples while identities fixed; distinct from ingestion-ID collisions.
224pairs,1344syntheticcases per mapping,fivefold/fourunique frozenepoch10 weights;
native organ proposal/update/6h0separated,body1344; parameter-free nonlinear
binding1344. Original formula parity<=4.44e-16. No training or promotion.
Allsaved E19 fittingcontexts:10240occurrences/3094uniquepeople;9855contexts
exposed to distinct co-observed members, not an error-causation proportion.
13tests; original role/cache/scaling/payload bindings checked; patientrows private.
RECONSTRUCTION_ORGAN_ASSIGNMENT_{PROTOCOL,FINDINGS,SUPPORT}_20261004.md;
scripts/audit_organ_assignment.py,audit_organ_assignment_support.py;
artifacts/native_organ_assignment{,_support}_20261004.json.
Local information merging located, wholebody bypass distinguishes inputs;
follow-up real-outcome binding intervention completed in E27 above. Runtime unchanged.

2026-10-04 E25 complete full-support organ readout:5319744queries/10roles,
940new HGB100 readers plus235verified raw controls; fixed frozen encoder/masks/roles.
Selection47targets/1797884queries/29580patients. Organs patient-macro .849663 vs
body .854767/raw .825737;112fit groups lose raw, all47selection patient means lose raw.
Screens body2/shuffled3/expanded0/raw0/native9; jointincremental/allcontrols0.
Independent1175savedreader replay and statistics verified83993exit0;41tests.
Verifier numerical summation changed to math.fsum after32p-value roundoff mismatches;
final max2.9976e-15, no tolerance or scientific gate relaxed. No runtime/promotion.
Full finding RECONSTRUCTION_FULL_ORGAN_FINDINGS_20261003.md; artifacts
native_reconstruction_full_organ_readout_{assessment,verification}_20261003.json.
This closes fixed fullfit concatenation, not all organ representations. Reused
development domains, not fresh lockbox. Earlier incomplete entries are historical.

2026-10-03 E25 continuation: added100000 original-query organ states; total
117408/5319744, all10roles inventoried, zero complete. Saved nativebody/point
errors0 and prefix/hash/key verification pass. New train_full_organ_readout.py
and score_full_organ_readout.py implement prespecified full-support pipeline,
refusing partial caches;37tests incl synthetic actualHGB100 and saving/replay.
No real full-support reader fits or accuracy screens yet; NOTpromotion.
Progress artifact native_reconstruction_full_organs_progress_117408_20261003.json.
Existing protocol/extraction source unchanged; complete rootcause unresolved.

2026-10-03 E25 FULL-SUPPORT ORGAN READOUT: admitted, NOT COMPLETE. E24 showed
the weak boundedraw control; this retains all E3 fitting/selection roles, all53
inventoried/47supported targets andfivefold. HGB100 body/process, trueorgans,
shuffledorgans, bodyexpansion, raw, nativecontrol;53x5prespecified screens.
Only extraction engineering executed:17408/5319744, nativebody/point exact,
raw/truth exact;1024independent organreplays exact; resumeprefix unchanged;
34tests. Fullreader training/scoring notdone, noefficacy/promote claim.
scripts/cache_full_organ_compartments.py,verify_full_organ_pilot.py,
audit_full_organ_progress.py; tests/test_full_organ_cache.py,test_full_organ_progress.py;
RECONSTRUCTION_FULL_ORGAN_PROTOCOL/PROGRESS_20261003.md; full_organs aggregate
pilot/progress reports. Privatecaches ignored; runtime/weights unchanged.

2026-10-03 E24 READER FIT SUPPORT COMPLETE: E3fullraw vs E23smallraw on
exactE23 selectionqueries, all5fold131groups/28supported/53inventoried.
Raw/truth/roles identical,smallfit subsetfullfit,no selection leakage.
No training: savedraw265/HGB100 weights. Full .638187/small .799944/native
.715713 patientmacro, .716334/.947003/.842483 componentmacro. Threefixed
contrasts,53x3corrected,5000bootstrap:0screens.27/28 bothmeanswinsvssmall,
20/28vsnative; notorganrepair. Fitmedian26325vs75, actualleaves15vs2.785
(posthocdescriptive).262modelreplays exact, independentlossaggregation,
summaryroundoff8.88e-16,12tests. Limitedfit confoundsE23negativeassessment,
notpureNcausality. scripts/audit_reader_fit_support.py,verify_reader_fit_support.py;
tests/test_reader_fit_support.py; RECONSTRUCTION_READER_SUPPORT_PROTOCOL/
FINDINGS_20261003.md; reader_support assessment/verificationaggregateJSON.
No release change; fullorganfitting/validatingrepair still incomplete.

2026-10-03 E23 ORGAN COMPARTMENT READOUT COMPLETE: same frozenepoch10/E21
masked contexts, all53 inventoried/28 supported, fivefold131target-folds,
1310 Ridge/HGB readers,4569queries/1851people. Body+organs versus body,
same-width shuffledorgans/bodyexpansion,raw; all eight comparisons0screens.
HGB body .783467/organs .789964/native .715713/raw .799944. Fitmedian75;
new small-support raw also weaker than native, NOT absence-of-organ-information.
Ridge131/131 fitbetter but100selectionreversals; HGB110fitbetter/56reversals.
Allsavedreader predictions reproduced; native independent replay15360contexts
body/organs/point exact. Sevennewtests. Trainer aggregation-key bug fixed via
separate score entrypoint without modifying source-bound fits/weights/gates.
RECONSTRUCTION_ORGAN_READOUT_PROTOCOL/FINDINGS_20261003.md;
scripts/probe_organ_compartments.py,score_organ_compartments.py,
verify_organ_compartment_replay.py; tests/test_organ_compartments.py;
eight organ_compartments aggregateJSONs. No runtime/interval/promotion.
Closed bounded direct-concatenation reader branch; full repair unresolved.

2026-10-03 E22 POST-OBSERVATION SUFFIX: frozenepoch10, exactE19/E20 contexts,
independent equal-specRidge/HGB readers on post/advance-only/update-only/full.
Fivefold177groups708newmodels88122queries2048people; 53family36supported.
Joint10post<update,8advance<body,0post<advance/0update<body;4x2x53screen.
Knowncontent attribution, NOT unknown inference or future repair. HGBmacro
.686716/.727531/.885117/.893199patient. 708savedreader predictions reproduced,
originalstates0error, pairedassessment exact, macro roundoff<=7.11e-15,37tests.
scripts/probe_state_suffix.py,verify_state_suffix.py;tests/test_state_suffix.py;
RECONSTRUCTION_SUFFIX_PROTOCOL_20261003.md/FINDINGS; seven suffixaggregateJSONs.
No runtime/promotion/weight/intervalchange. Next repair must preserve useful
history AND improve masked inference, not delete allupdates to rememberoldvalues.

2026-10-03 E21 CONTENT TRAINING: intervene in state objective, not frozen-reader
coordinates. Same E19 hash patients/context roles; fixed .25 auxiliaryweight,
one head warmup +two joint passes, primary/detach,real,shuffled;15models.
94908queries/2048patients; all53slots, masked46scored/27minimumsupport,
content47/41. Masked primary .875550->real .874808(0.0848%),0screens.
Content .887196->.885561(0.1842%),Hgb/RR bothcontrolpasses,notmaskedpromotion.
Known content improvement does NOT establish unknown inference repair.
All15 models exactpredictionreproduction, frozenencoderunchanged,24tests.
scripts/train_content_retention.py,verify_content_retention.py;
tests/test_content_retention.py; protocol/findings CONTENT_TRAINING_20261003;
seven native_reconstruction_content_training aggregateJSONs. Fixedbranch
complete,nohyperparametersearch,no runtime/weights/interval/promotion.

2026-10-03 E20 ASSIMILATION: same E19observed truths,53schema36supported/35all5.
177groups354newRidge/HGBreaders. Event/post/final each64+same159process.
88122queries/2048patients; duplicatefold lossesaveragedbefore pairedanalysis.
Fourcontrasts53*4screen: event>post19Ridge/12HGB(joint12),post>final19/17(joint13).
Known-content loss of readability already afterassimilation, additional latergap;
notirreversibleloss or masked inference/forecastrepair. BUN/Cr notjointpasses.
Fixedbatch event reproduction corrected BEFOREanyfit; final/event exact0,
originalbaselinesunchanged,all354savedreaders reproduced,177scalersfinite,24tests.
RECONSTRUCTION_ASSIMILATION_PROTOCOL_20261003.md +FINDINGS,sevenaggregateJSONs;
scripts/probe_assimilation_content.py,verify_assimilation_content.py,
tests/test_assimilation_content.py. No runtime/weights/interval/promotion.

2026-10-03 E19 OBSERVED CONTENT: last-observed event versus final persistentbody.
Known observations,NOTmasked-target inference.53schema,36supported/35all5;
2048fit/1024selection hashpatients per fold,onecontext each,177groups/1062fits.
88122queries/2048unique scoredpatients; repeatedfoldloss averagedfirst.
Ridge/HGBevent means better36/36;23/19correctedscreens,intersection16.
Evidence localizes accessible-content gap along state path,not irreversibility.
SevenaggregateJSONs observed_content_{fold0..4,assessment,verification}_20261003;
protocol/findings,probe_observed_content.py,verify_observed_content.py,
test_observed_content.py;19tests,531scalersfinite,allsavedpredictionsreproduced.
No encoderrepair/runtime/promotion; next separate immediateupdate from later
state evolution,do notrepeatE17pooling orE15alignment asuntested.

2026-10-03 E18 RAW BLOCK DIAGNOSTIC, all47/fivefolds,940fits,59629queries.
Same E17 bounded hash patients; refitted removal of latest, mean, lag/count,
not test-only corruption or another latent readout. Preserve presence/masking.
Raw patient macro .869179; remove mean .873490, latest .876621, lag/count .870159.
0correctedpasses in seven contrasts; timing/count not sole mean-advantage source,
but workflow remains via presence/sampling, no information-loss inference.
235 raw baseline reproductions/artifact groups verified;17tests,noouter/runtime.
RECONSTRUCTION_RAW_BLOCK_PROTOCOL_20261003.md + FINDINGS, six aggregate JSONs;
scripts/audit_reconstruction_raw_blocks.py, tests/test_reconstruction_raw_blocks.py.
Negative/limited diagnostic, not promotion; keep full root-cause objective open.

2026-10-03 STAGE PROBE COMPLETE: all47/fivefolds,2350 fits,59629queries.
Same frozenepoch10 and legal masked history; separately trained HGB/Ridge
eventmean,last,body with identical process covariates plusraw/process controls.
Fixed512fit/256selection hash sampling;5703unique patients.0correctedscreens.
HGBbody .885501 vsmean .911521;body better41/47means,Ridge36/47.
No universal aggregation-destruction evidence; no promotion/full-data claim.
RECONSTRUCTION_STAGE_PROBE_PROTOCOL_20261003.md + FINDINGS, sixaggregateJSONs,
scripts/audit_reconstruction_stage_probe.py,score_reconstruction_stage_probe.py;
12tests,hash/roles/query/truth checks. ROOT_CAUSE_LEDGER E17 and graph updated.

2026-10-03 HISTORICAL STATE-PATH RECONCILIATION (not a new experiment).
See ROOT_CAUSE_LEDGER E16: existing20260920 bounded7 BUN/all-target dynamics,
fit exposure, weighted objective, integration-step and finite-step diagnostics.
8 aggregate reports/22 nonprivate bindings checked;10 regression tests pass.
Reject repeating universal deletion/finer steps. Do not infer epoch10 failure
location from bounded7. No private replay, new fitting, outer or promotion.

2026-10-03 OBSERVATION-ALIGNED COORDINATES complete47/all5/705HGBheads.
Question: samebody information in fit-learned observation axes improves readout?
Frozen329-wide aligned/random/shuffled arms; 1797884sourcequeries,29580patients.
0fullscreens/0body/raw/controlwins;macrobodygain.2675/.1606%;45/47loserawmeans.
Fit229/235beatbody,153/235loseraw;decodeobservations!=recovermaskedtargetadvantage.
Numerics470scalersvalid,15tests;fixedbranchclosed,noouter/runtime/promotion.
RECONSTRUCTION_ALIGNED_COORDINATE_PROTOCOL_20261003.md and FINDINGS;
scripts/train_aligned_coordinates.py,score_aligned_coordinates.py,
audit_aligned_coordinate_numerics.py,summarize_aligned_coordinates.py;
artifacts/native_reconstruction_aligned_coordinates_{fold0..4,assessment,numerics,diagnosis}_20261003.json.
ROOT_CAUSE_LEDGER.md E15/graph.md updated; prior paused entry is historical.

2026-10-03 DOCUMENTATION ONLY: ROOT_CAUSE_LEDGER.md is the mandatory task-start
diagnostic entry, with evidence IDs, competing explanations, disconfirming tests,
closed branches and stopping rules. Not another experiment or promotion.
Aligned-coordinate intervention PAUSED by user: folds0/1/2=47 targets each,
fold3=17; full assessment absent. No resumption/training in this documentation task.
Root cause and full generalization repair remain unresolved; runtime unchanged.

2026-10-03 SAME-MAP Cx10 intervention COMPLETE47/all5/940fits,allconverged.
OnlyC100/n->1000/n, samefrozenmap/labels/offset, body/raw/process/shuffle.
Bodymacro .873760/.823452,raw .902195/.851674;gain1.45/1.40%and1.28/1.33%
vsoldC. Body3/raw6correctedoldCwins,0native/HGBwins,0fullscreens. Fitimproves
235/235real/processbutHGBfitstillbetter232/235. Notpower/strictgateonly.
RECONSTRUCTION_KERNEL_RELAXATION_PROTOCOL_20261003.md and FINDINGS;
scripts/train_relaxed_kernel.py,score_relaxed_kernel.py;
artifacts/native_reconstruction_relaxed_kernel_{fold0..4,assessment}_20261003.json.
23tests,oldfit/queryparity;noouter/runtime/weights/interval/promotion.
Closekernelrepairbranch: scaling+weakerpenaltypartial,notstrongbaselinefix.
Noquery-drivenC/bandwidth/landmarksearch;globalsharedstateproblemstillopen.

2026-10-03 FIT kernel objective/spectrum COMPLETE47/all5/1410savedmaps.
No refit/queryreads. Centeredkernel effective ranks31.39/33.91 scaledbody/raw;
rayL1gains1.3666/1.7865 improvefit .690/1.311% butworsenpenalizedobjectiveall235.
Rayobjectivealreadywithin5.23e-7,notfullmultivariateoptimalityproof. Oraclefit
gainstilllosesHGB232/235,recovers5.04/7.06%mediangaponly. Shrinkagepartial,
notmissinginformationproof orcompletefix;readoutdirection/mapstillopen.
RECONSTRUCTION_KERNEL_OBJECTIVE_PROTOCOL_20261003.md and FINDINGS;
scripts/diagnose_kernel_objective.py,summarize_kernel_ray_gap.py;
artifacts/native_reconstruction_kernel_{objective,ray_gap}_20261003.json.
20tests,noouter/runtime/weights/interval/promotion. Same-mapfit-internal
regularizationinterventionnotyettested; noquery-tunedkernelparametersearch.

2026-10-03 fit-only kernel SCALE REPAIR COMPLETE47/all5,940fits1797884queries.
WeightedStandardScaler beforefixedRBF/SVR, samecontrols/strongbaselines. Raw
lagdistance99.34->22.31%,numeric.412->29.23%; rawmacro2.53/2.92%betterold,
10targetsbeatoldrawkernel. Body/raw0fullscreens/0HGB-or-nativeincrements;
rawALL47patientmeansworseHGB,eachsourcefitlosesHGB232/235. Notpower-only.
19tests;705mapsfinite;constantvariancefloatingwarnings audited(-9.21e-34floor,
scale1);rawquerymaxz3566.7,21constantfitvariationcombinations,noqueryclipping.
RECONSTRUCTION_SCALED_KERNEL_PROTOCOL_20261003.md and FINDINGS;
scripts/train_scaled_kernel.py,score_scaled_kernel.py,audit_scaled_kernel_numerics.py;
artifacts/native_reconstruction_scaled_kernel_{fold0..4,assessment,numerics}_20261003.json.
Noouter/runtime/weights/interval/promotion. Partialmetricrepairconfirmed,
fullsignalreadabilityunresolved;kernel spectrum/objective diagnosticstilluntested.

2026-10-03 invariant RBF readability COMPLETE47/all5,940fits/1797884queries.
Fit-onlyNystroem128+LinearSVR epsilon0;body/raw/process/withincomponentshuffle;
allsolversconverged,rotationoutputerror<=5.10e-11.0fullscreens/0body-or-rawHGB
increments;bodymacro .884742/.834098,raw .937564/.889135. Bodybeatskernelraw17
butNOTinformationretentionproof. Postresultfitdistanceauditall235:lagdominance
median99.3419%,numeric.4120%,lag>90%EVERYcombination;scale-mixingdefectinnew
rawkernel,notprovenJEPAdefect. EachkernelsourcefitlosesHGB232/235.
RECONSTRUCTION_KERNEL_PROTOCOL_20261003.md and FINDINGS;
scripts/train_kernel_probe.py,score_kernel_probe.py,diagnose_kernel_metric.py;
artifacts/native_reconstruction_kernel_{fold0..4,assessment,metric}_20261003.json.
17tests/484assessmentbindings;noouter/runtime/interval/promotion. Scale-aware
metricrepairnotyetrun; fixedfailedmetricmustremaincomparator.

2026-10-03 lossless coordinate readability COMPLETE47/all5,940fits.
Question: finiteHGBcoordinatebias vsinformationloss? FullweightedPCA/random
orthogonal transformations, all64/265dims retained; same100HGB/labels/roles.
Rawrotation increasesmacro11.03/11.16%,47patient46componentmeanharms; inverse
error<=7.89e-13. Coordinateconfoundprovenwithoutinformationloss. BodyPCA0new
increment/fullscreen,mean .858206/.816076 vsbody .857139/.811241. Notrepair.
RECONSTRUCTION_COORDINATE_PROTOCOL_20261003.md and FINDINGS;
scripts/train_coordinate_probe.py,scripts/score_coordinate_probe.py;
artifacts/native_reconstruction_coordinates_{fold0..4,assessment}_20261003.json.
16tests/484bindings;1797884queries29580people. Noouter/runtime/weights/interval
change. Distinct fromoldtruncatedPCA64ridge; noencoderloss/fullretention claim.

2026-10-03 native-anchored nonlinear residual COMPLETE47/all5,940fits.
Real truth, frozenbody64+native, residual/absolute/scalar-only/conditionalshuffle
controls; same65width100HGB. 1797884queries29580people,485bindings,16tests.
0full8comparison screens;0vsbody/raw/absolute/observationskip,8vsnative.
Macro .855898/.810218 vsbody .857139/.811241,raw .825737/.787947.
46/47patientrawmeanharms: notjuststrictsignificance.8scalar/11shufflescreens
showstateinformation,notnewcapabilityoverbodyHGB. Noouter/weights/runtimechange.
RECONSTRUCTION_NATIVE_ANCHOR_PROTOCOL_20261003.md and FINDINGS;
scripts/train_native_anchor_probe.py,scripts/score_native_anchor_probe.py;
artifacts/native_reconstruction_native_anchor_{fold0..4,assessment}_20261003.json.
Priorlinearresidual/blendnegative retained; thisnonlinearparameterization now
testednegativeforincrement. Notirreversiblelossproof; fitanchorsnotOOF.

2026-10-03 teacher error geometry COMPLETE47/all5,1797884sourcequeries.
Question: wrongteacher imitation orfailedstudent approximation? Frozen parent
predictions, exact absolute-loss identity,patientfold-first/component means;
teacherquality x movement partitions. 93.59/92.58%grossharm awaytruth;
70.42/72.40%grossharm teacherworse; betterteacher+awaystudent17.40%population
alsoharms. Large benefits/harms cancel; descriptive hindsight,notlegalrouter.
RECONSTRUCTION_TEACHER_ERROR_GEOMETRY_PROTOCOL_20261003.md and FINDINGS;
scripts/diagnose_teacher_error_geometry.py,tests/test_teacher_error_geometry.py,
artifacts/native_reconstruction_teacher_error_geometry_20261003.json.
11tests/239bindings pass,no refit/outer/runtime/interval/promotion change.
Different from20260918jointtail/ranking audit; notuniqueencoderrootcauseproof.

2026-10-03 maskedraw teacher probe COMPLETE47/all5;705fits/1797884sourcequeries.
Samefrozenbody64, fixedoriginalrawHGBteacher fitoutputs; MAEstudents body/process/
withincomponentshuffle, noextra numericinput at inference.23full imitation passes
but0truth increments overbody/rawteacher;17/23patienttruthmeanharms. No promotion.
Macrotruth .853788/.811520 vsbody .857139/.811241; imitation14.53/15.83%better.
RECONSTRUCTION_STATE_TEACHER_PROTOCOL_20261003.md and
RECONSTRUCTION_STATE_TEACHER_FINDINGS_20261003.md; scripts/probe_state_teacher.py,
17tests, artifacts/native_reconstruction_state_teacher_{fold0..4,assessment}_20261003.json.
Optimalteacher median counterexample guards imitation!=accuracy. Source-only,
insampleteacher targets, noouter/runtime/weights/intervalchange. Notoldlatent
retention; no irreversiblecompression orfullretention claim.

2026-10-03 observation skip COMPLETE47/all5,705fits,1797884source queries.
Question: recover maskednumeric observations beside SAMEfrozenbody,notoutput
blend oroldcalibrationridge. FixedMAEHGB329:body64+raw265/process/shuffledraw;
fit-only,patient/componentdisjoint,source-only,alloriginalreferencesverified.
10bodyincrements;6beatbothbranchcontrols(albumin/ALT/AST/HCO3/HR/totalprotein).
0raw-onlyincrements,0fullsixcomparisons; macro .828889/.784964 vsbody
.857139/.811241,raw .825737/.787947. Keep candidate; noouter/runtimepromotion.
RECONSTRUCTION_OBSERVATION_SKIP_PROTOCOL_20261003.md and
RECONSTRUCTION_OBSERVATION_SKIP_FINDINGS_20261003.md; train/score_observation_skip,
12tests, artifacts/native_reconstruction_observation_skip_{fold0..4,assessment}_20261003.json.
Exposingnumericinput helpsbodypath,but sharedstatebeyondraw stillnotproved;
notirrevocablecompressionlossevidence,norfutureforecastresult.

2026-10-03 SAME-state nonlinear readout blend COMPLETE47x5/source only.
Native+body100 coefficient fits exact equal-component MAE on OTHERcomponents;
matched native+process/native+shuffle and fixed50:50 controls. 1,797,884queries,
44fullysupported targets,unsupported retained. 0fullscreens/0vsbody100corrected
increment. Error Pearson median.98937,rank.96640;opposite9.369%. Tiny macro gain
butferritin/troponin retreattonative notbeatit;BUN/INRslightharm;O2satuncertain.
26tests/13bindings,alloriginalmeans reproduced;private rows notcommitted.
RECONSTRUCTION_READOUT_BLEND_PROTOCOL_20261003.md and FINDINGS ofsameprefix;
artifacts/native_reconstruction_nonlinear_blend_20261003.json. Distinct readout
set fromold near-miss/own-full blending,notnewmethod. No selectedprefix stacking
because indirect labelleakage;no outer/encoder/runtime/interval/promotionchange.

2026-10-03 frozen readout PREFIX complexity intervention COMPLETE47/all5.
sklearn staged_predict1/5/10/25/50/100; choose stage using other source components
only, same search for body/raw/process/shuffle. Full1,797,884queries;1288supported
23unsupported component decisions,44targets fullysupported. 0fullscreens and
0corrected incremental vsbody100; patientmacro tinygain,component unchanged.
Troponin body1005.226455->prefix5.177608patient;component3.699823->3.641345,
but native/control gates fail. Centralfit harm grows afterearlystages; tailgain
compensates. Ferritin/BUN/INR patient means worsen,not universaloverfitfix.
20tests, endpoint parity and boundfiles verified. Findings/protocol
RECONSTRUCTION_PREFIX_COMPLEXITY_*_20261003.md and aggregate report
artifacts/native_reconstruction_nonlinear_prefix_20261003.json. Private row
outputs staylocal. Reusedsource/conditionalbootstrap,notforecast orpromotion;
no runtime/weights/interval changes. Early-stopping hypothesis now tested.

2026-10-03 nonlinear FIT-versus-SOURCE selection diagnostic COMPLETE47/all5.
Frozen existing HGB/native/controls,3,521,860 fit occurrences,no refit/outer labels.
232/235 body fit benefits vs native;54patient/58component reverse on selection.
Ferritin central fit gain->query harm5/5;troponin central fit ALREADY harms5/5
while overall fit improves. BUN/INR all5 overall both-role benefit;not promotion.
Fit-bounded body beats weighted median and null actual training assignment235/235.
Distinguish null own-state evaluation from its fit objective. Float32 label
convention verified every row; original float64 population unchanged.
All25 bindings match,14tests; findings/protocol RECONSTRUCTION_NONLINEAR_FIT_GAP
dated20261003; artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json.
Current masked reconstruction only,reused source,overlapping folds;no runtime/
weights/gate change. Next candidate same-head stage-complexity diagnosis,
not yet tested or a validated fix.

2026-10-03 nonlinear SOURCE support-decision probe COMPLETE47/all5.
Within-original-fold leavecomponentout, weightedRidge100 estimates body-minus-
native absoluteerror;constant/aggregate4/mask-age106/shuffle106. No targettruth
or numericvalue feature, no outerfit.1288complete23unsupported groupfits;44
fullysupportedtargets. Variablechoice worsebody/constantmacro,0fullscreens;
only1/45patient2/45component error-differenceMSE betterconstant. Selectoractive,
notconservativegateblock.12tests. RECONSTRUCTION_SUPPORT_SELECTOR_PROTOCOL_20261003.md,
RECONSTRUCTION_SUPPORT_SELECTOR_FINDINGS_20261003.md andaggregateJSON. No
runtime/promotion. Next SAMEheadfit-vsselection diagnosis,not more support
rules/offsets presentedasnew. Privatequeryoutputs remainlocal.

2026-10-03 nonlinear SOURCE error decomposition47targets/all5 COMPLETE.
Fit-normalized truth bands with patient-equal fold weights,matched percomponent
patient-hash median correction ALLarms. Ferritin/troponin regress near center,
improve tails; after matched offsets still worse. O2sat small reversal after
centering not promotion. BUN/INR benefit patterns differ; no one universalfix.
RECONSTRUCTION_NONLINEAR_SOURCE_ERRORS_20261003.md +
native_reconstruction_nonlinear_source_errors_20261003.json;7tests andoriginal
aggregate/hash parity. Retrospectivebands cannot route runtime. No outerfit,
new weights,intervals,promotion. Next maskedinput-support diagnosis,not prior
offset/tail/loss ideas relabeled as new universal repair.

2026-10-03 nonlinear readout SOURCE-only precision/influence COMPLETE47x4.
Pairedpatienterrorsaverageoverlappingfolds,originalMAEreproduces;allgroupskept.
BUN/INR12/12beneficialmeansvsallcontrols,leavesingleoutnegative,no supported
nominal/twohalfharms;sourceblockedcertainty/heterogeneousmagnitude notsignflip.
Troponin/ferritin/O2sathaveadversegroups;18/47targets nominalharmvsnative.
NominalCIsnotmultiplicitycorrected,varianceconditional/singletonsunknown.
RECONSTRUCTION_NONLINEAR_SOURCE_COMPONENTS_20261003.md +
native_reconstruction_nonlinear_source_components_20261003.json;9focusedtests.
No training/gaterelaxation/outerselection/runtimechange. Do not use same
repairforall:additionalfrozen/disjointdevelopmentevidenceforconsistentnear-miss,
conditionalerrorrepairforgenuinelyadversetargets;notnewlockboxorfinalpromotion.

2026-10-03 same64dim/fixedHGB readout shuffle COMPLETE235fits,all47/fivefolds.
Withinfitcomponent reassignment, noquerytruthfit, realtrainedencoderunchanged.
Source1797884queries/29580people,outer1800214/73670; originalparity/hash/rolespass.
Matchedshuffle screens29source/26outer/24overlap. Sevenpriorouterjointpositives
allretained,butsourceconjunction4;bothrolesonlyALT/AST. Notruntimepromotion.
BodyouternormalizedMAE .891238/.877110 vsnull .983810/.965183.24tests.
94singletons/3521860fitpairoccurrences retainedexplicitly,notindependentpeople.
RECONSTRUCTION_NONLINEAR_PLACEBO_FINDINGS_20261003.md +protocol,5foldproofs,
native_reconstruction_nonlinear_placebo_assessment_20261003.json. Identifies
readoutpairingbenefit,notfullyshuffledencoder/physiology-only/newexternalevidence.
Remainingfirstblockerfor5outerpositives:sourcecomponentincremental/control
confidence(12groups);don'tlowergate/inferpower-only. No runtime/privateGitdata.

2026-10-03 fixed nonlinear readout diagnostic COMPLETE47targets/fivefolds.
Same epoch10 sharedstate frozen; fit-only HGBabsolute_error100trees, body64/
raw265/process265 versusunchangednative/population. Source29580uniquepeople,
outer73670people1800214pairs; noouterfit/headselection, exactroles/hash/parity.
Bodyouterpatient/componentmacro improves2.85%/1.90%, raw5.98%/4.48%. Body7
incremental+controls8x47screens(ALT/AST/BUN/Ca/HR/INR/K), raw15; diagnosticnot
fullpromotion. Body15/47harm>=1mean; withoutALTASTgain1.23%/.88%; sourceouter
incrementalintersection4. Notuniqueencoder-rootcauseproof ornewexternalclaim.
17tests, completealljobs. RECONSTRUCTION_NONLINEAR_FINDINGS_20261003.md; three
aggregate source/outer/robustnessJSONs plus20cache/training/inferenceproofs.
No runtime/interval/forecast/neuralweight change. Privatefeatures/headsnotGit.
Next source-designed matchedshuffledstate fornewreadout, notnativepermutation
reuseorouter-derivedrouting; fullgeneralizationstillopen.

2026-10-02 input delivery integration repair, not a prediction experiment.
Old archive lacked the native-history builder, while weight-only/private-cache
tests passed. New canonical CSV bridge and CLI retain original time/mask/unit
semantics. Isolated synthetic 188 fold-target parity checks +4tests pass; no
patient data and identical checkpoint hashes. Complete private GitHub release
coverage-reconstruction-20261002-input-complete; see
COVERAGE_INPUT_DELIVERY_FIX_20261002.md and input bundle verification JSON.
No runtime/promotion/interval change; ongoing nonlinear outer inference separate.

2026-10-01 completeSOURCE current-reconstruction47target conditionaldiagnosis.
Fixedpatienthash two-way percomponentmedian correction(real/process/population)
plus exactexpectedwithinhalf outputpermutation. Recurring56/69meanharms retain
35/45aftermatchedcorrection;29/37ofremainingstillhavepositiveassignmentbenefit.
Notnewruntimecalibrator/independentreplication/significance/promotion.30tests;
RECONSTRUCTION_SOURCE_ERROR_MECHANISM_20261001.md +residuals/assignmentJSONs.
Allgroupsretained,insufficienthalvesnull,noouterlabelcorrection. Biaspartly
explainsbutdoesnotresolvefailure;noindividualinformationclaimalsorejected.
No neuralweights/runtimechange/livejob. Do notrepeatglobaloffsetasnewfix.

2026-10-01 fullSOURCE10 fivefold pairedassessment COMPLETE/all47.1797884query
occurrences,29580uniquepeople;overlappingqueries/fitsnotindependent. Original
source8metrics/monitor10parity reproduced<=4.44e-16;18tests. Patientmacro3/5
component4/5improve. Fullsource/outerchangedirection differs103/233component;
17/26outerfailtargets recurring>=30personsourcecomponentmeanharmvscontrol.
See RECONSTRUCTION_FULL_SELECTION_FINDINGS_20261001.md, fiveverificationJSONs,
native_reconstruction_full_continuation_selection_20260930.json. Notsolecause
proof/significantharm/promotion;no runtimechange/livejob. Needconditionalerror
diagnosisusingcompletepairs;priorgenericreadout/calibrationnegativesretained.

2026-09-30 fullSOURCE10 progress folds0/1 COMPLETE363526pairs14845peopleeach47targets.
0certifiedouter2reuse,1newinference;overlappingqueriesnotindependent. Bothoriginal
fixedmonitors reproduce<=2.22e-16;hash/role/gridverified,8tests. See
RECONSTRUCTION_CONTINUATION_SELECTION_PROGRESS_20260930.md. Remaining2/3/4+
fullpairedassessment;no efficacy/promotion/runtimechange;no livejob.

2026-09-30 SOURCE/monitor diagnosticCOMPLETE47targets. Fullsource8 originalmeans
reproduced;18/26outerfailtargets havesupportedsourcepopmeanharms,17overlapwith
recurringouterharm. Monitor/outercontinuation signs differ103/233components,
84/233patient;notindependent/significantcomparisons. Missing2outercells null.
RECONSTRUCTION_CONTINUATION_SOURCE_20260930.md;aggregateJSON;10tests pass.
Fullsource10notavailable;needpairedfullsourceassessment. No fitting/promotion.

2026-09-30 continuation CONTROLDRIFT/all47COMPLETE. Reusedsameaggregatecomponents,
exactMAEmargin decomposition/crossedepochprocess comparisons. Newreal-oldprocess
component screen retains3of5additions,notPTT/uricacid;oldreal-newprocess passes
latter2.21/26failedtargets recurring>=30person meanharm vsfixedpopulation,
notstatisticalharm/newgate. RECONSTRUCTION_CONTINUATION_MARGIN_20260930.md;
artifacts/native_reconstruction_continuation_margin_20260930.json;13tests.
No fitting/runtime/promotion;nextsource-only check,notouter-informedselection.

2026-09-30 continuation FULL OUTER5folds1800214pairs73670patients47targets complete.
Same47family gates;5integritytests pass. MacrorealMAE improves~.37%patient/.23%
component. Base screens17->21;onlyALTpassesbothbaseandincrementalscreens(28/37
groupsimprove,9harm). Lostbicarbonate;new5basescreenslackincrementalpass.
26basefailures21allmeansbetterbutuncertain;sourcecomponentconsistency remains
the main measuredblocker,notsolecauseproof. RECONSTRUCTION_CONTINUATION_OUTER_FINDINGS_20260930.md;
artifacts/native_reconstruction_continuation_outer_20260929.json. Reuseddevelopment,
candidateonly,no runtime/promotion;no livejob. Epoch10fourfit sanitizedrelease
alreadyuploaded toprivateGitHub withmatching source/scalers/usage/hashmanifest.

2026-09-30 fold2 continuedfit equivalence certified; independent monitor
8/9/10 real/process completed. Fixedmonitor realMAE worsens, but full heldout
363526pairedqueries show tiny real mean gain and process harm. Sampled monitor
is not an adequate full-role surrogate in this fold. Scripts
certify_monitored_continuation_fold2.py,monitor_reused_fold2.py,
summarize_monitored_continuation.py,infer_monitored_continuation_outer.py;
RECONSTRUCTION_CONTINUATION_FOLD2_20260930.md. Remainingouter0/1/3/4 and
prespecifiedpairedall-target scoring. No runtime/promotion/interval change.

2026-09-29 fold3 real/process fixedepoch10 complete.7167fixedmonitorqueries,
47targets,pairedRNG/queues/seen350083verified. Realpatient/componentmacro
improve~.13%/.47%;26/28targets improve, processmacroworsens. Descriptiveonly,
notpromotion. RECONSTRUCTION_CONTINUATION_FOLD3_20260929.md. Fold4both and
fold2certification/fullassessment remain;runtimeunchanged.

2026-09-29 fold1 fixedepoch10 real/process continuation complete;50384updates
perextraepoch,pairedseen358799/RNG/queuesexact. Fixedmonitor realmacro slightly
improves,butonly24/47patient22/47component targets. Processmacroslightlyworse.
See RECONSTRUCTION_CONTINUATION_FOLD1_20260929.md;no promotion/runtimechange.
Remainingfolds3/4both,fold2certification/fullassessment. Notindependentof fold0.

2026-09-29 fold0 real/process fixedepoch10 continuation COMPLETE.6862identical
monitorqueries,47targets,pairedRNG/pending/seen exact358828. RealMAEmacro slightly
better,processMAEworse; notalltargetsbenefit/no statisticalclaim. See
RECONSTRUCTION_CONTINUATION_FOLD0_20260929.md;private immutable epochmonitorJSONs.
Otherfolds andpairedfullassessment pending; no runtime/promotion change.

2026-09-29 latest candidate release: fold0 real fixed epoch10 and6862-query
monitor completed. Export exactstate/scaler and47/47source-history output parity;
isolated strictload andremoteSHA256verified. GitHub prerelease
coverage-reconstruction-20260929-fold0-epoch10 contains actual sanitizedweights,
matching source, usage, hashes andparity; no patientdata/optimizer. Other
continuation folds/controlarms unfinished; no promotion/runtimechange.

2026-09-29 monitored continuation firststage:fold0real epoch9+fixed6862query
monitor COMPLETE. Epoch8reproduced beforetraining;immutable9saved. Patientmacro
.911950->.909960,component .906847->.907090. Descriptiveonefold,notstopselection.
Fixed10endpoint andotherarms/folds pending. No runtime/promotion.

2026-09-29 fixedsource monitor implemented,epoch8baseline fromverified predictions
all47x5;max32identityhashqueriespercomponent.4tests role/identity/completeness/
weighting. RECONSTRUCTION_FIXED_MONITOR_20260929.md +epoch8aggregateJSON.
No newtraining or continuationclaim;monitor integration/checkpoints pending.

2026-09-29 coverage loss replay:4independent exactschedules,constantmedian
loss varies,epochcorrelation withmodelloss .699-.799. Online lossesnotfixedcohort
learningcurves;cannotdiagnoseconvergence. RECONSTRUCTION_COVERAGE_LOSS_COMPARABILITY_20260929.md
andJSON.1objectiveparitytest;no training/runtime/promotionchange.

2026-09-29 coverage variance:21/30failedtargets singletonblocksfullnoiseestimate;
9estimable. MAP2singletons,0supportednominalharm17benefitgroups eachreference.
18/19targets have supportednominalharm somewhere;notmultiple-adjustedproof.
RECONSTRUCTION_COVERAGE_VARIANCE_20260929.md +JSON;3testspass. No promotion.

2026-09-29 coverage component influence:30failures29majoritysignwinprocess,
26leaveanyonenegativemean. MAPsmallgroupsharmlargegroupsbenefit;notuniversal.
No exclusion/thresholdchange. RECONSTRUCTION_COVERAGE_COMPONENT_INFLUENCE_20260929.md
andaggregateJSON,3tests. Next distinguish samplingnoise vsdomainheterogeneity.

2026-09-29 coverage vsreplacement frozennumericablation:47x5fixedhashmax128,
29939sourcequeries. Exact savedpredictionreproduction. Neutralizationharms
207/235new197/235old,median .048253/.040595. Numericpathactive; notfullcohort,
nottransfer orcausalproof. RECONSTRUCTION_COVERAGE_NUMERIC_PATH_20260929.md.
No runtime/promotion. Do not repeat as novel evidence that anyvalueisused.

2026-09-29 coverage support/freshness SOURCEonlydiagnosis235targetfolds.
Numericadvantage weaklyassociatedwithaggregatecounts/meanage:medianrho
.02044/.01559/.02175/-.00317. No fittedselector/outeruse. Cannotruleout
variable-specific/nonlinear effects. RECONSTRUCTION_COVERAGE_SUPPORT_FINDINGS_20260929.md.
No runtime/promotion;next matchednumericperturbations afterprior-historycheck.

2026-09-29 Coverage failure decomposition:17pass25uncertainnegativeallmeans,
5meanfail.29of30 fail numeric-vsprocess atcomponentlevel;23populationcomponent.
Directblockerlocated,notuniquerootcause. MAPpatientimprovementnotstablegroupgain.
RECONSTRUCTION_COVERAGE_FAILURE_DIAGNOSIS_20260929.md;
native_reconstruction_coverage_failure_diagnosis_20260929.json.3testspass,
exactexistinggatedecisionagreement. No runtime/promotion.

2026-09-29 Coverage outer comparison COMPLETE:1800214pairs73670people47targets.
Samebudget broaderexposure reducesmacroerror~0.5%;base descriptive13->17retains13.
Incremental vsoldreal onlyMAPpasses47-familybothlevels, but MAPbasefails.
Nojoint base+incrementalpass;no promotion.20/47bothmeansimprove,processalsoimproves.
RECONSTRUCTION_COVERAGE_OUTER_FINDINGS_20260929.md;
artifacts/native_reconstruction_coverage_outer_evaluation_20260929.json.
Not untouchedexternal or futureprediction. Allfiveoriginalbaselines reproduced.

2026-09-28 coverage outer3 COMPLETE380292pairs14754people. Frozenfourarms;
testrole,uniqueidentity,finite/hash and exact originalbaseline reproduction
verified. Private coverage_outer_20260927/fold3.json. Onlyouter4 pending before
full aggregate interpretation. No runtime/promotion change.

2026-09-28 coverage outer2 certified reuse:selection0 samequeries and allfour
trained states/scalers/medians/cache. Exact priorbaseline reproduction.363526pairs
14845people; not independent replication. scripts/certify_coverage_outer2.py;
private coverage_outer_20260927/fold2.json. Outer3-4 remain; no promotion.

2026-09-28 coverage outer1 complete:356944pairs14735people. Four-arm frozen
inference; roles,identities,finite/hash and exact original baseline reproduction
verified. Private coverage_outer_20260927/fold1.json records dependencies.
Outer2-4 pending, no outer efficacy interpretation/promotion/runtime change.

Coverage outer0 inference COMPLETE380978pairs14759people. Explicit test role,
same four-arm frozen inference. Exact baseline reproduction and paired identity,
truth/scaler/population,finite/hash checks pass. Output private
coverage_outer_20260927/fold0.parquet/json. No outer efficacy interpretation yet;
outer1-4 remain. Runtime/promotion unchanged.

Coverage sampling source-selection contrast COMPLETE all5folds. Sameupdates,
47targets, four frozen arms. Patient macro improves3/5, component5/5; descriptive
only, overlapping query patients/fits. Not a uniform repair or promotion.
RECONSTRUCTION_COVERAGE_SELECTION_FINDINGS_20260927.md;
artifacts/native_reconstruction_coverage_selection_summary_20260927.json.
Outer development comparison still pending, fixed protocol unchanged.

Coverage selection fold2: completed fixed four-arm paired inference on356944
pairs/14735people. Verified role/identity/finite outputs/hash. Selection differs
from fold0, so no prediction reuse. Private metadata:
data/research_recovery/additional_development_domains_20260921/coverage_selection_20260926/fold2.json.
No efficacy claim until allfold aggregation; selection3-4/outer remain pending.
No runtime change or promotion.

COVERAGESELECTION1 COMPLETE363526pairs14845peoplefourarms;selectionmembership,
unique/finiteverified. Samequerypopulationasfold0notnewpatients. Noefficacyyet.

COVERAGESELECTION0 COMPLETE363526pairs14845people4arms. Exactoldbaseline
reproductionagainstoldouter2(identity/scaler/truth/predictions)verified.
Noefficacyinterpretationuntilaggregate;no runtime/promotion.

COVERAGEALLTRAINING COMPLETE:fold4processfinished;all5pairedcheckpoints8epochs,
403072updates,seen/RNG/pendingmatchexact.64trained+16certifiedequivalentepochs.
Noefficacyyet. native_reconstruction_coverage_training_fold4_process_20260925.json.

COVERAGEFOLD4REAL COMPLETE8epochs403072updates305209seen,planverified.
native_reconstruction_coverage_training_fold4_real_20260925.json;noefficacyclaim.

COVERAGEFOLD3PROCESS COMPLETE8epochs403072updates290051seen;matchedreal
RNG/pending/seenverified. native_reconstruction_coverage_training_fold3_process_20260925.json.
Noheldoutaccuracyclaim.

COVERAGEFOLD3REAL20260926 COMPLETE8epochs403072updates290051seen,planned
exposureverified. native_reconstruction_coverage_training_fold3_real_20260925.json.
Trainingonly,nopromotion.

COVERAGEFOLD1PROCESS20260926 COMPLETE8epochs403072updates296623seen;
real/processmatchingseen,RNG,pendingconfirmed. Artifact retains planneddate
native_reconstruction_coverage_training_fold1_process_20260925.json.
Noheldoutscore/promotion/runtimechange.

COVERAGEFOLD1REAL COMPLETE8epochs403072updates296623seen,simulationmatched.
native_reconstruction_coverage_training_fold1_real_20260925.json;noefficacy.

COVERAGEREUSE20260925 CERTIFIEDfold0->2,samefit/scaler8schedules/RNG/pending,
privatecheckpointsstrictreload/hashchecked. Notindependentreplication.
native_reconstruction_coverage_fold_reuse_20260925.json. Noefficacy/promotion.

COVERAGEFOLD0PROCESS COMPLETE8epochs403072updates296652seen;exactreal/process
seen/RNG/pendingmatchverified. Trainingonly,no heldoutaccuracy/promotion.
Artifact native_reconstruction_coverage_training_fold0_process_20260925.json.

EXPORTPARITY20260925:47source-selectionnativehistories,public/private
scalers+predictions exact. Smokeonly/noaccuracy/promotion;export_parityJSON.

COVERAGEFOLD0REAL20260925 COMPLETE8epochs403072updates296652seen;trainingonly.
UserrequestedweightsuploadedGitHubprereleasecoverage-reconstruction-20260925-fold0.
Sanitizedweightsstrictloadindependentdirectorypasses;archive+manifest+usage,
no patientindices/rawdata/optimizer. Noaccuracy/promotionclaim.

COVERAGETRAINING20260925 WIREDNOTRUN:isolatedtrainer+checkpointpendingcycle,
sameoriginaloptimization,noresumeoldweights. CLIhelp/diffchecked. Fixedprotocol
RECONSTRUCTION_COVERAGE_TRAINING_PROTOCOL_20260925.md. Noefficacyclaim.

COVERAGESCHEDULE20260925:sampler3tests+5existing,exactoldseenreproduction4fits,
~25%moreuniqueatfixedbudget. Simulationonly/nofit/promotion/runtime.
RECONSTRUCTION_COVERAGE_SCHEDULE_20260925.md,coverage_scheduleJSON.

ENCODEREXPOSURE20260925:checkpointseen/fitpools audit,~33%uniqueeligiblepairs,
commontarget~25%;boundedencoderweightsnotuncappedcorrector. Fixed8draw-epochs
not8passes/convergence. RECONSTRUCTION_ENCODER_EXPOSURE_20260925.md;noruntime.

WEIGHTRESPONSE20260925:source163rho+.3421,descriptive/noindependentp;
sizebinharms~42-46%,allbinmedianimproves. Notmerelysmall-domaintradeoff.
PATIENT_WEIGHTfindingsupdate,weight_responseJSON;no runtime/promotion.

PATIENTWEIGHT20260925 COMPLETE:singlefixedweightablation163matchedsource
LOCO;109beatbalanced81beatunchangedcomponentmeans,29bothrefs/bothlevels,
no5foldjointtarget;163/358smallqueryharms.2tests,noouter/runtime/promotion.
RECONSTRUCTION_PATIENT_WEIGHT_FINDINGS_20260925.md.

WEIGHTCONCENTRATION20260925:231nonemptycalroles223smallcomponents,median
KishESSratio.017181;patientweightimbalancequantified,notcausalproof/encoder
diagnosis.3tests,nofit/runtime/promotion. RECONSTRUCTION_WEIGHT_FINDINGS_20260925.md.

CONCENTRATION20260925:point-onlysourceCV162matched,54meanwins43majority,
36leaveanyoneout,1allcomponents. Smallquerygroupsensitivityflips30means.
MAP5foldstablevsbase,chloridefold4fragile. Descriptive,norefit/noouter/
promotion/runtime.2tests. RECONSTRUCTION_CONCENTRATION_FINDINGS_20260925.md.

POINT-ONLY20260925 COMPLETE:fixedHuberonlyfrozenpointinput,sourceLOCO47x5,
163complete162matched. Betterthanfull127,unchanged54,offset56,allrefs15;
97bothlosebase. Chloride/MAP5/5basebutnoallrefs5/5.2tests,noouter/runtime/
promotion. See RECONSTRUCTION_POINT_ONLY_FINDINGS_20260925.md and point_only JSON.

NUMERICAL REPAIR20260925:fivefailed sourceHuberfits reproduced+sameobjective
warmcontinued,allconvergedlittlepredictionchange4/5stilllosebase. Original75
completecomparisons exact;O2sat0addsone=>76/235,159supportlimited. Notglobal
accuracyrepair.1test,noouter/runtime/promotion. See
RECONSTRUCTION_NUMERICAL_REPAIR_20260925.md and robust_convergence/seen_domain_completed JSON.

SEEN DOMAIN20260925 COMPLETE:sourceonlyfixed20%querypatients,matchedfitN,
samequeriesseen/unseencomponentHuber.75/235complete;60seenbetterthanunseen,
47bothloseunchanged,17bothwin,7seenonly,4unseenonly.160incompleteexplicit.
Notdomain-onlyfailureorcausaleffect;2tests,noouter/runtime/promotion.
See RECONSTRUCTION_SEEN_DOMAIN_FINDINGS_20260925.md and seen_domain JSON.

ROBUST SOURCE20260925 COMPLETE:fixedHuber47x5sourceLOCOONLY,162complete;
36beatunchanged(vs25ridge),83beatridge,13beatallreferences;no target5/5
allreferencewinner.5nonconvergedcomponentrefitsexplicit;noouterrun/promotion.
Prior138CVharms122surviveanysinglecomponentremoval.2new+5priortests.
See RECONSTRUCTION_ROBUST_SOURCE_FINDINGS_20260925.md and robust_source JSON.

CONDITIONAL REPAIR20260925 COMPLETE/REJECTED:all47x5,1800214outerpairs,
numeric/processcontext ridgealpha.1 versus unchanged/medianoffset.0screens;
numericmacro1.119574vs.925567.138/163completeSOURCE-LOCOharm;111fitMAEbetter
thenCVworse,27fitMAEworse. Identicalfitobjective replayall235reproducesouter;
35/189supportedfitMSEimprovesbutMAEworsens. Notonlyobjectiveorouterdatabase
problem.5tests,noruntime/promotion. Protocol/results/fitobjective JSON:
RECONSTRUCTION_CONDITIONAL_REPAIR_FINDINGS_20260925.md.

VARIABLE TRANSPORT20260925 COMPLETE: exact1800214masked snapshots,all235folds
(6emptyroles),12137variablecomparisons.857observedfraction shifts>=.2;
5160supportedwithincomponentresidualcorrelations,1291signflipsbutonly4both
abs>=.1 (531bothstronger). Differentobservationidentitiesconfirmed,notgeneral
physiologyreversal/rootcauseproof. No fittedhead/runtime/promotion;7newtests.
Priorrawridge negative notrerun. See RECONSTRUCTION_VARIABLE_TRANSPORT_20260925.md
and native_reconstruction_variable_snapshot/variable_transport_20260925.json.

Additional-domain matched training STARTED (2026-09-23):
SUPPORTMATCH DIAGNOSIS20260925:235tested136anysupported;0>=80%retainedboth.
Selectedbinscannotestablishpopulationrootcause;no exclusions/promotion.
3tests. See RECONSTRUCTION_SUPPORT_MATCHED_SHIFT_20260925.md.
EXACT SUPPORT20260925:1800214maskedhistoriesoutcome-freefeatures,exactouter
join/payloadSHA. Albuminfold2counts/freshnesssimilar,notuniversalolder/fewer
explanation.2tests;no fit/promotion. See ADDITIONAL_OBSERVATION_SUPPORT.
CONDITIONAL SHIFT DIAGNOSIS20260925:235targetfolds,229complete139supported;
124within-binmeanresidualcontributionlargerthancomposition. Coarseprediction
binsourceonly,notcausal/MAEproof.3tests,no correction/runtimepromotion.
See RECONSTRUCTION_CONDITIONAL_OFFSET_SHIFT_20260925.md.
OFFSET ACTIVATION COMPLETE20260925:real8/235active,5componentgains3harms,
0incrementalscreens. MAPallmissed;albuminharmprevented. Fixedsource-onlyrule,
allarms;twotests,nothresholdsweep/runtimepromotion. See ACTIVATION_FINDINGS.
SOURCE-CV TRANSPORT DIAGNOSIS20260925:235targetfolds,163complete.42help/help,
21help/harm,47nonhelp/help,53nonhelp/nonhelp. MAPsource+outerhelp5/5;
albuminsourceharm5/5outerharm4/5. Source evidenceignoredbyunconditionalfit,
notperfectselector. Two tests,nogate/runtimechange;see OFFSET_TRANSPORT.
MULTIDOMAIN OFFSET COMPLETE20260925:all5cal,189fit46zero;235replayverified.
29/27targetmeansbetter;0fullincrementalscreens. MAP13.26625->12.82157,
partialbiasrepairnotfulltransfer. Albuminworsens/sodiumstillloses. No runtime/
promotion. Newscorerraretargetunionfixbeforefit,5offsettests. See calibrationfindings.
Multi-domainpointcalibration20260925:fixedprotocol,fold2calexactequivalent
reuse380978pairs14759people;fold0fresh356944pairs14735peoplecomplete.
Role/identity/labels/scalesverified,two tests. Folds1/3/4pending,nooffsetfit
or correctedscore/runtimepromotion. Notintervalcalibration/freshlockbox.
All47domainfailurediagnosis COMPLETE:MAP17/27supportedcomponents harmed,
LOOallharm;notonesmalloutlier. HR/bicarbmeanbenefit survivesLOO/ordinaryCI,
failsfixedmultiplicity. Sodiumsupportedsensitivityworse. Mixedbiasdirections;
no outer-fittedcorrection/exclusion/promotion. Two tests;see DOMAIN_DIAGNOSIS.
ALL OUTER SCORING COMPLETE:1800214pairs73670people47targets38components;
13descriptivepasses,21allmeanbetterbutnonsignificant,13meanfailures.
Normalizedmacro .925567/.975469/.978512(real/process/population).
Exactidentity/hash/rolechecks;maskedCURRENTreconstruction,nopromotion.
See RECONSTRUCTION_ADDITIONAL_FINDINGS_20260924.md and aggregateJSON.
Outerfold2 COMPLETE:363526pairs14845people; exactTESTidentity/labels,
finitepredictions/positivescales/hashes verified. Threeof5complete,
1101448rows44339people. Equivalentfitreuse explicit;folds3-4pending.
No scoring or runtime/promotion change.
Outerfold1 COMPLETE:356944pairs14735people, exactTESTidentities/labels,
positive scales/finite predictions and hashes verified. Twoof5folds complete,
737922rows29494people;folds2-4pending. No scores interpreted or runtimechange.
Outerfold0COMPLETE20260924:380978pairs14759people;exacttestidentities/labels,
finitepredictions/hashverified. Fourfoldspending,noscoreinterpretationyet.
Fixed47targetevaluationprotocol precedes outcomes;no promotion/runtimechange.
COMPLETE20260924:80/80states(64executed16equivalent);all10checkpoint
fit-only/RNG/median/finite/hashchecks pass. Fold4loss.6539388/.7108092,
243475uniquepairs. Outerassessment/calibration/promotionpending;runtimeunchanged.
Latest20260924:fold3both8epochscomplete,64/80states(48trained+16reused).
230194uniquefitpairs,fitloss.5549454/.6073397,allchecks pass. Fold4both
andouterassessmentpending;no promotion/runtimechange.
Latest56/80states:40trained+16certifiedfold2reusefrom0(exactfit/scalers/
ordered8schedules/baselines/RNG). Fold3realcomplete8epochs,403072updates,
230194uniquepairs,.5549454fitloss. No outerevaluation/promotion.
Latestfold1both8epochscomplete:403072updates/arm236632uniquepairs,
fitloss.5469040/.5939681;checks pass.32/80states,no outerevaluation.
Identityaudit corrected newfold0/1equivalence prose:15differentfitpeople
each,scalersdiffer;independenttrainingperformed,noinvalidreuse.
Latestfold0both8epochsCOMPLETE:403072updates each,236509uniquepairs;
fitloss.5558083/.6010830;matchedRNG/seen/medians andfinite/hashchecks pass.
16/80stages,no outerevaluation/promotion. Earlier epoch1progress historical.
Fold0real/process epoch1complete,50384updates each47targets;43132uniqueof
698766fitpairs. Same RNG/seenpairs/baselines,allfinite,onlyfit.2/80stages;
fitlossnotgeneralization. No outerevaluation/promotion. See
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_TRAINING_PROGRESS_20260923.md` and fixed protocol.

Additional-domain full history cache COMPLETE (2026-09-23):
105sites/1800214examples/73670people;20.7h originalprocess no restart.
4.92GiB;fivefolds no unsupportedobservedinputs;independent fullidentity/label
check matches original centers;SHA+SQLiteintegrity valid. No training/promotion.
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_FULL_CACHE_20260923.md` and
`artifacts/native_reconstruction_additional_full_cache_20260922.json`.

Additional-domain fit-only scaling COMPLETE (2026-09-22):
Lossless SQLite pilot follow-up:847exacttensorroundtrips,2.15MBdb/12.77MBraw;
2cachetests pass. No precision/historyloss;fullcache/trainingpending.
`artifacts/native_reconstruction_additional_tensor_cache_20260922.json`.
Tensor follow-up: fixed prior pilot847examples37acceptedpeople;4235fold
transforms verified,rawlabels/masks match,no unsupportedinputsinpilot.
12.77MBpayload;25.29GiBillustrativefullextrapolationNOTpopulationestimate.
See `docs/research/notes/RECONSTRUCTION_ADDITIONAL_TENSORS_20260922.md`. No training/promotion.
Follow-up12randommixed-source streams match acceptedlabelvalues in full
reference builder;9combinedtests pass;allupstream/scaling bindingsverified.
All1800214accepted raw labels recovered;five patient-equal fit-only scalers,
heldout mutation invariant, six unsupported channels. Five tests. New SQL
conflict extraction defect fixed before use;no historical model attribution.
No training/runtime/promotion. See `docs/research/notes/RECONSTRUCTION_ADDITIONAL_SCALING_20260922.md`
and `artifacts/native_reconstruction_additional_domain_scaling_20260922.json`.

Additional-domain partition/support COMPLETE (2026-09-22):
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_PARTITIONS_20260922.md`. Outcome-blindGroupKFold
fullhistorycomponents,all73944people5outerfolds;distinctfit/selection/cal/test.
36/47targets>=30people3componentsallroles/folds,11failfeasibilitynotaccuracy.
Calibration7-8components vsold1;role0/1trainingidentityoverlapexplicit. Three
tests;allacceptancehash/countchecks. No model/normalizer/calibratortraining,
runtime orpromotion. Nextfit-onlyschema/scaling+exacthistorypreparation.
`artifacts/native_reconstruction_additional_domain_partitions_20260922.json`.

Additional FULL mask-support replay COMPLETE (2026-09-22):
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_MASK_FULL_20260922.md`.1810110frozenanchors,
1800214accepted/5467labelrejected/4429emptyindependenthistory;73670people47
targets.46targets>=6components>=30people. All105shards/hashbindingsandoriginal
anchorcolumnsverifiedexact. Prior850referencepilotnotfullreferenceencoding.
13.6h originalprocesscompleteddespitetoolhandleloss,no restart. DATA ACCEPTANCE
only,notaccuracy/newpromotion. Selectedcohortlimitunchanged. No roles/training/
runtime;nextsource/cal/testcomponentsupportandfit-onlyvocabulary.
`artifacts/native_reconstruction_additional_masked_history_full_20260921.json`.

Additional mask-support parity PILOT COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_MASK_PARITY_20260921.md`. Onehash-fixedpatientper
38components850anchors;847accepted/3rejected,allreferenceacceptanceand
variable/eventcountsexact. Three testsinclrandomizedsourceversion/quality/
sibling/windowparity. Complete1810110anchorfullrunstillpending;no population
acceptance/efficacyclaim. No roles/training/runtime/promotion.
`artifacts/native_reconstruction_additional_masked_history_pilot_20260921.json`.

Additional candidate anchors COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_ANCHORS_20260921.md`. Frozenvalue-independenthash
sampleonepatient-target acrossallstays/sites:73812people1810110pairs47targets.
Notfresh-conflict/masked-historyaccepted;no redrawlater. Fourtests incllocal
minima/globalequivalence andemptyinput;hash/identityverified. No roles,training,
runtime orpromotion. Next exactavailability/sourceversion replay+target/sibling
maskacceptance,retainingrejectedanchorsindenominator.
`artifacts/native_reconstruction_additional_domain_anchors_20260921.json`.

Additional selected-cohort preflight COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_COHORT_SUPPORT_20260921.md`. Full73944people;
rawinsulin127992rows5394people vsacceptedunits/hr32rows15people,ratechanges5.
ml/hr127565rowscannotbeconvertedwithoutverifiedconcentration. No targethas
30people/componentunderacceptedrateflag. Oldlineageconditionsonpotassium
future-pairavailability+ratechanges+MAP;notclinical/preanchorphysiologysubtype.
This newdomainpooldoesNOTreplaceoldselectedcalibrationcohort. Retrospective
privateflagsneverinputs. Fourtests;nomodel/runtime/promotion. Fullmaskedhistory
supportstillpending;continuebroadpreparationwithoutrelabelingpopulation.
Reports `artifacts/native_reconstruction_additional_insulin_support_20260921.json`
and `artifacts/native_reconstruction_additional_insulin_units_20260921.json`.

Additional-domain bedside extraction COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_DOMAIN_BEDSIDE_20260921.md`. Same existing canonical
nursing identity/entrytime/units/quality/freshness contract.70770people/5vitals,
27-28supportedcomponents each;labunion73812people47variables. Emptyhospital
helperfailure reproduced/repaired;16tests,105shardhashesverified. No clinical
cohort qualification,targetmaskedhistoryacceptance,roles,training or promotion.
Raw/privateeventsignored. Next qualifyhistories/cohorts beforepartition/fit.
`artifacts/native_reconstruction_additional_domain_bedside_20260921.json`.

Additional-domain lab extraction COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_DOMAIN_LABS_20260921.md`. Allcandidatepatients,
existingunits/bounds/identity/availability/freshconflictcontract.72817people
freshlabs42variables,41with6+supportedcomponents;creat/BUN/glucose~70kpeople
and30components>=30. No selectedcohortqualification,historyacceptance,roles
or efficacyyet. Three tests,no runtime/promotion. Privateshardsignored/hashbound.
`artifacts/native_reconstruction_additional_domain_labs_20260921.json`.

Additional domain inventory COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_ADDITIONAL_DOMAINS_20260921.md`. Outcome-blindfullrawhistory
connectedcomponentsexcludeallknownroles/ancestry/currentsource/confirmation.
73944candidatepatients105hospitals38components after12503transitivebridge
exclusionsbeyonddirectdisjointness. No roleassignment/extraction/accuracyclaim.
Three tests; privateprovenancehashbound; nextsamecontracttargetsupportaudit.
`artifacts/native_reconstruction_additional_development_domains_20260921.json`.

Calibration-domain feasibility COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_CALIBRATION_DOMAIN_SUPPORT_20260921.md`. Outcome-blindcomplete
patient/sitecomponents;preserveoutertest;enumerateall>=3cal/>=3fitpartitions.
Currentselectedtargets only3well-supportedcomponents,eachinnercal1component;
0/165prospective targetfoldsupportpasses. Notnewgateorfull-eICUdata-ceiling.
Four tests,no roles/model/runtime/promotionchanged. Additionaleligible
development-hospital extraction/provenance inventory needed before retraining.
`artifacts/native_reconstruction_calibration_domain_support_20260921.json`.

Calibration transport localization COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_CROSSFIT_CALIBRATION_TRANSFER_20260921.md`. All165foldtargets,
86supportedsource+outer,20oppositesigncorrections,38equalgroupharm. Source
offsetsexactlyreproduced;outeroracle explicitlydiagnostic/notdeployable.
Equivalentfold0/1samefunctiondifferentrequiredcorrections demonstrates
populationdependentbias. Five tests,no runtime/promotion;calibrationdomain
supportneedsfeasibilityauditbeforefullretraining.
`artifacts/native_reconstruction_crossfit_calibration_transfer_20260921.json`.

Independent source offset repair COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_CROSSFIT_OFFSET_CALIBRATION_20260921.md` andprotocol. All33x5,
encoder_tune-onlyequalgroupmedian;frozenencoder,testpatientsdisjoint.24/14/24
numericgainsall/selected/broad,butmacroworseallscopesand0incrementalscreens.
Creatininepatientmeanbetter/componentsselectedworse;albumin/proteinregress.
Three tests,no weights/runtime/promotion. Do not retrofitbesttargets/shrinkage.
`artifacts/native_reconstruction_crossfit_offset_calibration_20260921.json`.

Frozen creatinine numeric ablation COMPLETE (2026-09-21):
`RECONSTRUCTION_CROSSFIT_NUMERIC_ABLATION_20260921.md` andprotocol. All10192outer
patients/5folds,exactoriginalreproduction; bicarb/K/Hgbmedianreplacement each
worsensoverallbothgroups. Inadverse156personcomponentKremovalhelpsmorethan
bicarbbutstilllosescontrols;simplemarginalcorrelationexplanationinsufficient.
Two tests,no sourcefit/featureselection/promotion/runtimechange. NumericOOD
sensitivityonly,notcausaloruntouchedtest. Private predictions hashbound.
`artifacts/native_reconstruction_crossfit_numeric_ablation_20260921.json`.

Exact outer variable relations COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_CROSSFIT_VARIABLE_RELATIONS_20260921.md`. All33targets247610
fixedouterexamples, samefoldmaskedinputs, inversefoldscale. Per-inputsupport,
age,rho bygroup/component; no selection/retraining. Creatininebicarbrelation
differsdespitesameage, potassiumstable; phosphate supportthin. Glucoseage
differencesvariable-specific. Exploratorynotcausal/modelrelianceevidence.
Three tests, no runtime/promotion; target/siblingleakageassertionspass.
`artifacts/native_reconstruction_crossfit_variable_relations_20260921.json`.

Exact outer support audit COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_CROSSFIT_OBSERVATION_SUPPORT_20260921.md`. Replay all247610
eligible same33targetexamples, fold masks/source hashes/labels verified.
Anchor-relative measurement age excludesmissingsentinel/preservesobservedzero.
Glucose selected lessfresh despite longerhistory; creatinine large adverse
component has similar summarysupport and labelmean, hence no universal agefix.
Three tests. No training/selection/runtime/promotion or causal attribution.
Aggregate `artifacts/native_reconstruction_crossfit_observation_support_20260921.json`.

Crossfit group-shift diagnosis COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_CROSSFIT_GROUP_SHIFT_20260921.md`. Fixed all33 outer predictions,
parent/hash verification; within-component vs composition exact decomposition.
Totalprotein selected component-harm sensitive to tiny groups; creatinine and
glucose also regress in larger groups. No universal smallN/composition fix.
Three tests; no retraining, exclusion, altered gate, runtime or promotion.
Aggregate `artifacts/native_reconstruction_crossfit_group_shift_20260921.json`.
Next matched-example observation-support audit, not posthoc subgroup routing.

Fresh encoder crossfit COMPLETE (2026-09-21):
`docs/research/findings/RECONSTRUCTION_ENCODER_CROSSFIT_FINDINGS_20260921.md`. Fivefold2arm8epochs,
all247611outerpairs/247610eligible,4all/broaddescriptivescreens0selected/both.
Realmacro1.673005vs1.748309process/1.726672population. Realnumbershelpsome
maskedcurrenttargets,but heterogeneousgains remain;notfutureforecast/promotion.
72executed8reusedepochs,15tests. Proteinpanelmutualinputs explicit caveat;
no matchedwarmstartattribution or newexternalclaim. Fixedpredictionsprivate.

Encoder crossfit FOLD3 TRAINING COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Botharms8epochs283008
updates each,6299fitpeople,finalsampledlossonly. Integrity/RNG/hashchecks pass.
64/80statesavailable(56executed8reused),fold4andouterevaluationpending.
Fixedepoch8notbesttrainingepoch; no promotion/runtime/heldoutclaim.

Encoder crossfit FOLD2 TRAINING COMPLETE (2026-09-21):
`docs/research/notes/RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Real/processboth8epochs,
283008updates each,5513fitpeople. Fitloss.491907/.545704only;nooutermetrics.
Integritychecks pass.48/80statesavailable (40executed8reused),folds3-4pending.
No runtime/promotion change; numericalfitgainisnottransferproof.

Encoder crossfit FOLD1/EQUIVALENCE COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Independentrerunreal
8epochs exactmodel/optimizer/RNG/logmatchfold0, sameorderedfit/scalers.
Processcontrolreusedonlyafterexactevidence; explicit0newupdates.32/80states
available (24executed8reused), folds2-4pending. Two tests/noefficacyclaim.

Encoder crossfit FOLD0 TRAINING COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Real/processboth8epochs,
283008updates each,finaltrainingloss.922825/.999242,noouterassessment.
16/80armepochscomplete;remaining4foldsunchangedprotocol. Integrity/RNG/finite
checks pass. No promotion or runtime change; do not call traininggain transfer.

Independent encoder crossfit STAGE1 (2026-09-20):
`RECONSTRUCTION_ENCODER_CROSSFIT_TRAINING_PROTOCOL_20260920.md` and
`docs/research/notes/RECONSTRUCTION_ENCODER_CROSSFIT_PROGRESS_20260920.md`. Freshreal/process,
all33targets;fold0epoch1both35,376updatescomplete. Fivefold2arm8epochscope
unchanged (2/80complete). Noouteraccuracyresult orpromotion.11tests pass.
UnlikecachedheadCV, encoder learns onlyfoldfitinputs, nolearnedreleaseweights.

Encoder fold inputs COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_ENCODER_FOLD_INPUTS_20260920.md`. Full10737patientfit-only
scalers/roundtrip;sourceunsupportedmyoglobin/aniongapisolatedwithfixedsource
vocabularymask,notquery-derivedselection. Trainedpairs247610/247611eligible;
oneemptyhistoryand404nontrainedunsupportedtargetsreportednotdiscarded.
Nine tests/no modeltraining/runtime/promotion. Matchedencodertraining remains.

Encoder crossfit preflight COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_ENCODER_CROSSFIT_PREFLIGHT_20260920.md`.10737formerfitpeople,
14sites/8whole-historycomponents;44multisitepatients. Five safe outerfolds,
innerselection1-2sites;31/33patient-support, fibrinogen/troponinexceptions.
No encoder trained or accuracy claim. Three tests, no runtime/promotion.
Next fullencoder comparison must address inherited preprocessing/weights and
thin-domain epochselection, not reuse cachedstates asindependent outerinputs.

Robust raw/body matched probe COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_ROBUST_RAW_BODY_PROBE_20260920.md`. All33x3SmoothL1same64/
regularization;99converged,no warnings. Macrobody1.851/1.574raw1.901/1.611;
raw0passes,14/13numericwins. BUNpatientadvantagebutsite-equalharmboth,
glucose/PTTbodybetter. Two tests/no runtime/promotion; lossconfoundnotsolecause.

Probe tail/objective COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_PROBE_TAIL_OBJECTIVE_20260920.md`. Current33probe,top2targets
77.92%/80.40%ofpositiveridgeregression; CPKtail98.60%squaredlossshare at1.507%
weight. Sourcefixedmean/mediancontrols,PTTcounterexample;no labeldeletion/
headchanges. Two tests,no promotion; lossconfoundsinformationprobe,notsolecause.

Raw-probe scaling warning COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_RAW_PROBE_SCALING_20260920.md`.32warnings/288negativevariance
entries allconstant,min-9.7e-35;finiteoriginalscales. Full33x3repairreplay
maxprediction1.44e-11/MAE2.03e-13. Two tests,no runtime/promotion; numerical
cleanupnotgeneralizationcause. Earlierraw/bodyconclusionretained.

Current raw/body matched64 probe COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_CURRENT_RAW_BODY_PROBE_20260920.md`.33targets/fullsource,
currentbounded7notoldencoder;source-onlyPCAraw/process64/body64,ridge100.
Raw0gatepasses,14/33numericbodywinspergroup. Glucose/PTTbodybetter;BUNsmall
rawgain. Two tests/no runtime; PCA/order/objectivelimits,scalerwarningsnot
fullylocalized,allMAEsfinite. Newprivate rawcacheexactjoined tobodyfeatures.

Exact MAE site readout COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_MAE_SITE_READOUT_20260920.md`.33matchedLPpairs, samebody,
fixed.1MAE/L1 plusoptionalexactsiteMAEconstraints.66success,17realchanges,
16nearoriginal. Macro+.36%/.44%gain but0incrementalpassesbothrefs. PTTstill
patientmeanharms;BUNgrouptradeoff. Two tests/no runtime. Metricmismatchnow
tested/repairedincandidate, notcompletecause; avoidrepeatingheadpenaltysweeps.

Site-constraint metric reconciliation COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_SITE_CONSTRAINT_METRICS_20260920.md`. Exactfrozen33heads,
692siteconstraintsallSmoothL1passbut94MAEharms;45/66groupsaffected. Fiveheld
harmgroupswithoutanysourceMAEharmincludingPTTselected8/8sourcegains. Metric
mismatchpartialnotsolecause. Two tests,no model/runtimechanges; exactMAELP
andmatchedcontrol remainunrun,scipyavailablewithoutnewdependency.

Source-site constrained readout COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_SITE_CONSTRAINED_READOUT_20260920.md`. Fixedstrength1matched
controls,all33currentbodylinearheads,exactsourcegroup/siteSmoothL1constraints.
Allsolvessucceed;aggregatebothgain,butincrementbeyondmatchedonly.05%/.10%;
21numericgainsbothvsoriginal,0fullincrementalpasses. PTTstillharmsheldsites.
Two tests/no runtime/promotion; not oldrobustloss or bodygradient experiment.

Readout latent support COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_READOUT_LATENT_SUPPORT_20260920.md`. Frozen33heads/features,
pergroupfit-only99%LedoitWolf distance.14/17mean-regressinggroups insideharm
atN>=30; broadglucose3084/PTT1317. Parent patientmeans exact. Ellipsoidnot
calibratedsupportoridenticalstates;no distance-onlyrepair established. Two
tests,no fittingofpredictor/runtime/promotion; no outcomechosennewthreshold.

Conditional nested readout CV COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_NESTED_READOUT_CV_20260920.md`. All33, same fixedencoder/anchor,
5outer/4innerhospitalfolds, inner-only penalty choice/transforms. Mean delta
-.022519ordinary->-.006126nested;14gains17harms2unchanged. Notpureselectionbias
estimate(smallerinnerselectionset);notencoder-independent. Glucose/PTT still
internalgain/external-sourceharm. Two tests/no runtime/promotion. Old0917
independentheadconfirmation already done; wholeencoder crossfit remainsdistinct.

Readout CV transfer COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_READOUT_CV_TRANSFER_20260920.md`. Fixed existing choices,
matched group/site weighting,24 changed/9 unchanged;18 mean gains but predicted
-.030964 vs realized-.001430, rank.224. Six mean negative transfers,12 numeric
two-group gains,zero common gates. Encoder/anchor exposed to inner CV sites;
cannot isolate selection optimism from shift. One test,no new fitting/runtime.
Next: encoder-independent selection, not another heldout-selected head sweep.

Anchored readout COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_ANCHORED_READOUT_20260920.md`. Sourcefit5sitefolds,
fixed4penaltiesplusunchanged,currentSmoothL1samecapacity. Mostregression
recovered,broad+.20%gain,selected1.10%harm;0same-targetgatesboth.
Two tests,no runtime/promotion. No furtherheldoutpenaltytuning.

Frozen readout SVD COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_FROZEN_READOUT_SPECTRUM_20260920.md`. Fitonlybottom16,
.655%medianvariance/70.95%newweightnorm; ablationrecoversminorityregression,
20/33numericgainsbothbutnoaggregateoriginalrecovery. One test,no promotion;
do nottune rankfromselectionoutcomes.

Current frozen-body readout refit COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_FROZEN_READOUT_REFIT_20260920.md`. Samecapacity33linearheads,
currentbounded7featuresfullsource,weightedSmoothL1. Fit-5.41%, bothsource
selectionmacroregress,0incrementalpasses. Onlyheadchanged,no runtime/promotion.
Two tests,fullfeaturescached; not oldencoder ridgeprobe or proofinformationabsence.

Bias-polish source selection COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BIAS_POLISH_SELECTION_20260920.md`. Full3413source-selection
people77219pairs, all33. Bothsourcegroupsgateintersection0; candidateexact
original. Three tests,no runtime/promotion. Rejectuniversalbiasrepair; not
claimsignalabsenceoranythingaboutfutureforecasting.

Bias-polish site diagnosis COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BIAS_POLISH_SITES_20260920.md`. Existinghelperexactpairedreplay;
BUNselectedLOOallimprove butsiteevidencefails; bicarbbroad5/7harm,actual
weightingreversal. No uniformlowpowerexplanation,no gatechanges/promotion.

All33 bias polish COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BIAS_POLISH_20260920.md`. Fullsource247611pairs,exactscalar
SmoothL1biases only. Evaluationmacroimprovesboth,12/33numericgainsboth;
zero same-targetincrementalpatient/sitepassesboth. BUNselectedaveragegain
butsitefail; bicarbbroadsitefail. Three tests,privatecandidate,no promotion.

BUN bias stationarity COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_BIAS_STATIONARITY_20260920.md`. ExactscalarSmoothL1optimum
underexpectedsourceweights improvesMAEandtrainingloss; originalgradient.194.
Broadselectiongain7.28%,selectednotstable. Availableindependentbiasdirection,
notglobalconvergenceclaim. Two tests,no runtime/promotion; alltargetpolishnext.

BUN signed site residual COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_SITE_RESIDUAL_20260920.md`. Exactsite/signedcachejoin,
broad4/4negativeCI; selected71/90onesitewideCI. Reviewdistinguishesinsulin
cohortinclusionfromrawavailabilityanchors. No hospital-onlycause or validated
regimeclaim; no runtime/split/promotion, cachedmedian tests reused.

BUN conditional residual COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_CONDITIONAL_RESIDUAL_20260920.md`. Fixedfitquartilebins,
broadallnegative medians; selectedbins11/12/19/48insufficientevidenceof
conditionalshift. Compositiondifferencehighestbin53.3vs28.2%. Two tests,
no new correction/runtime/promotion; inspectsourcecontract/supportnext.

BUN affine transfer COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_AFFINE_TRANSFER_20260920.md`. Source-fit medianlinear
calibration, crossapply90/3086. Slopesnear1, no commonincrementalbenefit over
offset. FullSpearman.440/.384; marginalcompressionnotproofamplitudebug.
Two tests,no runtime/promotion; no outcome-selectedcalibrator.

BUN source-fit offset transfer COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_OFFSET_TRANSFER_20260920.md`. Full/bodyoffmedianoffsets
fit529/9664, crossapply90/3086, identitiesdisjointbaselineexact. Broadfull
gain7.94/8.21%, selectednotstable; calibratingboth retainsoppositebodyoff
effects. Offsetispartialnotuniversalrepair. Two tests,no runtime/promotion.

BUN integration-resolution COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_INTEGRATION_RESOLUTION_20260920.md`. Fixedweights90/3086,
steps1/.5/.25, exactbaseline. Broad MAEgain0.0285%, selected CIcrosszero;
too small to explain body-off4.751%penalty. Substep enters learnedfield,
not pure numericalsolver test. Two tests, no fitting/runtime/promotion.

Protected dynamics finite-step COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_DYNAMICS_FINITE_STEP_20260920.md`. Actualfixed.01bodyparameter
updates,2protectedandmatchednormcontrols,all33source-selectiontargets. Baseline
exact; BUNsmallmixedchanges,protectedmeanharms12-16/33. No robustnonharm or
promotion; fourtests. Distinctfrompriorgradient-onlybutnotfulltrainingclaim.

Protected body-dynamics descent probe COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_DYNAMICS_DESCENT_20260920.md`. Notoldbodyupdategate; implicated
bodydynamics gradients,all33x2sourcegroups,twodisjointup-to64probes. Feasible
projectionsretainBUNdescentbutotherprobe20/27negativegroup-targetdots. No finite
step/Adam/nonharmgeneralization claim. Three tests,no runtime/promotion.

BUN weighted-objective reconciliation COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_WEIGHTED_OBJECTIVE_20260920.md`. Cachedfitpairs,exactfinal
samplerRNG/seen sets,3752selectedepochdraws/group. MAE/SmoothL1bodyoffgains
persist under expected and realized weights. Not historical per-step losses,
not fullobjective or convergenceproof. No training/runtime/promotion.

BUN fit/exposure loss diagnosis COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_DYNAMICS_FIT_20260920.md`. Full529/9664fitpairs; exactsampler
allcounts/finalRNG/sets. Selected7BUNseen527/2934. Broadseenno-body improves
MAEandactualSmoothL1; not merely externaldomainshift or alternativeMAEmetric.
Other-targettradeoffs persist; not optimizerconvergence/gradientconflictproof.
Four tests,no modelupdate/runtime/promotion; privatepredictionsnotcommitted.

All33body/organ branch diagnostic COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_ALL_DYNAMICS_BRANCHES_20260920.md`. Originalsource90+90,
2470/2025pairs, fullperpatienterrors reproduce. Bodyoffharmsmacro both,
24/33selected21/33broad; zero same-targetnominalCIgainsboth. Organoffmixed.
Universalnodynamicsrepair rejected; usefulandharmfulstateevolutioncoexist.
Four tests,no runtime/promotion. No furtherglobaldeletion asuntestedfix.

BUN body/organ branch ablation COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_DYNAMICS_BRANCHES_20260920.md`. Fixed90/3086sourcepairs,
normalfullscoreexact. Broadbodyoff4.751%gain,organoff1.492%;selectednotimproved.
Bodyoffblocksorgan-to-bodytransitionroute,equalsbothoff;notorganirrelevance.
No-dynamicsalgorithmnumericaldifference1.192e-7 recorded,notstatisticalrelaxation.
Two tests,no fitting/runtime/promotion. Next all33branchdiagnostic,notglobaldelete.

BUN history/dynamics factorial COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_HISTORY_DYNAMICS_20260920.md`. Fixedstate,90/3086source,
originalclock/updategap/fullpatienterrorreproduction. No dynamics retainsall
measurementsandimprovesbroad4.751%;selectednotimproved. Furtherolderupdate
deletionintervalcrosszero. Strongerlocalizationtodynamics,notnewtrainedrepair.
One focusedtest,no runtime/promotion. Bodyversusorgan evolutionnext,notblind
historytruncation or claiming allJEPAfuturefailures explained.

BUN site/support localization COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_HISTORY_STRATA_20260920.md`. Exactcachedpairs andresolved
sites; fixedpreanchorcount/span/agebins. Broad4/4sitemeansfavor latestcoverage,
3nominalCIs; selectedadequateonlyonesiteCIcrosszero. Same449+33events opposite
means, notdefinitiveinteractionproof. Longhistorybroadbenefitnotonesiteartifact.
Three tests,no learnedgate/runtime/promotion. Timepath/content distinctionnext.

Full BUN latest-coverage replication COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BUN_FULL_HISTORY_20260920.md`. Fixedbounded7,90/3086source
patients, originalper-patienterrors/countsexact. Latestcoverage reduces broad
MAE4.006%(additional3003replicate), selectedharms4.711%. Last8union same signs.
Not newexternal validation or universal cutoff; no fitting/runtime/promotion.
Five tests; next source-site/support distinction, not immediate oracle routing.

Latest-variable coverage control COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_LATEST_COVERAGE_20260920.md`. Same originalsource33targets,
2470/2025pairs; preserve latest observation per independent variable before
removing repeats. Full/last1 MAEs exact. Latest beats last1 aggregates both,
but vsfull selectedharm/broadgain; BUN reversal,creatinine intervals crosszero.
Seven tests; not causal memoryproof or fullsource confirmation. No modelupdate,
runtime/promotion. Latest coverage must not be confused with repeatedhistory.

History suffix diagnostic COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_HISTORY_SUFFIX_20260920.md`. Original90+90source-selection,
33targets,2470/2025pairs; full MAEs exact. Last1 aggregate harms both; last8
mixed. Latest-variable support loss confounded with temporal history removal.
Four tests; not full3323 replay, no fitting/runtime/promotion. Next distinct
control preserves latest event per independent variable before deleting repeats.

Full-source final creatinine update COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_CREATININE_FULL_UPDATE_20260920.md`.90/3095sourceencoderpairs,
exactoriginalscores/forwardreplay. Broad update-.07537CInegative;selected
-.01769CIcrosszero. Original83alsoimprove:small8probeharmhypothesisrejected.
No >=30broadlast-variablepresentstratummeanharms;notcausal/multiple-testgate.
No update removed, no fitting/runtime/promotion. Remaininggeneralizationscope
unchanged;do notrepeat smallprobe forgetting claim.

Final-update trace COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_FINAL_UPDATE_TRACE_20260920.md`.525fixedsourceexamples,33targets,
separate finaladvance/update/gap,exactfinalreproduction. Updatehelpsmacro both,
creatinine8+8probeharm,othersmixed. Smallretrospectivepairedtrace,notcausality/
forecast/promotion. Tests gap and maskedlast-event paths;no stateupdate removed.

Full gradient paths COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_GRADIENT_PATHS_20260920.md`. Fixedbounded7,66sourceprobes,
99connected/2245trainabletensors;body/organstatepaths andmixers nonzero.
Unusedfuture/uncertaintybranches expected;no deadsharedstate rootcause evidence.
No optimizerupdate/evaluation/architecture/runtimechange or promotion.

Bounded site evaluation COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BOUNDED_SITE_EVALUATION_20260920.md`. Fixedepoch7both,identical
83874/4637pairs;11bindings/referenceMAEs verified. Broadmacro+.266%harm,
selected-.316%gain;creatbothbetter,BUN/lipasebothworse.0correctedpairedpatient/
site passes and0allcomparisons; newprocess/placeboabsent,nopromotion. This
specific site-probability intervention tested,notuniversalsolution.33+6scope.

Bounded site matched training COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BOUNDED_SITE_TRAINING_20260920.md`. Uniformreplacement vs [.5,2]
bounded source probabilities,8epochs283008updates each,commoninit/schedules.
Bothselect7,source macro-1.14%,20/33numericgains pergroup;renaltradeoff persists.
Seven tests, no heldout evaluation yet/no runtime or promotion.33trained6pending.

Bounded site sampler preflight COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_BOUNDED_SITE_PREFLIGHT_20260920.md`, fixedprotocol. Purefit
membership weightprojection [.5,2]mean1,5tests. MedianTV.139/.276,minESSfraction
.672/.677; no labels/modeltraining. Expected capnotrealizedrepeatcap. Matched
uniformreplacement training stillpending; do not claim accuracyrepair/promotion.

Two-group site preflight COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_SITE_BALANCE_PREFLIGHT_20260920.md`. Selected3ambiguouspeople98
exactunique rawstaymatches; complete335780sample-site mappings withshard/role
checks. Equal-sitepatient upweight up to48.73selected/80.27broad(untraineddirect
bili),trainedtrig52.39. Sampling-risk calculation, not efficacy/harm proof.
Six tests; no fitting/runtime. Boundedsource weighting required, not naive
equal-site replacement. Sixpendingheads remain explicit.

Source sample-site recovery COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_SAMPLE_SITE_RECOVERY_20260920.md`. All45ambiguous broadpeople
1286acceptedpairs regenerated exactly;4legacy people require shared-RNG full
ordering replay. All45actually have targets acrosssites, no representative-site
assignment. Private sidecar plus13479singleton mappings, oldtensorsunchanged.
Fourtests; no training/runtime/promotion. Not evidence of mixed-stay physiology
or accuracy gain; selected-group provenance preflight remains separate.

Source site alignment COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_SOURCE_SITE_ALIGNMENT_20260920.md`. Source-only hashes/roles/
fixedepoch7/16pairs checked; tune4sites with50.44%inlargest unambiguous site.
Patient-vs-site weighting signflips3/33, notcreat/lipase; insufficient solecause.
45source multi-sitepatients not assignable pertarget from cachedprovenance;
diagnostic exclusions explicit. Five combinedtests, no fit/selection/runtime.
Exact target-anchor stay/site recovery remains necessary before full reweighting.

Continued candidate site diagnosis COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_CONTINUED_SITE_TRANSFER_20260920.md`. Existing patient/site
statistics reproduce1e-10, unique pairs/hash bindings verified. Added site
weight sensitivity/LOO/minimum support. Broad-vs-old32/33favorable means but
22site-evidence failures; vs previous expanded10pooled mean failures/all33
site-evidence failures. Eight broad incremental weighting sign reversals;
not causal site-shift proof or gate relaxation. Three tests, no fitting,
runtime/interval/promotion changes. Source site-alignment preflight next.

Continued expanded-source replay + paired uncertainty COMPLETE (2026-09-20):
`docs/research/results/RECONSTRUCTION_CONTINUED_SELECTION_RESULTS_20260920.md`. Allscopesselect16,
full0-19trajectory; originalscoresreproduce. Creat/lipase source signreversal
butold/newCIs crosszero, notprovedharm/repair.4000pairedpatientdraws,unadjusted
postselection. Glucose/BUNsourcegainsmoreconsistent;no fullpromotion/runtime.
No selector-onlyrescue;33trained6pending retained.

Continuation source/evaluation localization COMPLETE (2026-09-20):
`docs/research/notes/RECONSTRUCTION_CONTINUATION_TRANSFER_20260920.md`. Sourcecreatbothimprove
slightly7->16,evaluationbothharm; lipasesame.15/33broad12/33selected sign
disagreements (not significance). Notaggregate-sacrifice-only explanation;
sampling/shift unisolated.2tests,no fitting/runtime/promotion. Next expanded
source replay9-19, preserving oldroles and full39scope.

Complete-exposure continuation COMPLETE (2026-09-20):
`docs/research/results/RECONSTRUCTION_COVERAGE_CONTINUATION_RESULTS_20260920.md`. Matched3arms8->19,
fullfinalpoolcoverage, selection16/6/18. Bothreusedcohorts macroimprove.350%/
.619%,23/33and25/33numericgains. BroadcoreAST/bilirubin/calcium;0fullincremental
both. Glucosebothgain,creatininebothharm. Selected16notfullcoverage;7tests,
oldlogs/hashchecks. No runtime/interval/future/promotion.39scope retained.

Expanded-source epoch replay COMPLETE (2026-09-19):
`docs/research/results/RECONSTRUCTION_EXPANDED_SELECTION_RESULTS_20260919.md`. Ninefrozenepochs on
3323sourcepatients; original90allscoresexact, additional3233separate. Equalgroup
best8vs7tiny.0312%gain; fullbroad31/33beatprior numerically, notpromotion.
Glucosecohortconclusionchanges;16/33harm8vs7inclrenal. Not soleNcause or
complete accuracyrepair. Two tests; no evalrefitting/runtime/interval/promotion.

Complete-source scope reconciliation COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_COMPLETE_SOURCE_SCOPE_20260919.md`. Actualexposure vsall39
shards; dense42-45%finalcoverage, sixpendingallhavesource support. Smallselected
groups stillinadequate forsome; expandedtune3323disjointsourcepatients verified.
Oldtune sparsecounts6-18 couldlimitselectionreliability,not provenerrorcause.
Two tests; no fitting/runtime/promotion. Source-only savedepoch replay next.

Score-versus-gradient attribution COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_SCORE_GRADIENT_20260919.md`. Saved8320parameter body-gate
norm/cosine matrices reproduce everymean-gradient dot. Fivehigh-scoretargets
51.19%MAE but22.12%/20.65%normmass; no majority-local-domination support.
Two8patient probes only, not fullencoder/Adam or rootcauseproof. Six tests,
bindingsverified; no newfitting/runtime/interval/promotion.

Source score concentration COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_SELECTION_CONCENTRATION_20260919.md`. Sixreal source
trajectories, allrecordedmacro/selectedepochs reproduce. Fiveof33targets carry
~50-52%score, but5/6arms unchanged byanysingle target removal; relativeerror
stillselectsexpanded7/moment5. Descriptive selection diagnosis, not gradient
dominance, newselector or repair. Five tests; no eval fitting/runtime/promotion.

Moment-clock sensitivity COMPLETE (2026-09-19):
`docs/research/results/RECONSTRUCTION_MOMENT_CLOCK_RESULTS_20260919.md`.8epochs, betas raisedto528,
otherwise originalaccumulation unchanged. Source selection improves slightly
(epoch5,1.816181), reusedbroad/selected macroerror worsens.517%/1.177%.
13/33numericimprovements each,0reportedcomparisonpasses. Glucose improvesboth;
BUN/creat selectedgain/broadharm. Not aggregate repair/unique rootcause;
33trained6pending, no runtime/interval/promotion. See moment_clock_source and
moment_clock_evaluation JSON artifacts. Supersedes earlier untested-clock notes.

Decay-only accumulated training/evaluation COMPLETE (2026-09-19):
`docs/research/results/RECONSTRUCTION_DECAY_RESULTS_20260919.md`.wd.229722vs.01,allsameothertraining
settings. Bothselected7; broad+.075%,selected+.476%macroerror. Noaccuracyrepair,
no fullmatchedcontrols/external/future/promotion;runtimeunchanged. Do not repeat
as untested; momentclock and stepcount are distinctremainingconfounds.

Actual optimizer clock audit COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_OPTIMIZER_CLOCK_20260919.md`.Accumulation changes528xsampletime
momentmemoryand decay(.7535->.9878sqrtLR).Checkpointstepdistributionverified,
zero-grad AdamW decaymatchingtested;notfullupdatematching/failurecausality.
4tests,no fit/promotion/runtimechanges;decay-only controlledtrial remainspending.

Balanced patient gradient accumulation COMPLETE (2026-09-19):
`docs/research/results/RECONSTRUCTION_ACCUMULATION_RESULTS_20260919.md`.2rates8epochs,same283008draws,
536stepsversus283008. SourcesqrtLR7selected;glucoseimprovesboth,butBUNharmsboth.
Broadmacro+.73%,selected-.60%;0diagnosticpasses,matchedprocess/placebopending.
4tests,no runtime/promotion;steps/decay/LR confoundpurevarianceexplanation.

Trained-gradient patient-count diagnostic COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_GRADIENT_SAMPLE_SIZE_20260919.md`.8->64patients raisescommon63
median cosine.024->.372;14negative remain.0repeatcrossgroupnegativeat64among30
supported;3selectedsparse tasks excludedonlyat64. Onefrozenlocalgate/noaccuracy
or causalityclaim.4tests; gradientaccumulation remainsuntested onthissetup.

Trained shared update gradients COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_TRAINED_GRADIENTS_20260919.md`.1056sourceexamplegrads,8320bodygate
params,66tasks2probes.4repeatcrossgroupnegative;glucose/troponinnonnegative.
26/66same-taskcrossprobe negative,median.098. Local/nooptimizerintervention,
notuniversalconflictproof;2tests,no runtimechange.

Paired exposure checkpoint feasibility COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_EXPOSURE_SELECTION_20260919.md`.All66x9scores; neithercondition
has a differentcommoncheckpointdominatingselected. Expanded10transientgainslost,
2neverimproved;onlyepoch0allnonharmingprior. No newgate/per-targetoracles.
Historicalretentionloss tradeoffs reviewed;2tests,no runtimechange.

Expanded source paired fit/selection diagnosis COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_EXPOSURE_RETENTION_FIT_20260919.md`.132source scores reproduce.
Selectedglucose losesprioronfit andselection; earlyepoch2gainlost atselected7.
Other targets have differentfit/tune/eval patterns. Notuniversaloverfitting or
gradientconflictproof; no target-oracle selection, newfit or runtimechange.

Paired exposure prior/increment diagnostic COMPLETE (2026-09-19):
`docs/research/notes/RECONSTRUCTION_EXPOSURE_INCREMENT_20260919.md`. All66 frozen differences replay;
broadconditionalmacro-.043965,selected+.007640. Differentfailuretypes: selected
glucoseconditionalharmdespitebetterprior; broadBUNslighttotalharmdespiteconditional
gain. No uniformoffsetrepair/causalityclaim. No training/runtime/promotionchange.

Fixed-budget source exposure pair COMPLETE (2026-09-19):
`docs/research/results/RECONSTRUCTION_SOURCE_EXPOSURE_RESULTS_20260919.md`. Six8epoch fits; broad
536->10201fit, fixedselected536 and90+90encoder, samepersistentqueuealgorithm.
Broadmacro-3.68%vssmall;selected+.42%;26/33 and16/33numericgains. ALT/AST/BUN
broad originalcoregate passes;0with added small-model comparator,0selectedcore.
No promotion/newexternal/future/calibration;9tests,runtimeunchanged. All39status
retained33trained/6pending; full-poolcoveragepending. Do not repeat as untested.

Source loading/queue verification COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_SOURCE_STORE_20260918.md`. Expanded10201fit/234618pairs for33heads;
both original90patient encoder sets unchanged. Hash/identity/nested-role checks,
bounded16shard cache,7tests. Historical embedded tags are not current split roles;
initial loader mismatch fixed before fitting. Queue exposure=min(N,4288).
No new training/accuracy/runtime result; matched paired trainer remains pending.

Full-source samples COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_FULL_SOURCE_SAMPLES_20260918.md`.13524sourcepatients,
319730acceptedpairs,all39basicrole support. Oldmatched626reusedunchanged;
no evaluation role reuse/fitting/promotion. Next matchedexposure trainingpending.

Joint exposure audit COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_JOINT_EXPOSURE_20260918.md`.Exactsampler/RNG283008draws,
lipase broad93.22uses/pair vs glucose8.51. Supportvsrepetition distinction,
notcausalfailureproof. Fullsourceexpansionpending; no runtimechange.

Joint error geometry COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_JOINT_ERROR_GEOMETRY_20260918.md`.33targetsfit/selection,
selectedCPK/lipase ordering collapses,centralpatientsalsoharm;broadtroponin
rankingretained. SmallN/descriptive,notsolecause;allincrements reproduced,
no fitting/runtimechange or truth-baseddeploymentroute.

Joint fit gap COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_JOINT_FIT_GAP_20260918.md`.Exact66selection scores,32/33fit
gainsbothgroups. All9/8selectionfailures gainonfit:sourcegeneralizationgap,
not onlyexternaltransfer/optimizationfailure. No uniqueness/significanceclaim
or runtimechange; all33trained scopepreserved.

Joint constant objective diagnostic COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_CONSTANT_OBJECTIVE_20260918.md`.33sourcefit equalgroup centers,
Huber-minus-median encoderMAE-.010736/+.011093. Mixednotuniformrepair; cannot
infer conditional neural effects.3tests,allfit optimalitychecks,no runtimechange.

Retention feasibility COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_RETENTION_FEASIBILITY_20260918.md`.Onlyepoch0retains all66
sourcegroup-targetmeans;4selected/3broad neverimprove in8epochs,5each transient
improvementsnotretained. Necessary-condition diagnostic,notnewgate.3tests,
alltargetscope/runtime unchanged;trainingrepair remainsnecessary.

Joint selection trajectory COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_JOINT_SELECTION_20260918.md`. All9epoch aggregate scores replay,
epoch6selected correctly.9selected/8broad targets worse than initialprior already
on source selection. Target harm hidden by aggregate gains; no causalgradient
claim or newcheckpointselection. Runtime unchanged.

Joint gate localization COMPLETE (2026-09-18):
`docs/research/notes/RECONSTRUCTION_JOINT_GATE_DIAGNOSIS_20260918.md`.33/33fail>=1site evidence
comparison bothroles;meanerrorsalso fail9broad16selected. Coregate remains0
without broad-only extra reference.22/11supported favorablemeansallcore are
notpasses.3tests,no threshold/selection/runtimechanges.

Joint-source training/evaluation COMPLETE (2026-09-18):
`docs/research/results/RECONSTRUCTION_JOINT_SOURCE_RESULTS_20260918.md`.1072fit180encoder,three8epoch
arms6/1/0selected. Both-cohort macro improvesold,30/33broad19/33selectedtargets.
0fullpasses; selectedglucose/bicarb/O2stillnumeric harms. Average tradeoff
mitigated not universalrepair. Doublecompute/people,5tests,no runtimepromotion.

Joint-source preparation COMPLETE (2026-09-17):
`RECONSTRUCTION_JOINT_SOURCE_PROTOCOL_20260917.md`; artifact
`native_reconstruction_joint_source_manifest_20260917.json`.1072fit/180encoder,
disjoint sourcepatients, originalshards preserved, heldout exclusions verified.
Training/evaluation pending; no outcome, promotion or runtime change.

Broad-source retention COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_BROAD_RETENTION_20260917.md`.178old evaluation patients4637pairs,
fixed three checkpoints.28/33pooled numeric harms,macro+2.49% versus expanded
cohort-4.18%. Distribution tradeoff now measured; not every harm significant,
no causal attribution or clinical gate. No fitting, runtime or promotion.

Broad prior/increment diagnostic COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_PRIOR_INCREMENT_20260917.md`. Saved83874pairs exactreplay,
3decomposition tests. Creatinine/BUN improvements exceed center-error changes;
O2prior improves more than model;glucose still loses newcenter. Descriptive
error identity, not causal isolation, newselection or promotion. All39retained.

Matched broader-source training/evaluation COMPLETE (2026-09-17):
`docs/research/results/RECONSTRUCTION_MATCHED_BROAD_RESULTS_20260917.md`. Fixed536fit90encoder,
33targets141504updates/arm,real/process/placebo selected5/2/1. Confirmation
83874pairs3433patients:28pooled improvements,0complete corrected passes.
Macro1.365619->1.308580. Partial non-old-control gainsALT/AST/calcium/platelets.
Glucose old-bias reduction is not individual-signal validation. Evaluator
sharedcachedmodel repaired with independentcopies; no runtime/promotion.
Old-cohort retention, repeats and external gates incomplete.

Equal-patient source comparison samples COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_MATCHED_BROAD_SAMPLES_20260917.md`. Outcome-blind fixedhash
role/representative-site matching536fit90encoder;626shards,12441/2113pairs.
All39scope, no redraw,2tests. Target support not matched; no causal attribution
or accuracy claim. Original33target matchedtraining pending, fullbroadpool kept.

Broad development event extraction COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_BROAD_SOURCE_EVENTS_20260917.md`. No insulin inclusion rule,
fixed provenance roles:fit10201people/5721709events,encoder3323/2487270events.
Same canonical contracts as confirmation;15hospitalshards;14tests pass. Initial
COPY issue fixed after terminal failure, prior provenance retained. No samples,
retraining, accuracy or promotion claim. Equal-size comparison needed for attribution.

Inherited cohort restriction audit COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_COHORT_SELECTION_20260917.md`. Rawinsulin parser verifies100%old
query stays have qualifyingratechange, versus125/4131newstays,118/3433newpatients.
Code lineage potassium-rate->MAPquery->reconstruction retains selectedstaylist.
Support expansion also changes case selection. No solecause/treatmentcausalclaim,
no inference use of retrospectivecriteria. Broader source development repair
pending; existing evaluation/reserved excluded. No runtime/promotion change.

Expanded frozen nominee confirmation COMPLETE (2026-09-17):
`docs/research/results/RECONSTRUCTION_EXPANDED_CONFIRMATION_RESULTS_20260917.md`.83874pairs,
33originalheads/28nominees,0confirmed,all39catalog retained. BUN/bicarb/WBC
actualharm atlargeN;O2sat meanbenefit lacks multiplicity-correctedsite support.
No fit/tuning onnewlabels;6tests,accepted counts/identity verify. Expanded
population not case-mix-matched;7sitesunchanged. No forecast/runtime/promotion.

Expanded source-confirmation sample build COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_EXPANDED_CONFIRMATION_SAMPLES_20260917.md`.3433patients,
88232accepted of88728one-per-target anchors, no redraw. All39represented;
36>=3siteswith30patients. Conflict-aware available-time blockmasked histories,
9tests pass. No accuracy evaluation or training; fixed nominees pending larger
confirmation. Private hash-indexed shards, no runtime/interval/promotion changes.

Expanded source-confirmation extraction COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_EXPANDED_CONFIRMATION_EVENTS_20260917.md`.3433people/4131stays,
2154036canonical lab+nursing events. Five bedside targets no longer unscanned;
existing time/unit/identity/quality policies preserved.14tests pass, privatecache
Gitignored. Reconstruction sample acceptance/predictions stillpending, no fitting,
external-independence, accuracy or promotion claim. Runtime unchanged.

Source-only expansion lab preflight COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_EXPANSION_LABS_20260917.md`. Fixed provenance candidates, reused
unit/name/range policy, finite labs with measured/revised offsets and<=1hdelay.
31/39targets>=3readoutsites>=30patients;BUN/bicarb/phosphate6sites. Three sparse
labs2sites;5bedside unscanned. Source-role inventory reproduces; aggregate/hash
only. No fullsample acceptance, fitting, evaluatedoutcomes or promotion.

Expansion provenance inventory COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_EXPANSION_INVENTORY_20260917.md`. Original139367people/200859stays;
additional role-contained readout3433/fit10201people, excluding currentcohort and
ancestralpatients/sites. No labs/eligibility assessed.49existingpeople have extra
raw-table out-of-role admissions, not proven model exposure. Existing query-level
history isolation must not be generalized to allrawadmissions. No data role moves,
modelchanges,promotion or freshlockboxclaim. Next sourceevent support/boundary audit.

Confirmation-site support audit COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_CONFIRMATION_SUPPORT_20260917.md`. All28pooledscores reproduce;
BUN6/7sites below30 including singleton, bicarbonate/phosphate alsoonly1site>=30.
Prior equal-site reversal is not established domainfailure. Descriptivebootstrap
singletonbounds are degenerate; no confidenceclaim. No fittingchanges, patient
exclusions, promotion or runtime. Support/power confounding remains unresolved.

Independent head confirmation COMPLETE (2026-09-17):
`docs/research/findings/RECONSTRUCTION_CONFIRMED_HEAD_FINDINGS_20260917.md`. Existing frozenbody,
fit-site5fold nomination then independent confirmation, fixedBonferroni39.
0/28accepted, all33trained predictions remainoriginal. BUN pooled/site signs
differ;phosphate alpha preference nottransferable;bicarb underpowered siteevidence.
Three tests pass, originals reproduce1e-10. Safetyfilter notaccuracyrepair;
all39scope preserved. No threshold tuning, newlockbox or promotion.

Nested training and selection-margin localization COMPLETE (2026-09-17):
`docs/research/findings/RECONSTRUCTION_NESTED_TRAINING_FINDINGS_20260917.md`. Matched3x8epochs,
bitwise identical selected8/4/1weights, independenthead selection28supported.
Three nominal gains/three harms,macro1.797586->1.785098.20/22changedheads have
descriptive selection intervals crossingzero, including all3evaluation harms.
Scores reproduce1e-10;6regressiontests pass. Post-selection bias explicit;
no tuned acceptance policy, runtime or promotion. Next repair must independently
confirm source nominations rather than use inspected evaluation to choose heads.

Ancestry crosswalk COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_ANCESTRY_IDENTITY_20260917.md`. Original patient table verifies
4559source and1916query stays, patient/hospital consistency. Zero current-source
patient/stay/hospital overlap across all reconstruction roles. Namespace resolved;
historical source-file mismatch remains. No established leakage or model repair,
no runtime/promotion. Do not repeat unresolved-ID or overlap claims.

Warm-start ancestry diagnosis COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_WARM_START_20260917.md`. Byte-identical original checkpoint,
declared training cohort counts fail current reconstruction; IDs lack verified
crosswalk. Zero exact-string overlap cannot support independence. No evidence
of exact gradient exposure or blanket release invalidity. Source-fit-only
initializer/scaler or verified exclusion needed; no runtime/promotion changes.

Independent encoder/head selection preflight COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_NESTED_SELECTION_20260917.md`. Split only oldtune by whole
components,90encoder/89head,4/7sites; fit/eval/reserved unchanged.28/39supported
allroles at30patients;11explicit unsupported. No outcome-based split choice.
Oldcheckpoints ineligible for independence claim; retraining pending.3tests,
full hashes/support pass. No runtime/promotion or reduced project scope.

Source original/probe selector COMPLETE (2026-09-17):
`docs/research/findings/RECONSTRUCTION_READOUT_SELECTION_FINDINGS_20260917.md`.33targets, tune-only
selection choices saved before evaluation.3nominal gains(1median),3harms;
macro2.934232->2.966656. Including original/median does not guarantee retention.
Encoder and readout share tune, independent gate not established.3tests and
original/probe score reproduction pass; no runtime, interval or promotion.

Raw-history versus body probe COMPLETE (2026-09-17):
`docs/research/findings/RECONSTRUCTION_RAW_STATE_FINDINGS_20260917.md`. Strict-site33target source
fit/tune ridge+population selection, same masked inputs. Raw beats body0targets;
body improves original4, two are baseline-only. Bicarb/phosphate nonconstant
nominal improvements, but body harms5others. No blanket encoder loss claim,
external independence or promotion. Two raw-summary tests added.

Remaining support after masking COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_REMAINING_SUPPORT_20260917.md`.4637exact matched pairs,33MAEs
reproduce; lipase/triglycerides recent independent events almost universal,
still harm. Glucose17older-than6h cases harm descriptively. Not all-history
freshness or clinical signal adequacy proof. No cutoff optimization, fitting,
exclusions or promotion. All39catalogstatuses retained.

Readout calibration repair COMPLETE (2026-09-17):
`docs/research/findings/RECONSTRUCTION_READOUT_CALIBRATION_FINDINGS_20260917.md`. Fixed protocol,
sixV2 source tune-only offset and nonnegative affine L1 for real/process,
sourceeval alltargets. No6/6 original-model gain; creatinine affine2gain2harm,
triglyceride constant2/6. Not a demonstrated universal repair. Fourtests and
cached score/hash/role checks; no runtime/promotion. Do not repeat affine
calibration as untested or call reused sourceeval an external lockbox.

Source ranking localization COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_SOURCE_RANKING_20260917.md`. Six old V2 fit/tune replays,
alltarget MAEs reproduce. Triglyceride ranking drops0.662->0.088median before
external transfer; lipase retains0.357tune. Not universal signed reversal or
broad siteproof (0-1supported triglyceride sites). No calibration/update,
new significance or runtime promotion. Three ranking-helper tests added.

Error-shape diagnostic COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_SITE_ERROR_SHAPE_20260917.md`. Reproduced33target MAEs1e-10,
fit-derived center/tail thresholds with retrospective evaluation grouping.
Lipase/triglyceride majority and central harm rules out tail-only account on
this slice. Triglyceride ranking negative(N30); do not flip based on evaluation.
Glucose/bilirubin mean gain tail-concentrated; not prospective routing proof.
No training, patient exclusions, runtime change or promotion.

Source objective/gradient diagnosis COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_SITE_OBJECTIVE_20260917.md`. Source scores reproduce1e-7;
AST/ALT dominate57.27%net but not all gain,8target initial-baseline regressions.
Fit-MAD score ranking unchanged.33target x8 deterministic fit gradients at
selected endpoint,45,774persistent-layer parameters: lipase/triglycerides not
opposed by other-target mean, despite heldout harms. No update/AdamW simulation,
full-gradient or independent causal proof. No runtime, selection or promotion.

Strict site training/evaluation COMPLETE (2026-09-17):
`docs/research/findings/RECONSTRUCTION_SITE_TRAINING_FINDINGS_20260917.md`. Matched real/process/placebo
8epochs,33targets,141504updates/arm.178evaluationpatients4637pairs;12nominal gains
vsallreferences,2harms(lipase/triglycerides). Newpartition not paired old/new
accuracyrepair; pretraining/multiplicity/reuseddata limits remain. Eighttests;
no runtime/calibration/promotion. Sixunsupported catalogtargets retained.

Common site partition FROZEN (2026-09-17):
`docs/research/notes/RECONSTRUCTION_SITE_PARTITION_20260917.md`.536/179/178source patients,11disjoint
sites perrole,870reserved. Assignment uses provenance only, all39target support
reported;33trainable/evaluable,6unsupported. Transitive patient/site tests3pass.
Not newmodel efficacy or pretraining independence. Matched training pending.

Strict source-site support preflight COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_SITE_SUPPORT_20260917.md`. Found cohort-history hospital overlap
despite patient isolation:3sites/646sourcepatients. Connected-component guard
leaves893source/quarantines676;33targets per-target feasible,6need pooled-site
support. No common newpartition or training yet. Correct scope: this reconstruction
cohort, not blanket release invalidation. Runtime unchanged.

Source hospital increment audit COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_SOURCE_HOSPITALS_20260917.md`. Six V2source-only tune/evaluation,
allavailable targets. Bilirubin only1site supported/split, O2/glucose mixed site
directions; creatinine mean retains gain after single-site deletion6/6. New
audit's local-query-index mismatch caught;8multi-site patients unknown/inpooled.
Not original training bug, site-excluded retraining or promotion. Runtime unchanged.

Target-level source fit/tune replay COMPLETE (2026-09-17):
`docs/research/findings/RECONSTRUCTION_SOURCE_FIT_FINDINGS_20260917.md`.Six V2replications,207300source
pairs,12selected tune scores reproduce. Triglycerides sourcegeneralization
reversal6/6; bilirubin favorable sourceeval6/6 then hospitalreversal6/6;
creatinine favorable means6/6 but limited paired significance. Full39catalog,
no silent unsupported exclusions. Failure stage localized, not unique biological
cause or demonstrated repair. No new weights/runtime/promotion;10tests pass.

Fixed-hospital prediction stability COMPLETE (2026-09-17):
`docs/research/notes/RECONSTRUCTION_FIT_STABILITY_20260917.md`. Seven saved fits, immutable prediction
hashes/MAEs/keys/truths verified.35supported complete targets median between-fit
MSE fraction2.59%, descriptive averaging MAE improvement2.40%. Variability alone
not sufficient explanation; no pure optimizer/noise attribution. All39target
report, five tests, no model selection/promotion/runtime change. Next target-level
fit/tune reconstruction diagnosis before choosing a repair.

Full seven-split reconstruction COMPLETE (2026-09-17):
`docs/research/findings/RECONSTRUCTION_SEVEN_SPLIT_FINDINGS_20260917.md`. All planned seeds complete;
full39target/comparator/role machine summary. Three nominal7/7 targets, not
multiple-testing-adjusted promotion. Split versus initialization still confounded.
Seed83 serialization recovered with exact restored source checks and saved
prediction identity validation; no training restart or runtime change.

Exact source recovery and seed71 COMPLETE:
`docs/research/results/RECONSTRUCTION_REPLICATION71_RESULTS_20260916.md`. Both upstream event files
regenerated with identical SHA256; original checks retained. Three eight-epoch
arms, six combined nominal gains; intersection remains three across six seeds.
Hospital lipase harms process-only. No promotion; seed83 pending.

Source availability incident and verified backup (2026-09-16):
`docs/research/notes/RECONSTRUCTION_SOURCE_RECOVERY_20260916.md`. Seed71 terminated before training:
two bound temporary event files absent.2366 files backed up and1763 original
shards hash-verified. Original extraction chain located, recovery unfinished.
No new accuracy result, skipped gate or runtime change.

Seed59 corrected replication COMPLETE:
`docs/research/results/RECONSTRUCTION_REPLICATION59_RESULTS_20260916.md`.38targets, eight matched
epochs, hashes verified. Five combined nominal gains; five-split intersection
still albumin AST totalprotein. Real/process last-epoch selection leaves
convergence open. Samehospital cohort, no promotion. Seeds71/83 pending.

Seed43 corrected replication COMPLETE:
`docs/research/results/RECONSTRUCTION_REPLICATION43_RESULTS_20260915.md`. Full39target/eight-epoch
comparison, hashes verified. Nine combined nominal gains; four-split intersection
shrinks to albumin AST totalprotein. Bicarbonate numerical increment nearzero;
creatinine evaluation uncertain. Same hospital holdout, no multiplicity-adjusted
or forecast promotion. Seeds59/71/83 remain under fixed protocol.

Seed31 corrected replication COMPLETE (2026-09-15):
`docs/research/results/RECONSTRUCTION_REPLICATION31_RESULTS_20260915.md`. Three matched eight-epoch
arms, source-only selection, new patient-role evaluation and hash verification.
38 targets (anion gap unsupported),14758 heldout pairs. Nine nominal combined
gains both roles; four overlap seeds7/19/31. Bilirubin harms process in hospital.
Repeated hospital cohort and unadjusted tests, no promotion/runtime change.

Seed19 corrected replication COMPLETE:
`docs/research/results/RECONSTRUCTION_REPLICATION19_RESULTS_20260914.md`. Allthree8epochs complete;
fresh patient-role heldout evaluation, fit-only medians, dependency verification.
Nine real-over-process nominal overlaps; seven beat allthree comparators both
roles, four also inseed7. Same hospital cohort, not independent or multiplicity
adjusted. Current reconstruction only, no promotion/runtime changes.

Seed19 corrected replication first epoch:
`docs/research/notes/RECONSTRUCTION_REPLICATION19_PROGRESS_20260914.md`. Newdriver process-wiring
defect diagnosed/fixed in V2; allarms rerun, unaffectedlogs exact. Eight-epoch
experiment incomplete; no heldout inspection or promotion. Use V2runner only.

Repeated-split source/support preflight complete:
`docs/research/notes/NATIVE_RECONSTRUCTION_SPLIT_SUPPORT_20260914.md`.38/39source-supported all7;
aniongap support gaps explicitly retained. Same hospitalholdout reused.
No training/effect confirmation; per-split refitting required next.

Frozen multiplicity sensitivity:
`docs/research/notes/NATIVE_RECONSTRUCTION_MULTIPLICITY_20260914.md`.156comparisons, totalprotein
borderlineadjusted p; MonteCarlo uncertainty crosses threshold. No formal
confirmation/promotion; original nominal findings retained with limitations.

Trained observation-process control complete:
`docs/research/findings/NATIVE_RECONSTRUCTION_PROCESS_FINDINGS_20260914.md`.Eight matched epochs;
real-over-process nominal gains7targets bothroles. Process itself beatspop15/9.
Separates fixed perturbation from trained comparator; single split, no promotion.

Frozen numerical-input path diagnostic:
`docs/research/notes/NATIVE_RECONSTRUCTION_NUMERIC_PATH_20260914.md`.All39 tested,12of14nominal
winners harmedboth when values neutralized retaining timing/support. Inference
OOD only; trained process control still needed. No new weights/promotion.

Frozen expanded patient alignment:
`docs/research/notes/NATIVE_RECONSTRUCTION_ALIGNMENT_20260914.md`.Exact all-other-patient expected
errors,38/39positive eachrole; all14nominal winners positive. Not a new gate,
causal/physiology-only proof or promotion. Source predictions reproduced.

Expanded real/placebo eight-epoch experiment COMPLETE:
`docs/research/findings/NATIVE_RECONSTRUCTION_EXPANDED_FINDINGS_20260914.md`.14current reconstruction
targets nominal gains bothroles; selectedplacebo0. Sharedstate not uniformly
better (BNP/magnesium harm); single reused split, no multiple-test adjustment,
no forecast/interval/runtime promotion. Supersedes partial training below.

Expanded real-vs-deranged shared-body training epoch1 complete:
`docs/research/notes/NATIVE_RECONSTRUCTION_EXPANDED_TRAINING_PROGRESS_20260913.md`.All25511fit
pairs trained; real tune improves~3.5%,placebo selects0. Full8epochs pending,
no heldout error inspection or promotion; resume optimizer/RNG checkpoints.

Expanded exact as-of reconstruction samples complete:
`NATIVE_RECONSTRUCTION_EXPANDED_20260913.md`.49460patient-target pairs,
39source-eligible/36all-role-supported variables; all1763shards verified.
No training or efficacy claim; full shared-body/control training pending.

Full-cohort original-event support upper bound:
`docs/research/notes/NATIVE_RECONSTRUCTION_SUPPORT_CEILING_20260913.md`.36/53potentially meet30
patients eachrole; not conflict/masked-history eligibility or accuracy evidence.
No training/runtime change. Expand exact builder before next efficacy audit.

Full-coverage matched-budget training complete:
`docs/research/findings/NATIVE_RECONSTRUCTION_FULL_COVERAGE_FINDINGS_20260913.md`. Balanced selects7,
eventweighted0; small descriptive heldout improvements. ALL comparisons lack
minimum30patient support (roles16). Not statistical failure/equivalence proof;
larger support preflight needed, no runtime/promotion.

Exact multi-target sampler exposure audit:
`docs/research/findings/NATIVE_RECONSTRUCTION_EXPOSURE_FINDINGS_20260913.md`. Selected checkpoints
consume124/879 and115/879 fit pairs; event-selected omits12 direct targets.
Full-coverage training completed (above); no statistically established efficacy.

Frozen shared-body source-selected linear probe:
`docs/research/findings/NATIVE_RECONSTRUCTION_LINEAR_FINDINGS_20260913.md`. No supported improvement
with retrained ridge/placebo across35targets/twoencoders. Not informationabsence.
No runtime or encoder change; tiny neural corrections not solely head proof.

Frozen multi-target patient-signal audit:
`docs/research/notes/NATIVE_RECONSTRUCTION_PATIENT_SIGNAL_20260913.md`. Same-target derangement,
fit-constant diagnostic; nearconstant readout corrections, latent statusunknown.
All parent scores reproduced; no runtime/weight changes or promotion.

Actual shared-body multi-target sampling comparison:
`docs/research/findings/NATIVE_RECONSTRUCTION_SAMPLING_FINDINGS_20260913.md`.35targets, two8epoch
fits complete; no supported heldout gain or balanceadvantage. Some CIsunsupported.
Private selectedweights retained, no runtime/promotion. Not merely gradient smoke.

Initial multi-target body gradient diagnostic:
`docs/research/notes/NATIVE_RECONSTRUCTION_GRADIENT_BALANCE_20260913.md`.232samples38targets,
densevitals row imbalance confirmed; initializationcosines not causalconflict.
No training/promotion; avoid balancing based onlyon scalar loss sizes.

Source-group encoding cache equivalence:
`docs/research/notes/NATIVE_RECONSTRUCTION_CACHE_EQUIVALENCE_20260913.md`. Same-source-group-only
memoization,64anchor exacthash,1877anchor counts match. No runtime/efficacy.

Original-event multi-target joint-gradient bridge:
`docs/research/notes/NATIVE_RECONSTRUCTION_JOINT_STEP_20260913.md`. Tokenwidth fixed,15tests,
four realtarget losses update samebodyparameters once. No heldout/training
performance/promotion. Oldbuilder sourcebindings preserved as historical.

Raw-event multi-target example construction:
`docs/research/notes/NATIVE_RECONSTRUCTION_BUILDER_20260913.md`.14tests, real8fitpatient smoke
6249masked examples across targets; no cached-history union, no future revision
leak in tests. No accuracy/training/runtime/promotion; fullcohort remains.

Multi-target native-cache union rejected:
`docs/research/notes/NATIVE_MULTITARGET_CACHE_SCOPE_20260913.md`. Query-conditioned conflict/revision
views differ at same historical time; naive union unsuitable for earlier new
anchors. Raw-event reconstruction needed. No completed inventory/training/runtime.

Feature-level support audit and fixed20patient guard:
`docs/research/findings/NATIVE_BODY_SUPPORTED_RANGE_FINDINGS_20260913.md`.446numericfeatures retained;
no incremental gain over range bounding or population/placebo. Zero counts
include maskedtargets. No runtime/promotion; not a general no-information proof.

Fit-only range guard for multiwindow reconstruction:
`docs/research/findings/NATIVE_BODY_FIT_RANGE_FINDINGS_20260913.md`. SparseCPKslopes2fit/0tune support,
extreme extrapolation mitigated uniformly across numericfeatures. No stable
population/placebo gain; candidateonly, no runtime. Not priorJEPA rootcause.

Multi-scale masked current MAP and exact tail audit:
`docs/research/findings/NATIVE_BODY_MULTISCALE_FINDINGS_20260913.md`. No stable improvement; new
catastrophic evaluation tail localized to short-window CPK feature contribution.
Source-only reliability investigation remains, no promotion/runtime change.

Baseline-independent raw reconstruction control:
`docs/research/findings/NATIVE_BODY_UNRESTRICTED_FINDINGS_20260913.md`. Same current-only role/seed,
no nested-baseline support or numeric transformation. No significant raw
advantage in either heldout role; no runtime/promotion. Not information absence.

Pre-mask original-time validation:
`docs/research/notes/NATIVE_BODY_TIME_VALIDATION_20260913.md`. Four reproduced invalid-time cases
repaired; 14 tests pass. All266872 original query histories legal, ruling out
this exercised defect in the inspected cache, not all timing problems. No promotion.

Expanded current-only masked MAP raw control:
`docs/research/findings/NATIVE_BODY_EXPANDED_FINDINGS_20260913.md`. Removedfuturepairselection,
onefreshquery/allsourcepatients; no stable rawridge gain across311/194heldouts.
Bounded summaries, not allinformation/model impossibility. No runtimechange.

Native masked raw-history ridge control:
`docs/research/findings/NATIVE_BODY_RAW_CONTROL_FINDINGS_20260913.md`. Samecohort/mask/donors,
fit-only scaler/source-tune alpha. Raw not above population bothroles;
placebo gain sourceonly. No stable encoder-specific loss established.

Native body masked MAP real/placebo training:
`docs/research/findings/NATIVE_BODY_MASKED_FINDINGS_20260913.md`.53fit27tune50+50heldout,8epochs.
Real selects0/no populationgain, placebo comparisonNS. No promotion/runtime.

Native body reconstruction supervision:
`docs/research/notes/NATIVE_BODY_RECONSTRUCTION_20260913.md`. Four leakage/gradient tests and
one realfreshsource step. Body path now supervised, not accuracyvalidated.
No runtime/weightssaved/promotion. Matched patient-shuffled training pending.

Native replay parameter movement:
`docs/research/notes/NATIVE_ORGAN_PARAMETER_UPDATES_20260913.md`. Exact epoch0/8 comparisons:
organ updates nonzero, body-specific updates zero botharms. Structural scope
confirmation, not unique rootcause or accuracy repair. No runtime/training.

Owner event mixture audit:
`docs/research/notes/NATIVE_OWNER_EVENT_MIXTURE_20260913.md`. MAP/lactate directowner pool; no
lactate afterfinalMAP in96sourcepilot streams. Not final-overwrite explanation,
not earlier-event accuracy ablation; no training/runtime change.

Nested excluded-patient baseline and residual training:
`docs/research/findings/NATIVE_NESTED_RESIDUAL_FINDINGS_20260913.md`. Baseline32epochs +2residual8epoch
arms complete. Patient separation verified; both residualarms select1 but
no significant heldout gain or state-overanchor benefit. Smallpilot, no promotion.

Native pretraining identifier overlap audit:
`docs/research/notes/NATIVE_PRETRAINING_OVERLAP_20260913.md`. Current recorded source and MAPquery
subject/stay intersections0, hashboundnow; historicalhash unavailable.
Do not claim pretraining contamination or guaranteedisolation. No training.

Native residual baseline exposure:
`docs/research/notes/NATIVE_RESIDUAL_TRAINING_EXPOSURE_20260913.md`. Exact sampler epoch25 replay,
64/64 residualfit patients seen,22/64 queries seen; otherroles0. Not crossfit.
Descriptive slope shifts, not causal proof. Clean excluded-patient residual
training remains needed; no runtime/calibration/promotion.

Native objective trained controls:
`docs/research/findings/NATIVE_ORGAN_OBJECTIVE_FINDINGS_20260913.md`. Same64fit32tune8epochs,
fixed teacher, centered-angular teacher, value-only: allselect0. Loss shortcut
not necessary for forecastfailure; centeredloss insufficient. No runtimechange.

Native replay teacher/constant latent diagnostics:
`docs/research/notes/NATIVE_ORGAN_LATENT_OBJECTIVE_20260913.md`. Source fit-mean learned latent
correction reproduces~99.36% tune fixedteacher loss improvement. EMAshift not
soleexplanation. No optimizer updates; constant fittedfromfitoutputs only.
Objective shortcut evidence, not universalcause/forecastrepair/promotion.

Conditional owner-state readout training:
`docs/research/findings/NATIVE_ORGAN_CONDITIONAL_FINDINGS_20260913.md`. Current+change vs matched
change-only/anchor-only. Conditional selects1 with tiny tune gain, heldout
bothNS; controls select0. Four tests; no serving/promotion. Report
`artifacts/native_organ_conditional_pilot_20260913.json`.

Native organ state variation probe:
`docs/research/notes/NATIVE_ORGAN_STATE_VARIATION_20260913.md`. Initial/final none/sparse source
fit/tune states: current variation persists, future-change variation low and
shrinks after training. Tune delta/current RMS1.48%/1.87%, not information
retention. No prediction gain or solecause proof, no training/runtime change.
Report `artifacts/native_organ_state_variation_20260913.json`.

Native state-input matched training control:
`docs/research/findings/NATIVE_ORGAN_STATE_INPUT_FINDINGS_20260913.md`.64fit32tune seed7, identical
eight-epoch replay with/without head state coordinates. Full logs exact;
both selectepoch0, tune worsens to8.41926/8.42216. No state benefit demonstrated;
anchor correction alone reproduces failure. Not full-body impossibility or
unique cause. Report `artifacts/native_organ_state_input_control_20260913.json`;
fixed-head attribution separately complete. Runtime/intervals unchanged.

Native replay exact fit trace:
`docs/research/notes/NATIVE_ORGAN_REPLAY_TRACE_20260913.md`.16tunescores exact; fitimproves while
tuneworsens. Finalepoch8 no significant heldoutchange; selectedepoch0 unchanged.
Not disconnectedlearning or graphbenefit. Smallpilot, no runtime/promotion.

Native anatomical replay paired pilot COMPLETE:
`docs/research/findings/NATIVE_ORGAN_REPLAY_PILOT_FINDINGS_20260913.md`.64fit/32tune,none/sparse8epochs
both selectepoch0; no source-tune incremental benefit. Heldout64each exactly
baseline byconstruction, not newefficacy.9bindings; no fullcohort/gate/promotion.

Native organ real future-supervision bridge:
`docs/research/notes/NATIVE_ORGAN_FUTURE_SUPERVISION_20260913.md`. One sourceMAPpair2steps,
real value+EMA JEPA target loss reaches organupdate afterzero-start headstep.
Newfloat64anchor subtraction fixed. Integration only, no heldoutgain or
wholebodyutility claim; scoped/unscoped fulltraining pending, runtimeunchanged.

Native anatomical replay bridge:
`docs/research/notes/NATIVE_ORGAN_REPLAY_BRIDGE_20260913.md`.8sourcefit streams112events replayed,
3-8observedorgans, connected gradients. SameJEPA pre-mixer organ proposals,
14directcanonicalslots; measurement-only, graphclosed. Integration NOTtraining
or forecastgain/promotion. Future-state change readout/supervision stillpending.

Persistent chronological observation replay:
`docs/research/notes/PHYSIOLOGY_OBSERVATION_REPLAY_20260913.md`. Prediction/correction split,
clock countedonce, alternating-organ latent replay and gradients tested.
25tests; no trained raw-event bridge or forecastgain, no promotion. Previous
MAPdidnotcallthisAPI; do not claim its failures were doubletime counting.

Persistent update-mask repair:
`docs/research/notes/PHYSIOLOGY_UPDATE_MASK_REPAIR_20260913.md`. Reproduced no-observation bodychange
and NaNcontamination; masked beforeprojection and bodycorrection.21tests pass.
Unused by recentMAPnative trials; not their rootcause or efficacyrepair. Full
organ online assimilation stillneeded; no weights/calibration/promotion change.

MAP predict-only structural bridge:
`docs/research/notes/MAP_PREDICT_ONLY_CONTROL_20260913.md`. Originalfactorial forecasts reproduced,
671/672endpoint-matched source probes equivalent with fullcadence/target-only
projection and no correction withouttarget. Decay-only added events valid;
not whole-body efficacy or unique rootcause. Three tests; no runtime/promotion.

MAP projection content ALL7 COMPLETE:
`docs/research/findings/MAP_PROJECTION_CONTROL_FINDINGS_20260913.md`.84bindingsverified. No gains/harm
againstcontrol/full; harmsown7/7source/hospital. Other projection content not
necessary for gap; cadence alone not proven. No runtime/calibration/promotion;
supersedes pilot pending. Do not repeat as untested metadata-removal repair.

MAP training-matched projection content seed7:
`docs/research/findings/MAP_PROJECTION_CONTROL_FINDINGS_20260913.md`. Only history projection becomes
target-only; original activation and query retained. Support-control difference
NS bothroles, own harm both. Not unique cadence attribution or stable repair;
six fixed seeds pending. No runtime/calibration/promotion.

MAP unclippedclock COMPLETE21fits:
`docs/research/findings/MAP_UNCLIPPED_TRAINING_FINDINGS_20260913.md`.All7 sameprotocol. Corrected-old
gain0eacharm/role; corrected full-own harm7source5hospital.98bindings verified.
Not stable repair, no runtime/calibration/promotion; do not repeat as untested.

MAP clock retraining seed7: `docs/research/findings/MAP_UNCLIPPED_TRAINING_FINDINGS_20260913.md`.
3matchedfits no gain; hospital own/full harm,controlNS. Full-own gap persists.
Oneof7, sixpending; no runtime/calibration/promotion or rootcauseclaim.

MAP rawclock recovery COMPLETE60shards: `docs/research/notes/MAP_UNCLIPPED_CLOCK_20260913.md`.
All266872queries625722tokens, oldcontent/schedules exact;234666clockchanges.
182bindings verified. Input repair only; MAP retraining/accuracy pending.

MAP clock scope: `docs/research/notes/MAP_NATIVE_CLOCK_SCOPE_20260913.md`. CurrentMAP builder
retains clippedclock; quantified all7parts, source24.71%hospital18.35%zero
encoded active-step deltas seed7. EarlierKclocktrial not MAP repair. Input
limitation confirmed, accuracycausation/retraining pending; no promotion.

MAP direct latest-observation carry: `docs/research/notes/MAP_NATIVE_CARRY_20260913.md`.
21sourcefit probes exact prediction replay tolerance; almost no updates after
latestMAP on availability-native anchors. Rules against simple postlatest
overwrite narrative, not older-history forgetting; no repair/promotion.

MAP matched history/query factorial COMPLETE:
`docs/research/findings/MAP_CONTROL_FACTORIAL_FINDINGS_20260913.md`.14fits; expanded support history
harms even with own query7/7both; ownhistory+supportquery improves control
7/7source6/7hospital, no significant gain/harm vsown. History-path localization
not isolated cadence or full-body repair. No runtime/calibration/promotion.

MAP control path replay: `docs/research/notes/MAP_CONTROL_PATHS_20260912.md`. Original predictions
exact7/7; crossed own/control query/history, source-tune only. Query-only harm7,
history-only harm3, gain0. Not trained repair or sole-cadence attribution;
no runtime/calibration/promotion.

MAP numeric control COMPLETE7fits: `docs/research/findings/MAP_NATIVE_NUMERIC_CONTROL_FINDINGS_20260912.md`.
Same cadence, mask/age/context, no extra numbers. Control-own harms7/7both;
control-full gains0/7both. Numeric removal not repair; support/cadence/query
remain unseparated.18bindings verified; no runtime/calibration/promotion.

MAP numeric-content control PREPARATION:
`MAP_NATIVE_NUMERIC_CONTROL_PROTOCOL_20260912.md`. Same-cadence numeric ablation
helper and seven tests complete, matched training still pending. Distinct from
old glucose support and inference-only MAP scope replay; no efficacy claim.

MAP update counts: `docs/research/notes/MAP_NATIVE_UPDATE_COUNTS_20260912.md`. New query task,
all7 partitions, no outcomes in counts. Own/full active updates differ roughly
threefold; confirms comparison also changes cadence. Not causal repair; prior
scope-only replay already negative. No fitting/runtime/promotion.

MAP refinement seven-split COMPLETE: `docs/research/findings/MAP_NATIVE_QUERY_REFINE_FINDINGS_20260912.md`.
Same source-selected own/full models, new matched draws and reset AdamW32epochs.
Full-oldfull gains4source0hospital; full-own0both, harms7source6hospital.
Additional budget insufficient stable repair; not convergence/solecause proof.
Summary `artifacts/map_native_query_refine_summary_20260912.json`; no runtime
or calibration/promotion. Reused source/hospital evaluation, not fresh lockbox.

New MAP refinement seed7: `docs/research/findings/MAP_NATIVE_QUERY_REFINE_FINDINGS_20260912.md`.
Matched32extraepochs freshoptimizer: source improves botharms, hospitalnot;
full-ownharmremains. Oneofseven, no promotion or rootcauseproof.

New MAP shared gradients: `docs/research/notes/MAP_NATIVE_QUERY_GRADIENTS_20260912.md`.
42selected sourceprobes, smalllatent gradient; no dominantvalue reversal.
Not completeoptimizationpath or trained lossrepair. Runtimeunchanged.

New MAP value-objective: `docs/research/notes/MAP_NATIVE_QUERY_OBJECTIVE_20260912.md`.
Fullworse7/7 SmoothL1 allroles+bothseen. Not MAE-only mismatch; no totalJEPA
loss/path/unique-cause claim or modelchange.

New MAP actual sampled exposure: `docs/research/notes/MAP_NATIVE_QUERY_EXPOSURE_20260912.md`.
Exact RNG replay, bothseenharm6/7, neitherharm7/7. Querycoverage~10-11%at32;
not solecoverage explanation or budget repair. No runtime/promotion.

New MAP nativequery neural14fits + fitgap: `docs/research/findings/MAP_NATIVE_QUERY_TRAINING_FINDINGS_20260912.md`.
Full-own harm7/7 fitpopulation/tune/newsource/hospital; bothbeatpersistence7/7.
Original scores exact. Timingalone not repair; fit sampled-exposure unresolved.
No remaining seed runs for this protocol; runtime unchanged, no promotion.

New MAP nativequery neural seed7: `docs/research/findings/MAP_NATIVE_QUERY_TRAINING_FINDINGS_20260912.md`.
Actual newtask own/full fits complete, full-own harmbothpatient/hospital;
bothpersistencegain. One ofseven only; otherseeds pending, no promotion.

New MAP source-only transforms: `docs/research/notes/MAP_NATIVE_TRAINING_TRANSFORMS_20260912.md`.
Seven split transforms frozen,46supportedvars; compact/dense own/full equivalence
tested. Preparation only; matched neural fitting/evaluation remains pending.

New MAP neural batch smoke: `docs/research/notes/MAP_NATIVE_TRAINING_BATCH_20260912.md`.
All60shards loadedragged, dense-equivalence tests pass; real four-example
source-fit forward/backward/AdamW/EMA works. No trained efficacy/promotion;
full matched trial and training transforms stillpending.

MAP newquery nativehistory COMPLETE: `docs/research/notes/MAP_NATIVE_QUERY_HISTORY_20260912.md`.
60shards266872queries205723eligiblepairs,7461commonhistories exact. Full source
hash/unique query coverage/native assembly verification passed. Matched JEPA
training remains pending; no accuracy claim/runtime/promotion.

New MAP query nativehistory integration: `docs/research/notes/MAP_NATIVE_QUERY_HISTORY_20260912.md`.
Shard0/60,1885queries;19oldcommonhistories exact, allassembledsequences exact.
Four scheduler tests pass. Actual input bridge, not neural training/promotion;
remaining shards pending, no runtime change.

MAP neural-task scope audit: `docs/research/notes/MAP_NEURAL_TASK_SCOPE_20260912.md`.
Exact eligiblequery overlap3905/205723; oldnative differs from newtree task.
No newmodel test; identifies missing nativehistory-to-neural bridge.

MAP native-missing fitgap: `docs/research/notes/MAP_MISSING_FIT_GAP_20260912.md`.
Full-support fit4/7gain, newroles0/7; full-ownfit7/7other0/7. Originalexact.
No further missingrepresentation repair justified by this diagnostic alone.

MAP derivedgap support strata: `docs/research/notes/MAP_DERIVED_GAP_STRATA_20260912.md`.
Frozen current/history-only/none groups; supportedhospital147patients0/7gains
overfull/own/no-gap. No trained repair or promotion; no future-based grouping.

MAP derivedgap matched training: `docs/research/notes/MAP_DERIVED_GAP_TRAINING_20260912.md`.
21fixed fits, no-gap/derived/support; derived-full0/7gainboth, own0/7both.
Numerical hospital improvement not stable attribution or promotion.

MAP explicit derived gap preparation: `docs/research/notes/MAP_DERIVED_ANION_SUPPORT_20260912.md`.
Timestamp/availability-safe separate variable; fit/hospital support40.02/38.66%.
No training/accuracy/promotion yet; original reported gap preserved.

MAP anion-gap contract: `docs/research/notes/MAP_ANION_CONTRACT_20260912.md`.
Raw5262exact; same-time fullpanels withKmatch6.56%fit26.89%hospital.
Potential conventionmix, no proven modelimpact or automatic Ksubtraction.

MAP feature usage audit: `docs/research/notes/MAP_FEATURE_USAGE_20260912.md`.
Actual tree split gains and distinct source-patient support; rare<30 zeroall7.
Usage not causalattribution; no rarefilter repair/routing/runtime/promotion.

MAP missing support comparator: `docs/research/notes/MAP_MISSING_SUPPORT_20260912.md`.
Seven matched numeric-removed fits; full-support0/7bothroles, support-own0/7.
Closes native-missing attribution gap, no stable numerical increment/promotion.

MAP native missing routing: `docs/research/notes/MAP_NATIVE_MISSING_TRAINING_20260912.md`.
14 cloned own/full fits, NaN only for maskzero numbers, same explicit masks/ages.
Full-old gain0/7source2/7hospital; full-own0/7both. No stable repair/promotion.

MAP paired increment decomposition: `docs/research/notes/MAP_NATIVE_INCREMENT_DECOMPOSITION_20260912.md`.
Same frozen predictions, direct difference of support and numeric error contrasts.
Fit support increment larger7/7; numerical increment small, no heldoutgains.
Ordered fitted-model decomposition not causal physiology fractions. No promotion.

MAP-native event-equal fitting: `docs/research/notes/MAP_NATIVE_EVENT_WEIGHT_TRAINING_20260912.md`.
21 cloned fixed trees, event-within-patient equal weights, unchanged evaluation.
Full-original/own0/7gainbothroles, persistence7/7. Negative training repair,
distinct from prior scoring sensitivity. No runtime/calibration/promotion.

MAP-native partition recheck: `docs/research/notes/MAP_NATIVE_PARTITION_CHECK_20260912.md`.
Allseven patient ANDfutureevent role intersections0; no new repair/promotion.

MAP shrinkage support control: `docs/research/notes/MAP_NATIVE_SHRINK_SUPPORT_20260912.md`.
Full reproduction exact, matched supportalpha; full-support1/7source0hospital.
No stable numerical attribution/equivalence/runtime/promotion.

MAP-native increment shrink: `docs/research/notes/MAP_NATIVE_INCREMENT_SHRINK_20260912.md`.
Source-tune fixed grid, partialrepair vsfull3/7source5/7hospital; own4/7source
0hospital. No full gate/calibration/runtime/promotion or neuraltraining.

MAP tree leaf patients: `docs/research/notes/MAP_NATIVE_LEAF_PATIENTS_20260912.md`.
Actual numeric tree traversal exact. Seed7 full1/3000leaf<5patients;
not widespread singlepatientleaf diagnosis or constraint-retraining proof.

MAP native sampling-gap scoring: `docs/research/notes/MAP_NATIVE_SAMPLING_GAP_20260912.md`.
Fixed predictions, raw>=2/snapshot<2 preanchor groups; full-own0/7gain all.
No-gap hospital harm persists. No causal loss attribution/routing/promotion.

MAP-native raw versus lag support: `docs/research/notes/MAP_NATIVE_RAW_DENSITY_20260912.md`.
Qualified asof sourceversions, recentdraw counts; labs sparse in both, vitals
some gridloss. Seed7 descriptive, not repair/wholedata absence. Six tests.

MAP-native density: `docs/research/notes/MAP_NATIVE_HISTORY_DENSITY_20260912.md`.
V2actualdrawtime criterion, sparse repeated labs vs dense vitals in snapshots.
Seed7descriptive, notrawstream absence/accuracycause; no promotion.

MAP-native outcome weights/support: `docs/research/notes/MAP_NATIVE_OUTCOME_WEIGHTS_20260912.md`.
Frozen scoring sensitivity keeps full-own0/7gain outsidefit; support-own7/7fit
but0/7newpatients. No event-balanced training or runtime/promotion.

MAP-native tree fit/stopping: `docs/research/notes/MAP_NATIVE_TREE_FIT_GAP_20260912.md`.
Originalscores exact. Full-own fitgain7/7 but newpatients0/7; earlier source-
tune stopping0/7gain overfixedfull. Not unique rootcause or promotion.

MAP-native raw-history training: `docs/research/notes/MAP_NATIVE_ANCHOR_TREE_20260912.md`.
21fixed HGB fits, seven original roles; full-own0gain1harm bothroles,
full-support0/0, persistence7/7. Newtask not directly oldMAEcomparable;
no native neural/conformal/runtime/promotion. 41bindings verified.

MAP-native fixed-lag input reconstruction: `docs/research/notes/MAP_NATIVE_ANCHOR_SUPPORT_20260912.md`.
All266872queries,60shards;7461overlap current/history exact, labelcurrent exact
within scaling tolerance. Matched fitting pending; no accuracy/promotion.

MAP availability-native support: `docs/research/notes/MAP_NATIVE_ANCHOR_SUPPORT_20260912.md`.
Same stay universe/roles,205723pairs1702patients, medianage10min. Old labels
exact; futureevent reuse explicit. New task not trained, no accuracy/promotion.

MAP preanchor historychange strata: `docs/research/notes/MAP_HISTORY_CHANGE_SUBGROUPS_20260912.md`.
Fixedpredictions, no futuredefinedgroups; changegroup0/7gainboth,38hospital
patients. Not proven dilution/absence, no routing/promotion.

MAP recorded pressor history training: `docs/research/notes/MAP_PRESSOR_HISTORY_TRAINING_20260912.md`.
Seven matched HGB controls exact, ratehistory0/7gain vs currentrate allroles;
oldsubset1/7harmboth. Not a sufficient repair, no JEPA/runtime/promotion.

MAP pressor history support: `docs/research/notes/MAP_PRESSOR_HISTORY_20260912.md`. Pastcutoff1/3h
rate/age/known/delta, current exact; norepi recordedchange substantial.
Eighttests, onlinefalse. No training/promotion; not fullstart/stop stream.

MAP recorded rate variation: `docs/research/notes/MAP_PRESSOR_VARIATION_20260912.md`. Sourceonly,
eightchannels all7fits; principalnorepi/vasopressin nonconstant and supported.
No predictiveinformation proof/channel exclusion or runtime/promotion.

MAP recorded-rate strata: `docs/research/notes/MAP_PRESSOR_SUBGROUPS_20260912.md`. Fixedpredictions,
support stratum0/7gain overbaseline/supportbothroles,45hospitalpatients.
Not equivalence or no-treatment interpretation; no routing/promotion.

MAP recorded pressor increment: `docs/research/notes/MAP_RECORDED_PRESSOR_TRAINING_20260912.md`.
Seven matched HGB baseline/support/rate, oldtune fixed, baseline exact.
Rate-baseline0/7gainboth old/all source/hospital; no stable repair. Retrospective
snapshots only, no JEPA training/runtime/promotion.

MAP recorded pressor cache: `docs/research/notes/MAP_RECORDED_PRESSORS_20260912.md`. Five separate
drugs/three dimensions, latestinvalid/conflict blocks stale values;8tests.
Private cache built, recordedtimeonly/onlinefalse. No fitting/promotion.

MAP pressor inventory/units repair: `docs/research/notes/MAP_PRESSOR_CAPTURE_20260912.md`.
Explicit units/min unrecognized; fixed60x drugunit/hour,zero preserved,13tests.
Recovered vasopressin support2101queries/231patients. Missingavailability
limits onlineclaims; nativeMAP didnotconsume pressors, not gapcauseproof.
No training/promotion; newparser affects futureextractions only.

MAP update scope: `docs/research/notes/MAP_UPDATE_SCOPE_20260912.md`. Fixed seven fullweights,
MAP-only history updates, fullquery retained. Originalexact7/7; gain1source0hosp,
own0both. Not trained repair; content/cadence jointlyremoved. No runtime/promotion.

MAP value-JEPA gradients: `docs/research/notes/MAP_LOSS_GRADIENTS_20260912.md`.42source-fit probes,
own/full paired batches, selectedweights. Full13/21opposing butratio~.0153;
no rawcombined value reversal. Not wholeoptimization proof, no runtime/refit.

MAP value loss audit: `docs/research/notes/MAP_VALUE_OBJECTIVE_20260912.md`. Exact SmoothL1 with
scale16.3086, full worseown7/7tune; descriptive. MAE choice alone insufficient
explanation; latentloss conflict not measured. No training/runtime/promotion.

Expanded MAP fitting replay: `docs/research/notes/MAP_EXPANDED_FIT_GAP_20260912.md`. All14selected
weights replay, oldpredictions/tune reproduced. Full-own sampled-both1gain3harm,
sampled-neither0gain5harm; source-tune0gain2harm. Samepatient unseenrows are not
independent outcomes; selectedepochs differ. No fit/runtime/promotion.

Expanded MAP exposure: `docs/research/notes/MAP_EXPANDED_EXPOSURE_20260912.md`. Exact sampler replay
and source-tune curves, no refit. Allpatients seen at32, only36.9-39.3%rows;
fullbest tune worseown7/7, latebest improves3/7. Not solebudgetcause or repair.
Selected fitting error remains unmeasured; no runtime/promotion.

MAP expanded native fitting: `docs/research/notes/MAP_EXPANDED_NATIVE_20260912.md`. Seven own/full
fits, old tune/scoring fixed, old predictions as controls;49directbindings match.
Full-oldfull0/7gainboth; full-own0/7both(sourceharm4). Own expansion1/7hospital.
Not a sufficient selection repair; fixed compute/composition confounded.
No intervals, new database evaluation or runtime/promotion.

Expanded MAP source-transform equivalence: `docs/research/notes/MAP_LABEL_SELECTION_20260912.md`.
All7 oldfit support/numeric/event stats exact acrossnativecache layouts.
Pretraining check only, no accuracy/promotion.

MAP all-native rebuild: `docs/research/notes/MAP_LABEL_SELECTION_20260912.md`, native subsection.
427180groups/55468queries; old5526histories exact, no truncation. Neural
training pending, no runtime/promotion. map_all_native_tokens_20260912.

MAP expanded fit trial: `docs/research/notes/MAP_SELECTION_TRAINING_20260912.md`. Oldtuneheldfixed,
28treearms with3leafchoices; oldcontrols exact. Full improvesoldfull4/1
source/hospital on oldrows, full-own0both. Partialrepair, not JEPA promotion.

MAP all-query input rebuild: `docs/research/notes/MAP_LABEL_SELECTION_20260912.md`, appended result.
Oldquery/history sixfields exact;33477MAPpairs with matchingcurrent truth.
Fixed-lag cache only, native expansion/training pending; no runtime/promotion.

MAP label-selection audit: `docs/research/notes/MAP_LABEL_SELECTION_20260912.md`. Unrelated Kfuture
eligibility removed;33477MAPpairs vs3925,1505patients vs1303,oldoverlapexact.
Newlabelcache only, inputrebuild/training pending. Not wholeeICU or promotion.

MAP freshness: `docs/research/notes/MAP_FRESHNESS_20260912.md`. Fixed1h preanchor split; fresh
full-own sourceharm4/7, gain0bothroles; olderharm3. Not stale-only failure,
not trainedselection or promotion.

MAP cross-model scores: `docs/research/notes/MAP_CROSS_MODEL_20260912.md`. Summarytrees0/7gain over
ownJEPA in bothroles; no stable fullJEPA benefit. Matchedlabels, not matched
inputrepresentation/algorithm. No fitting/promotion; qualifies summaryrepair.

MAP summary component training: `docs/research/notes/MAP_SUMMARY_COMPONENTS_20260912.md`. Full lag
heldfixed; ownsummary gain3/7both, addingother support/numbers0/7increment.
Samewidth and exactfullcontrols. Ownhistory repair evidence, notcoupling proof.

MAP summary attribution: `docs/research/notes/MAP_SUMMARY_ATTRIBUTION_20260912.md`. Saved support
model gains2/4source/hospital overlagfull without other numeric inputs.
Support-own0/0; no unique numerical-excursion attribution, no fitting/promotion.

MAP event-window history: `docs/research/notes/MAP_EVENT_HISTORY_20260912.md`. Five matchedtreearms,
seven source splits; exactlagcontrols. Full-lagfull gain2/4source/hospital,
noharm; full-own/support0gainboth. Partial representation repair, no neural
retrain/calibration/runtime/promotion. Aggregate map_event_history_20260912.

Event-summary quality: `docs/research/notes/EVENT_SUMMARY_QUALITY_20260912.md`. False-quality finite
spike reproduced then excluded; as-of revisions and validzero preserved.
Prevents contamination in future bedside summaries, not prior MAP failure
attribution or forecast repair. No runtime/promotion.

MAP raw history: `docs/research/notes/NATIVE_MAP_RAW_HISTORY_20260912.md`. Source-only tree selection,
same MAP labels and splits, full-own0/7gainboth/sourceharm3; full-support0gain/
sourceharm2. Fixed-lag not native events; no separate treatment features.
No calibration/runtime/promotion. Non-recurrent failure, not no-information proof.

Native MAP21fits: `docs/research/findings/NATIVE_MAP_FINDINGS_20260912.md`. Same anchors/roles, own/full/
non-MAP numeric-neutral support. Common20k bootstrap x3 agrees, full-own0/7gain
both/sourceharm5. Same-method subset retains harm5; not switching-only cause.
Source-only tuning, no MIMIC/calibration/runtime/promotion. Exact aggregate
reports native_map_common_bootstrap and map_measurement_method_20260912.

Bedside label scope expansion: `docs/research/notes/BEDSIDE_FUTURE_LABELS_20260912.md`. Five measured
6htargets at unchangedanchors; MAP3925pairs. Fixed quality flag omission in
future helper before use,12tests; no training/efficacy claim.

Bedside support strata: `docs/research/notes/BEDSIDE_HISTORY_SUPPORT_20260912.md`. Fixed5events
each, past-only; neither stratum real-support gain. Not a learned rule or
data-sufficiency proof, no promotion.

Common bedside bootstrap: `docs/research/notes/BEDSIDE_BOOTSTRAP_STABILITY_20260912.md`.20kdraws,
3streams agree, former1000draws second supportlab hospital pass disappears.
Prediction unchanged, no promotion; name-independent paired helper tested.

Bedside fit gap: `docs/research/notes/NATIVE_BEDSIDE_FIT_GAP_20260912.md`. Real-support3/7fitgain,
0tune,1evaluation,0hospital. Same points; one supportlab CI salt-sensitive.
Not sole hospital problem or rootcause proof; no training/promotion.

Bedside revision exclusion: `docs/research/notes/NATIVE_BEDSIDE_REVISION_CHECK_20260912.md`.
No repeated source identities in current cache; valid-to-invalid revision
snapshot/native regression passes. No forecast repair or new rootcause claim.

Bedside raw tree: `docs/research/notes/NATIVE_BEDSIDE_RAW_HISTORY_20260911.md`. Fixed-lag real vs
support0/7both; real-lab1/2gain, no harm. Same roles/labels, not identical
native representation. Does not isolate JEPA as solecause or promote.

Bedside trained support control14fits: `docs/research/notes/NATIVE_BEDSIDE_SUPPORT_20260911.md`.
Same event cadence, source-centered numbers zero in train/test. Realsupport
gain1source0hospital, harm0/1; supportlab gain1/2. No stable repair/promotion.

Bedside fixed components: `docs/research/notes/NATIVE_BEDSIDE_COMPONENTS_20260911.md`.7original
predictions exact; neutralnumbers0source1hospitalgain, removeobservations
sourceharm5hospitalgain1. Not stable deletion repair; cadence/support confounded.

Bedside rescale14fits: `docs/research/notes/NATIVE_BEDSIDE_RESCALE_20260911.md`. Temp z tail
motivates source-only new-channel transform; same cadence, controlsexact7/7.
Full-lab gain1source0hospital,harm0/1. Not sufficient repair or promotion.

Bedside14fit matched trial: `docs/research/notes/NATIVE_BEDSIDE_TRAINING_20260911.md`. Controls
reproduceexact7/7,44bindings. Bedside-lab gain0source1hospital,harm0/3; both
persistence7/7. Not stable incremental repair or promotion; observation values
and added event cadence remain combined, not a sole-cause demonstration.

Expanded bedside native tokens completed:207499groups,5526unchangedqueries,
48hreplay median68/max385. Both new query/history inputs ready; no training
or efficacy claim. See `docs/research/notes/NATIVE_BEDSIDE_EVENTS_20260911.md`.

Bedside quality/query build: existing bounds flag2352; invalid latest remains
missing rather than old-value resurrection. Identical5526queries, nonbedside
inputs exact; expanded snapshot cache created. Native history/training pending.
See `docs/research/notes/NATIVE_BEDSIDE_EVENTS_20260911.md`; no runtime or efficacy change.

Bedside native wiring: schedule/token observation whitelist expanded, source
identity namespaced, cross-source conflicts tested. Synthetic raw-to-token
path passes; not retraining, historical cause proof or promotion. See
`docs/research/notes/NATIVE_BEDSIDE_EVENTS_20260911.md`.

Bedside canonical/support: `docs/research/notes/NATIVE_BEDSIDE_EVENTS_20260911.md`.1.348Mevents,
same5526querytime support established for five missing vitals. Label-derived
units/rangequalification remain explicit limitations;11tests. No training,
accuracy or promotion; next native integration after quality checks.

Bedside inventory: `docs/research/notes/NATIVE_BEDSIDE_INVENTORY_20260911.md`. Same1916stays
numericnursing private extraction completed. Explicitentrytimes; not yet
unit/timequalified anchor features or efficacy. No promotion.

Native current input scope: `docs/research/notes/NATIVE_INPUT_SCOPE_20260911.md`. Repairedcache
41lab slots, key bedsidevitals absent, insulinrecord-only treatment. Scope
inventory, not new efficacy experiment or discovered rootcause.

Clean state support: `docs/research/notes/CLEAN_STATE_SUPPORT_20260911.md`. Full measured-state
prototype nearrows0/7increment, farhospitalharm3/7. No OOD-only repair claim.

Clean change size: `docs/research/notes/CLEAN_GLUCOSE_CHANGE_20260911.md`. Large-change fitgain7
does not generalize (1source/0hospital). Futuredefined strata not routing;
no amplify fix, training or promotion.

Conflict update scope: `docs/research/notes/NATIVE_CONFLICT_UPDATE_SCOPE_20260911.md`. Actual
checkpoint context and head changes verified7/7, no unintended frozencontext
explanation. No accuracy repair or promotion.

Clean outcome weighting: `docs/research/notes/CLEAN_FUTURE_EVENT_WEIGHTS_20260911.md`. Eventthen
patient equal errors retain0/7wholeincrement. No scoregate change/promotion.

Clean complementarity: `docs/research/notes/CLEAN_CONTEXT_COMPLEMENTARITY_20260911.md`. Correlated
own/fullerrors; equalblend gain0/7source1/7hospital. Descriptive oldpattern
replication, no prospective oracle or promotion.

Native teacher semantics: `docs/research/notes/NATIVE_TEACHER_CONTRACT_20260911.md`. Nonzeroanchor
equivalence test passes; teacher helper matches target-specific delta encoding.
Not new geometry efficacy or training repair.

Conflict initial gradients: `docs/research/notes/NATIVE_CONFLICT_INITIAL_GRADIENTS_20260911.md`.
Sameweights21probes,~1.3%median change/alignment.99995. No steps or repair;
do not infer later trajectory from initial geometry.

Conflict selection: `docs/research/notes/NATIVE_CONFLICT_SELECTION_20260911.md`. Both significant
hospital harms persist at equal selectedepochs. Minima verified; no selection
index bug, no trainingpath repair or promotion.

Conflict cross/scale: `docs/research/notes/NATIVE_CONFLICT_CROSS_20260911.md`. Fixed-input weights
carry hospitalharm2/7; direct replay/scale no significanthospital effect.
14diagonal scores exact; no trained repair/promotion or universal cause claim.

Conflict subgroups: `docs/research/notes/NATIVE_CONFLICT_SUBGROUPS_20260911.md`. Held-out raw-input
unchanged patients also harmed4/7hospital. Not isolated weights explanation;
source transforms/selection differ. No fitting or promotion.

Native conflict28fits: `docs/research/notes/NATIVE_CONFLICT_TRAINING_20260911.md`. Sameclean roles,
own/full x old/clean replay; clean-full vs oldfull and cleanown0/7gainboth.
54bindings verified. No accuracy repair, runtime, calibrator or promotion.

Native replay repair: `docs/research/notes/NATIVE_CONFLICT_REPLAY_20260911.md`. Query-asof source
version/conflict resolution,817histories rebuilt. Preserves earlier lawful
queries; distinct from incrementally mutating state. Neural accuracy pending.

Native schedule exposure: `docs/research/notes/NATIVE_SCHEDULE_CONFLICTS_20260911.md`. Original
hashes/pointers checked,514queries238patients actual unmasked conflicts.
No future retrospective deletion, native repair, accuracy or promotion claim.

Native conflict arrivals: `docs/research/notes/NATIVE_CONFLICT_ARRIVALS_20260911.md`.1551rawgroups
within6h, distributed arrivals evade same-group check; exact exposure pending.
No retroactive deletion, native repair or accuracy claim.

Variable identity: `docs/research/notes/EVENT_VARIABLE_IDENTITY_20260911.md`. Synthetic collision
guard added; actual646120rows unchanged,26tests. Not current gap explanation.

Event summary conflicts: `docs/research/notes/EVENT_SUMMARY_CONFLICTS_20260911.md`. Reject
same-time as-of disagreement as false excursion;274/648query support counts,
25tests. No trained repair/promotion yet.

Joint conflict accuracy: `docs/research/notes/GLUCOSE_CONFLICT_REPAIR_20260911.md`. Rebuilt aligned
input/current/future, tree old/clean training on same clean evaluation. Full
repair gain0/7both; no stable forecast repair or native JEPA conclusion.

Snapshot conflicts: `docs/research/notes/SNAPSHOT_CONFLICTS_20260911.md`. As-of disagreement masks
instead of ID-order selection; resolving revisions tested. Retained values exact;
22tests, no forecast gain yet. Native stream consistency still separate.

Future label conflicts: `docs/research/notes/FUTURE_LABEL_CONFLICTS_20260911.md`. Proven ID-order
truth selection defect fixed; new labels only reject conflicts (Mg0). No accuracy
repair claimed until retraining; old artifacts preserved, helper hash changed.

Mg L1/Huber offsets: `docs/research/notes/MAGNESIUM_LOSS_CALIBRATION_20260911.md`. Frozen predictions,
tune-only patient-equal calibration; full-own0/7both. Not full loss retraining.

Mg auxiliary gradient geometry: `docs/research/notes/MAGNESIUM_AUXILIARY_GRADIENT_20260911.md`.
Real-label weighted auxiliary norm .177primary, mixed alignment10/21opposed.
Distinct from output-retention loss; no fit or accuracy repair established.

Mg auxiliary exposure: `docs/research/notes/MAGNESIUM_AUXILIARY_EXPOSURE_20260911.md`. Exact
selected-epoch sampling replay, objective coefficients and patient support;
not gradients or accuracy repair. No runtime/promotion.

Mg event-weight training: `docs/research/notes/MAGNESIUM_EVENT_WEIGHT_20260911.md`. Same fixed-lag
tree, patient/event equal training, unchanged query evaluation; full-own0/7gain.
Controls reproduced. No stable repair, runtime change or promotion.

Mg shared future outcomes: `docs/research/notes/MAGNESIUM_ANCHOR_OVERLAP_20260911.md`. Stored
predictions, event then patient weighting; common-outcome incrementchange0/7.
No fitting, future-based routing, promotion or causal timing conclusion.

Mg native anchors: `docs/research/notes/MAGNESIUM_NATIVE_ANCHOR_20260911.md`. Support rebuilt at
Mg availability in same K-selected stay universe. Raw tree own/full/support:
full-own0/7source1/7hospital. New labels differ; not paired repair, no promotion.

Mg event summaries: `docs/research/notes/MAGNESIUM_EVENT_HISTORY_20260911.md`. Fixed-lag+6/24h
events, same source selection and roles, baseline scores reproduced. Full-own
and full-lagfull gain0/7 both; no runtime/promotion. Not raw-information proof.

Mg raw history tree: `docs/research/notes/MAGNESIUM_TREE_HISTORY_20260911.md`. Same features,
nonlinear full-own0/7gainboth; no new runtime/promotion.

Mg raw history ridge: `docs/research/notes/MAGNESIUM_RAW_HISTORY_20260911.md`. Full-own0/7gain,
harm7/7source5/7hospital; fixed-lag no latent, not information impossibility.

Mg rate-number ablation7fits: `docs/research/findings/MAGNESIUM_RATE_SUPPORT_FINDINGS_20260911.md`.
Support-vs-own/numeric0/7gainboth,hospitalharm1vsnumeric. No promotion.

Mg fixedpartition14fits: `docs/research/findings/MAGNESIUM_SAMPLER_ORDER_FINDINGS_20260911.md`.
Only sampling/selection vary;0/7gain both,seed7exact; no promotion.

Mg init identity: `docs/research/notes/MAGNESIUM_INITIAL_STATE_20260911.md`.2285tensors exact
across7seeds; correct initialization wording, sampling/partition unresolved.

Mg ensemble variance: `docs/research/notes/MAGNESIUM_ENSEMBLE_VARIANCE_20260911.md`. Whole variance
2.80xown; fixed ensemble MAEgapCIcross0. No averaging source mixedroles/promotion.

Mg unit exposure: `docs/research/notes/MAGNESIUM_UNIT_EXPOSURE_20260911.md`.1eligiblewindow,
hospitalzeroall7; fitting exposure4seeds, no universal-cause proof or runtime.

Mg raw units: `docs/research/notes/MAGNESIUM_RAW_UNITS_20260911.md`.13metadata conflicts in20906,
1future/1current pair; history impact pending, no conversion/runtime change.

Mg future identity/time: `docs/research/notes/MAGNESIUM_LABEL_IDENTITY_20260911.md`.2366raw
future matches, draw spacing5.067-13h;18%equal values not duplicate proof.

Mg dropout7fits: `docs/research/findings/MAGNESIUM_CONTEXT_DROPOUT_FINDINGS_20260911.md`. Transfer
of prior glucose augmentation,0/7gain vs matchedwhole/ownboth. No promotion.

Mg history depth: `docs/research/notes/MAGNESIUM_HISTORY_DEPTH_20260911.md`. Three pre-anchor
count bins all0/7gain bothroles; no history-depth router or promotion.

Mg support contract: `docs/research/notes/MAGNESIUM_SUPPORT_CONTRACT_20260911.md`. Native vs
snapshot inclusion exact7/7;41supported variables,35with30patients. No repair.

Mg patient influence: `docs/research/notes/MAGNESIUM_PATIENT_INFLUENCE_20260911.md`. Benefit/harm
mixed; some leave-one-out mean signs change. No exclusion or gate change.

Mg whole-own shrink: `docs/research/notes/MAGNESIUM_INCREMENT_SHRINK_20260911.md`. Tune-only grid;
blend-own0/7both, blend-whole2/7source0/7hospital. No promotion/runtime.

Own-filter isolation: `docs/research/notes/NATIVE_OWN_EVENT_ISOLATION_20260911.md`. Nonzero inactive
event insertion exact state/uncertainty invariance;2tests. No runtime/training.

Mg-primary auxiliary14fits: `docs/research/findings/MAGNESIUM_AUXILIARY_FINDINGS_20260911.md`.
Real vs primary-only0/7gainboth; shuffled source1gain1harm,hospital0/0.
No repair/runtime/promotion; distinct from glucose-primary auxiliary study.

Mg exact sampler replay: `docs/research/notes/MAGNESIUM_EXPOSURE_20260911.md`.437-453fitpatients,
whole selected99.3-100%patient exposure; final~99%row/eventcoverage. No broad
omission, no convergence/sufficiency proof. No fitting/runtime/promotion.

Matched Mg fit-gap: `docs/research/notes/MAGNESIUM_MATCHED_GAP_20260911.md`. Saved-prediction audit,
whole-own fit6/7gain, tune/evaluation/hospital0/7; no refit/promotion. Artifact
magnesium_matched_gap_20260911.json. Source generalization, not sole external shift.

Matched magnesium-primary7fits: `docs/research/findings/NATIVE_MAGNESIUM_MATCHED_FINDINGS_20260911.md`.
Identical objective/selection, whole vs saved verified own. Source/hospital
gain0/7harm0/7; both persistence7/7. Priority mismatch insufficient explanation;
no runtime/calibration/promotion. Artifact native_magnesium_matched_20260911.json.

Magnesium own-history7fits: `docs/research/findings/NATIVE_MAGNESIUM_OWN_FINDINGS_20260911.md`.
Same evaluated pairs, target-specific primary supervision/selection. Both
persistence7/7, whole-own0/7. Not equal-objective ablation; runtime unchanged.

Native auxiliary constant control: `docs/research/notes/NATIVE_AUXILIARY_CONSTANT_20260911.md`.
Source patient-equal weighted median delta, frozen predictions. Mg/K benefits
survive7/7both roles; own-history control still required. Runtime unchanged.

Native real auxiliary14fits: `docs/research/findings/NATIVE_TRUTH_AUXILIARY_FINDINGS_20260911.md`.
Existing .25real/shuffled loss on native cohort and source/hospital roles.
Mg/K/BUN gains partial; own-target trained control still needed, glucose0/7
shuffled gain. Not a new loss invention or complete repair; runtime unchanged.

Retention curve replay: `docs/research/notes/NATIVE_RETENTION_CURVE_20260911.md`.
Seven exact primary training replays plus per-target tune curves. No strict
all-target non-worsening positive epoch; numerical diagnostic not new gate.
Runtime unchanged; source selection alone not demonstrated repair.

Selection scope: `docs/research/notes/NATIVE_SELECTION_SCOPE_20260911.md`.
Saved initial/final source-tune actual labels; primary improvement coexists
with multi-target deterioration. Per-epoch trajectory unavailable, no claim
selection repair established. No new training/runtime/promotion.

Frozen sharedfuture14fits: `docs/research/findings/NATIVE_FROZEN_FUTURE_FINDINGS_20260911.md`.
Only latent_predictor fixed, encoder/EMA updates retained. No stable whole-own
gain, mixed actual retention outcomes. Not a successful repair or runtime change.

Native block restoration: `docs/research/notes/NATIVE_UPDATE_BLOCKS_20260911.md`.
History/mixers/sharedfuture/private restored separately at fixed source roles.
Sharedfuture implicated in mineral drift, BUN tradeoff; not a safe rollback or
retraining result. Original predictions reproduced; runtime unchanged.

Private update allocation14fits: `docs/research/findings/NATIVE_PRIVATE_UPDATE_FINDINGS_20260911.md`.
Same native own/whole inputs, only three glucose-private modules updated;
52other outputs exact but no stable benefit over original own/full. Privateown
hospital harm4/7; no universal repair, no runtime/promotion.

Same-weight retention: `docs/research/notes/RETENTION_SAME_WEIGHT_20260911.md`.
Source-fit only, two masks at identical checkpoints/batches. Opposition decreases
in both families;42prior probes exact. No optimization or accuracy repair claim.
Artifact `artifacts/retention_same_weight_20260911.json`; runtime unchanged.

Retention conflict localization: `docs/research/notes/RETENTION_CONFLICT_DIAGNOSIS_20260911.md`.
Same source-fit batches, uniform/selected saved checkpoints,42 probes;
primary-retention opposition12/21 vs6/21. Not JEPA-value repetition or solecause
proof. Aggregate `artifacts/retention_gradient_conflict_20260911.json`.
No retraining, new external evaluation, runtime change or promotion.

Selected retention7fits: `docs/research/findings/NATIVE_SELECTED_RETENTION_FINDINGS_20260911.md`.
Reduces BUN/creatinine uniform-retention cost but restores other regressions;
no stable primary whole-own benefit. No runtime/promotion.

Retention reference selection: `NATIVE_RETENTION_SELECTION_20260911.md`.
Source-only initial-vs-persistence gate; bicarbonate/K7/7,Mg6/7. Training of
selected retention not yet performed, no runtime/promotion.

Native retention repair7fits: `docs/research/findings/NATIVE_RETENTION_REPAIR_FINDINGS_20260911.md`.
Mg/phosphate hospital gain7/7; BUN/creatinine harm7/7. Preserving initial mistakes
is the measured tradeoff; primary gain unstable, no runtime/promotion.

Native retention diagnostic: `docs/research/notes/NATIVE_GLUCOSE_RETENTION_20260911.md`.
Unchanged nonprimary value heads, changed shared path, measured other-target
harm. Mixed gains too; no calibration/promotion or sole-root-cause inference.

Native initialization diagnostic: `docs/research/notes/NATIVE_GLUCOSE_INITIALIZATION_20260911.md`.
Initial hospital whole-own7/7 becomes final0/7 despite both improving. Source
initial ordering differs. Refines overfitting interpretation, no new fitting.

Student train-mode14fits: `docs/research/findings/NATIVE_GLUCOSE_MODE_FINDINGS_20260911.md`.
Existing activation/attention dropout, distinct from prior variable-block
dropout. Whole-own0/7both; limited old-model gain, no stable repair/promotion.

Native source-aware history28fits: `docs/research/findings/NATIVE_GLUCOSE_SOURCE_TRAIN_FINDINGS_20260911.md`.
Source vs zero partial gain/harm, no whole-vs-own gain0/7both. Same-width
controls and input bindings verified; not a stable repair or runtime promotion.

Native glucose source input: `NATIVE_GLUCOSE_SOURCE_INPUT_20260911.md`.
Exact native-group source-name features, six contract tests; accuracy contrast
still pending. Not a rerun of current-source offset, no runtime/promotion.

Native pretraining provenance: `docs/research/notes/NATIVE_PRETRAINING_PROVENANCE_20260911.md`.
Referenced current file disjoint after nativeidentity join; historicalhash
missing/counts differ, no leakage or independence overclaim. Runtime unchanged.

Delayed-arrival accuracy: `docs/research/notes/NATIVE_OUT_OF_SEQUENCE_ACCURACY_20260911.md`.
Pre-anchor groups, no fitting; failure also without delayedglucose. Hospital
delayed numericalbenefit6/7, no significantgain. Not causal/safeupdate proof.

Native out-of-sequence observations: `docs/research/findings/NATIVE_OUT_OF_SEQUENCE_FINDINGS_20260911.md`.
V2 uses exact48hwindow for both old/new draws:2,032glucose affectedqueries.
Not accuracy/causation; age already supplied, no defaultdiscard or promotion.

Known glucose source offset: `docs/research/findings/GLUCOSE_SOURCE_OFFSET_FINDINGS_20260911.md`.
42small tune-only fits, current source vs constant/shuffled0/7both roles.
No matchedown gain or predictive repair. No future-source inputs/runtime.

Glucose assay-name provenance: `docs/research/findings/GLUCOSE_SOURCE_SWITCH_FINDINGS_20260911.md`.
Raw labid join, allpaired names resolved,33.06%switch. Hospital numerical harm
concentrates more in switch group, but same-source new-patient failure remains.
Preserve eICU source_test_name in future extraction; no routing/promotion.

Native glucose joint support: `docs/research/findings/GLUCOSE_NATIVE_SUPPORT_FINDINGS_20260911.md`.
Source-only quartile bins/current-age-rate, patient-count support30. Supported
and undersupported groups both gain0/7source/hospital. Descriptive not route,
no high-dimensional support sufficiency or runtime/promotion claim.

Native projection components: `docs/research/findings/NATIVE_PROJECTION_COMPONENTS_FINDINGS_20260911.md`.
Source-only frozen linear decomposition; missing-age large mostlyconstant,
lab centered variation larger. No panel-dependent raw normalization defect;
not a signal-absence/forecast-error explanation. No runtime change.

Native glucose cadence sensitivity: `docs/research/findings/GLUCOSE_NATIVE_CADENCE_FINDINGS_20260911.md`.
Seven fixed weights, non-glucose updates skipped; gain0/7allroles. Original
scores reproduce. Content/cadence removal is off-training-distribution, not
an isolated cause; do not deploy target-only updates into shared state.

Native glucose output calibration: `docs/research/findings/GLUCOSE_NATIVE_CALIBRATION_FINDINGS_20260911.md`.
Tune-only offset/affine tests with matched own controls; whole corrected-own
gain0/7source/hospital. Smaller original signed bias does not imply lower MAE.
Not a global scale repair; no runtime/conformal/promotion changes.

Native glucose fit-gap: `docs/research/findings/GLUCOSE_NATIVE_FIT_GAP_FINDINGS_20260911.md`.
Fixed21fits, 2,000 patient bootstrap; fit whole-own6/7gain becomes0/7gain,
2/7harm on new source. Source failure precedes transport; treatment numeric
increment already unstable at fit. Original score/hash reproduction succeeds.
No fitting/runtime/promotion, not unique-root-cause identification.

Rate-query glucose21fits: `docs/research/findings/RATE_GLUCOSE_FINDINGS_20260911.md`. Own/real/support,
same eligible patients, real-own gain0/7both; real-support1/7hospital only.
Both own and real persistence7/7; no incremental/event promotion.

Rate-query multitarget label support: `docs/research/notes/RATE_MULTITARGET_SUPPORT_20260911.md`.
Same queries/roles, measured pairs across schema; potassium reproduced,
glucose5,438pairs. Not yet trained or promoted; conditional cohort retained.

Native rate-scale audit: `docs/research/findings/NATIVE_RATE_SCALE_FINDINGS_20260911.md`. Typical
normalized variation is nontrivial; tails shift across roles. No scale-collapse
or principal-cause conclusion, no fitted repair/runtime change.

Real-label projection14fits: `docs/research/findings/NATIVE_REAL_PROJECTION_FINDINGS_20260911.md`.
Same synthetic-selected optimizer, original native gain0/7all, source harm3/7.
Known-signal recovery does not imply real forecast repair. No runtime/promotion.

Synthetic observation-projection repair14fits:
`docs/research/findings/NATIVE_PROJECTION_FINDINGS_20260911.md`. Fixed10x first observation Linear LR;
real-support7/7everygroup, hospital artificial MAE .402314vs.580522. Same model,
known-signal uptake improved. Real-label effect untested; no runtime promotion.

Synthetic latent localization: `docs/research/findings/SYNTHETIC_LATENT_FINDINGS_20260911.md`.
Fixed source-only probes, weak sign recovery in belief/context; not a proof
that nonlinear recovery impossible. No neural fit or runtime promotion.

Native same-path artificial control14fits:
`docs/research/findings/NATIVE_SYNTHETIC_FINDINGS_20260911.md`. Small/unstable known-signal increment
recovered3/7source4/7hospital all; frozen whole gain0/7. Limits absent-signal
interpretation, not biological efficacy. No runtime/promotion.

Native treatment fit-gap: `docs/research/findings/NATIVE_FIT_GAP_FINDINGS_20260911.md`. Frozen14models;
real-support fit increment already tiny, six/seven abs paired effects<.0001.
No sole-cause conclusion; full-native-path positive control not yet tested.

Native memory-gradient probe: `docs/research/findings/NATIVE_MEMORY_GRADIENT_FINDINGS_20260911.md`.
Source-fit only, seven frozen checkpoints. Older-event gradients nonzero;
does not establish useful memory, excludes literal total erasure on this probe.
No refit or promotion.

Clock correction14fits: `docs/research/findings/NATIVE_CLOCK_TRAIN_FINDINGS_20260911.md`. Identical
labels/roles, original predictions reproduced; hospital incremental gain0/7,
source1/7 with harm1/7all. Not a stable accuracy repair; persistence7/7 retained.
No runtime/promotion. Replaces the clock-refit-pending status below.

Native clock correction input: `artifacts/native_clock_tokens_20260911.json`.
Only clock columns changed, signed pre-ICU events retained. Three tests pass;
not an accuracy repair until matched retraining, no runtime change.

Native clock diagnostic: `docs/research/findings/NATIVE_CLOCK_FINDINGS_20260911.md`. Positive physical
gap maps tozero69,164/263,668query-history transitions; synthetic late12h token
collision reproduced. Not accuracy causation; clock repair/refit remains pending.

Native event accuracy14fits completed:
`docs/research/findings/NATIVE_RATE_JEPA_TRAIN_FINDINGS_20260911.md`. Same queries/source roles;
native real vs snapshot gain0/7all, hospital bracket harm1/7; versus support
gain0/7all. Persistence7/7 does not establish incremental repair. No runtime.
This supersedes the training-pending entry immediately below.

Native event training integration (not a completed accuracy experiment):
`NATIVE_RATE_JEPA_TRAIN_PROTOCOL_20260911.md`; --native-events in existing runner.
Real/support native updates, unchanged patients/labels/source lab eligibility.
Seven focused tests and CLI pass. Full14fits pending; no runtime promotion.

This is the durable index of completed experiment families. Read the linked
findings before rerunning or extending an experiment. Aggregate JSON reports are
authoritative for exact values; this file records the decision and boundary.

## Release And External Evidence

| Experiment | Decision | Evidence |
|---|---|---|
| Unified source selection | Active policy has 28 globally eligible move / 13 persistence cells; the bundled patient currently activates 27 / 14 after patient-level abstention | `UNIFIED_SOURCE_SELECTION_FINDINGS_20260902.md` |
| eICU to MIMIC replication | Partial cross-database replication; database shift remains target-specific | `CROSS_DATABASE_REPLICATION.md` |
| NWICU external validation | Shared current-state signal reproduced; NWICU is now opened and cannot serve as a new untouched lockbox | `docs/research/findings/NWICU_EXTERNAL_VALIDATION_FINDINGS_20260901.md` |
| Multi-source interval calibration | Source-bound intervals validated for a subset; never reuse an interval across point sources | `MULTISOURCE_INTERVAL_CALIBRATION_FINDINGS_20260902.md` |
| Persistence interval rescue | One interval-only rescue was retained in the research selector; runtime validator remains authoritative | `PERSISTENCE_INTERVAL_RESCUE_FINDINGS_20260902.md` |

## Shared Body State And Coupling

Native filter integration: rate-only activation plus inert padding, seven
focused tests and eight-query forward smoke. `docs/research/notes/NATIVE_RATE_JEPA_ENCODER_READY_20260911.md`.
Same recurrence, no native training or promotion yet; query context unchanged.

Native new-event token cache complete:128,236groups including52,402rate-only;
six contract tests. `docs/research/findings/NATIVE_RATE_JEPA_TOKENS_FINDINGS_20260911.md` and aggregate.
No native training; rate-only groups need explicit update masks. No promotion.

Native schedule V2 materialization: preserved measured_at/unit, private
canonical parquet and ragged NPZ, exact query/group index roundtrip. Matching
V2 aggregate hashes; original counts unchanged. No native neural training.

Native event schedule support: same5,526queries, median42groups versus10fixed
snapshots. `docs/research/findings/NATIVE_RATE_JEPA_SCHEDULE_FINDINGS_20260911.md`, aggregate; four
schedule/input tests pass. Identity-safe grouping only, no forecast trial,
runtime or promotion. Extra events are not proven incremental signal.

Event-only history-mask support:220skipped rate-known positions,133queries;
hospital7patients. `docs/research/notes/RATE_JEPA_EVENT_MASK_SUPPORT_20260911.md`. Insufficient
subgroup inference, no accuracy experiment or promotion. Distinguish ten-lag
grid activation from untested native-event sequence updating.

Event-stage frozen sensitivity: belief/context/target-future latent inspected
under numeric removal; perturbation already small at belief. See
`docs/research/findings/RATE_JEPA_EVENT_STAGES_FINDINGS_20260911.md` and aggregate. Not solely a final
decoder loss, not causality or a repair. No fitting/runtime/promotion.

Frozen event-path utilization: seven checkpoints reproduce; event weights
nonzero .028-.133, numerical-channel removal hospital forecast shift~.001.
`docs/research/findings/RATE_JEPA_EVENT_PATH_FINDINGS_20260911.md` and aggregate. Weak usage not
disconnection; off-manifold, no causal/efficacy or promotion claim.

Same-filter event input14fits: `docs/research/findings/RATE_JEPA_EVENTS_FINDINGS_20260911.md`, protocol,
aggregate. Real-vs-original1/7source,0/7hospital; support gain0/7. Near-identical
predictions, component tests2pass. Record-time insulin only, lab-supported
steps; no treatment-irrelevance conclusion or promotion. Event-protocol hash
omission documented and future reporting fixed, original artifact preserved.

Full rate-query JEPA history event alignment:46,657snapshots,30,993supported
rates,1,939truezeros. `docs/research/findings/RATE_JEPA_EVENT_HISTORY_FINDINGS_20260911.md` and aggregate.
Same query/labels, separate treatment channels, no encoder training yet.
Record-time only, not independent bedside availability. No runtime/promotion.

Fixed seven-seed rate-JEPA ensemble, common held hospitals only:
`docs/research/findings/RATE_JEPA_ENSEMBLE_FINDINGS_20260911.md`, protocol and aggregate. Whole improves
persistence, own comparison CI crosses0 in both groups. No cross-seed source
evaluation averaging, no fitting/runtime/calibration/promotion. Partial
variance mitigation, not completed generalization repair.

Rate-query input-integrity check: current hashes match body/query/labels and
seven partitions; role patient overlap and duplicate queries zero. See
`docs/research/notes/RATE_JEPA_INPUT_INTEGRITY_20260911.md`. Does not certify warm-start pretraining
independence or infusion availability. Read-only, no promotion.

Rate-query sampler replay (no fitting):86.95-90.47%rows,90.52-92.30%future
events sampled, selected epochs4-29. `docs/research/findings/RATE_JEPA_SAMPLING_FINDINGS_20260911.md`
and matching aggregate. Unlike low-coverage glucose budget study, does not
motivate automatic epoch extension. Not convergence or sufficient-data proof.

Frozen rate-JEPA fit/tune/evaluation/hospital comparison:
`docs/research/findings/RATE_JEPA_FIT_GAP_FINDINGS_20260911.md` and aggregate. Seven support weights,
hash-bound own/whole, evaluation scores reproduce. Numerical increment
gain5/7fit,1/7new source,2/7hospital; role-size caveat retained. No fitting,
new external data, runtime or promotion. Do not equate attenuation with a
proved unique overfitting mechanism.

Train-matched rate-query support-only JEPA:7source-only32epoch fits, same
partitions and teacher; hash-bound own/whole predictions. Support-own gain5/7
source,0/7hospital; whole-support1/7source,2/7hospital all,0/7bracket. See
`docs/research/findings/RATE_ANCHOR_JEPA_SUPPORT_FINDINGS_20260911.md`, protocol and aggregate
`artifacts/rate_anchor_jepa_support_20260911.json`. Not numerical equivalence,
causality or a runtime repair. No calibration/promotion.

Rate-query frozen JEPA component diagnostic completed, seven weights with exact
full prediction reproduction. Same source/hospital patients; non-target value
neutralization harms5/7source and3/7hospital, age standardization has tiny
effects. `docs/research/findings/RATE_JEPA_COMPONENT_FINDINGS_20260911.md` and
`artifacts/rate_jepa_components_20260911.json`; runner
`scripts.audit_rate_jepa_components`. Off-manifold dependence does not establish
causality or replace train-matched support control. No fitting/runtime/promotion.

Expanded insulin/potassium paired inventory:3373sequence+lab pairs,1562patients,
54hospitals. `docs/research/notes/INSULIN_POTASSIUM_PAIR_SUPPORT_20260911.md`, JSON and private cache
hashes. Feasible expansion, not accuracy evidence. Roles pending before training;
event-time treatment caveat retained. No runtime or promotion change.

Full-table insulin inventory (outcome-free):5152sequencepatients,4909changing,
67hospitals. `docs/research/notes/EICU_INSULIN_SEQUENCE_INVENTORY_20260911.md` and aggregate.
Current bounded subset does not exhaust raw support; future-label matching
and patient/hospital roles still required before expansion. No accuracy claim.

Frozen rate sequence subgroup: exact pooled reproduction; known-history group
1-10evaluationpatients/seed, CI unsupported7/7. Nonzero insulin range274windows,
change259. `docs/research/findings/POTASSIUM_RATE_SUBGROUP_FINDINGS_20260911.md`. No fitting/promotion;
row-count coverage does not imply independent patient support.

Same-label rate histories:28fixed source fits,341insulin sequences,16diuretic.
Own gain1/7baseline/placebo; full0/7. `docs/research/findings/POTASSIUM_RATE_HISTORY_FINDINGS_20260911.md`
with protocol/JSON. Retrospective, no full dose/availability or promotion claim.

All-action preflight V2 adds six action supports and patient counts to
`docs/research/findings/TREATMENT_LEAF_SUPPORT_FINDINGS_20260911.md`; only insulin/diuretics satisfy
necessary minleaf exposure support7/7, other four0/7. Integrated into future
treatment runner reports. Not efficacy/power or model promotion.

Rare-treatment comparator positive control: potassium fit counts8-19 vs tree
minleaf30, two eval roles zero support. Perfect synthetic20/200binary signal
unlearnable at30, recovered at10. `docs/research/findings/TREATMENT_LEAF_SUPPORT_FINDINGS_20260911.md`.
Corrects action-specific interpretation; no real predictive repair/promotion.

Order-issuance experiment completed28fits: unchanged source pairs and tree
settings, count/recency only; no final cancellation/dose/start/stop information.
Own/full vs original and shuffled0/7gain. `docs/research/findings/POTASSIUM_ORDER_CONTEXT_FINDINGS_20260911.md`
and matching protocol/JSON. Partial numerical own gain, not stable repair.
Retrospective timing; no runtime promotion or pristine validation.

Medication-vs-infusion lexical audit:140missed brand orders repaired via
K-Dur/Klor-Con aliases, window recall727->739. Infusion unchanged20windows.
`docs/research/findings/POTASSIUM_MEDICATION_COVERAGE_FINDINGS_20260911.md`, original/V2 aggregates.
No accuracy claim; cancellation may be finalized later, orders not delivery.
This corrects extractor recall but not proven forecast failure.

Fixed source infusion-context refits:28fits, unchanged6174pairs and seven
patient partitions; own/full plus real/shuffled snapshots. Own baseline gain1/7,
harm1/7; full gain0/7; placebo gain0/7both. No stable repair. Full treatment
splits0/0/2/0/3/3/7, none potassium-repletion. Retrospective event-time only.
`docs/research/findings/POTASSIUM_TREATMENT_CONTEXT_FINDINGS_20260911.md`, matching protocol/JSON,
runner `scripts.audit_potassium_treatment_context`. No runtime changes.

Treatment rate semantics follow-up (V2): existing parser recognizes mL/h but
drug-compatible whitelist intentionally excludes it for potassium/bicarbonate.
Separate solution-flow channel retains20/36 query windows, magnesium13.
`docs/research/findings/POTASSIUM_TREATMENT_SUPPORT_FINDINGS_20260911.md`, V2 aggregate and
`event_treatment.rate_channels`. No inferred concentration/availability,
accuracy claim, model training or deployment change.

Potassium infusion support audit:
`docs/research/findings/POTASSIUM_TREATMENT_SUPPORT_FINDINGS_20260911.md`, aggregate JSON and runner.
Same6174queries/1309patients. Insulin rate497windows; potassium events20 but
recognized rate0. Scoped prior helpers do not form a complete potassium ledger;
eventoffset is not independent availability. No fitting/outcome association,
external read, or runtime change. Input gap quantified, not accuracy repaired.

Unbounded-above scalar amplitude diagnostic:
`docs/research/findings/POTASSIUM_EVENT_GAIN_FINDINGS_20260911.md`, dedicated protocol and aggregate.
Source tune weighted LAD gain, frozen event-window trees, disjoint evaluation.
Full amplification selected6/7; all arms improve originals0/7. Full-own1/7.
Not a uniform-amplitude repair; distinct from prior [0,1] adoption. No external
reads, interval recalibration, weight changes or runtime promotion.

Frozen event-window change decomposition:
`docs/research/findings/POTASSIUM_EVENT_CHANGE_FINDINGS_20260911.md`, protocol, V2 aggregate JSON.
Seven frozen source splits; full-own gains0/7 large,5/7 smaller actual changes.
Large direction .906 but movement .401 vs actual1.090. Fixed floating-point
quantile ties; pooled scores unchanged. Future-defined groups are descriptive,
not deployable selectors. No fitting, external reads, or runtime promotion.

Within-window raw event summaries:
`docs/research/findings/POTASSIUM_EVENT_WINDOWS_FINDINGS_20260911.md`, protocol, aggregate JSON.
21source fits; same pairs/partitions. Full-vs-original1/7, full-own3/7,
full-support5/7. Real information omission, not established principal cause.
No external read, calibration or runtime promotion.

Potassium-native availability query refit:
`docs/research/findings/POTASSIUM_NATIVE_ANCHOR_FINDINGS_20260911.md`, protocol and aggregate JSON.
21fits, inherited patient partitions. Full-own4/7source0/7external; full-vs-old
same-row gain0/7all roles. Not a query-selection repair; own source harms2/7.
No runtime, calibration or promotion changes.

Frozen nonlinear potassium transfer:
`docs/research/findings/POTASSIUM_TREE_TRANSFER_FINDINGS_20260911.md`, protocol and aggregate JSON.
21selected refits reproduce; full-own0/7both external roles, both persistence7/7.
No external tuning or runtime promotion. `docs/research/notes/ROOT_CAUSE_EVIDENCE_20260911.md`
separates proven defects from residual incremental-context instability.

Potassium nonlinear matched raw history:
`docs/research/findings/POTASSIUM_TREE_HISTORY_FINDINGS_20260911.md`, protocol and aggregate JSON.
63histogram-tree fits; full-own3/7gain0/7harm, full-support2/7gain, full-JEPA2/7gain.
Partial nonlinear evidence, no external/interval/runtime promotion.

Potassium matched raw history:
`docs/research/findings/POTASSIUM_RAW_HISTORY_FINDINGS_20260911.md`, protocol, aggregate JSON.
105Ridge fits; float64 authoritative. Full-own0/7gain3/7harm; full-JEPA0/7gain2/7harm.
Own persistence7/7. No latent-bypass repair, nonlinear-information claim or promotion.

Potassium train-matched support-only control:
`docs/research/findings/POTASSIUM_SUPPORT_ONLY_FINDINGS_20260911.md`, protocol and aggregate JSON.
14fits64epochs; own logs/predictions exact. Support-only vs full0/7gain0/7harm,
vs own0/7gain1/7harm. Not equivalence or workflow-only proof; no promotion.

Potassium component diagnosis:
`docs/research/findings/POTASSIUM_COMPONENT_FINDINGS_20260911.md`, protocol and aggregate JSON.
Seven fixed weights reproduce; block removal harms5/7, numeric1/7, age tiny.
No repair/promotion; not proof of workflow-only learning. Training-matched
support-only numeric-content control has now completed above.

Whole-source three-way evaluation:
`docs/research/findings/POTASSIUM_THREE_WAY_FINDINGS_20260911.md`, protocol and aggregate JSON.
14fits,785/262/262 per seed; evaluation excluded from checkpoint selection.
Own/whole persistence7/7, whole-own1/7gain1/7harm. No stable context repair,
no external/calibration/runtime change. Source re-partition is not pristine.

Source patient influence:
`docs/research/findings/POTASSIUM_PATIENT_INFLUENCE_FINDINGS_20260911.md` and aggregate JSON.
21comparisons/42MAEs exact; no single-case sign reversal. Original benefit50-59%.
194unique tune patients across500appearances; source expansion fit-only.
No metric-weight bug, no promotion. Whole-source evaluation now completed above.

Source event-balanced intervention:
`docs/research/findings/POTASSIUM_EVENT_BALANCED_FINDINGS_20260911.md`, protocol and
`artifacts/potassium_event_balanced_20260911.json`.14matched64epoch fits;
own/whole gain over original0/7both, whole-own0/7, whole-persistence7/7.
No source incremental repair, no external/calibration/runtime changes.

Source future-event reuse/provenance:
`docs/research/findings/AUXILIARY_EVENT_REUSE_FINDINGS_20260911.md`. Rebuilt53label columns exact;
future IDs now explicit, distinct equal-valued events retained. K24.52% excess
rows over unique outcomes; no claim this proves root cause or accuracy repair.

Numerical loss alignment intervention:
`docs/research/findings/POTASSIUM_MAE_FINDINGS_20260911.md`.14same64epoch fits, L1 replaces SmoothL1
only; own/whole improve0/7each vs old loss, no significant harms. No runtime
promotion or external evaluation. Do not repeat as an untested loss mismatch.

Local value/JEPA gradient diagnosis:
`docs/research/findings/OBJECTIVE_ALIGNMENT_FINDINGS_20260911.md`.63source probes,21checkpoints;
JEPA norm1-2.5% of value medians, combined nearly value-aligned. Strong local
cancellation not supported; no unique cause or weight-increase recommendation.

Potassium fixed-weight age diagnostic:
`docs/research/findings/POTASSIUM_FRESHNESS_FINDINGS_20260911.md`. Pooled exact reproduction; <=1h,
1-3h,3-6h whole-own0/7both roles. MIMICold persistence0/7. Freshness alone
does not explain/repair the extra-context gap. No threshold selected/deployed.

Frozen target-matched potassium transfer:
`docs/research/findings/POTASSIUM_EXTERNAL_FINDINGS_20260911.md`. Same14weights, rebuilt pairs on
207hospital/859MIMIC patients. Whole persistence7/7both, whole-own0/7both;
MIMIC harm2/7, marked care-unit heterogeneity. No promotion or model edits.

Target-matched neural own/whole control:
`docs/research/findings/POTASSIUM_MATCHED_ENCODER_FINDINGS_20260911.md`.14fits,64epochs, same source
pairs, target-specific selection. Whole lower MAE7/7 but CI gain0/7; both beat
persistence6/7. No new external evaluation or runtime promotion.

Source-only auxiliary own-history control:
`docs/research/findings/AUXILIARY_OWN_HISTORY_FINDINGS_20260911.md`. All15targets x7seeds x2ridge
controls on identical measured pairs. Potassium1/7, glucose0/7, creatinine2/7
significant JEPA wins; no target7/7. No new external read or promotion.

Latest source-only auxiliary movement/centering diagnostic:
`docs/research/findings/AUXILIARY_DYNAMICS_FINDINGS_20260911.md`; fixed 21 checkpoints versus source-fit
patient-median delta. Potassium6/7 above control, several other targets0/7.
Retrospective subgroup tradeoff is not a prospective gate. No promotion.

| Experiment | Decision | Evidence |
|---|---|---|
| Strict shared-state reconstruction | Partial current-state representation validated; not universal future dynamics | `docs/research/findings/SHARED_BODY_STATE_RECONSTRUCTION_FINDINGS_20260830.md` |
| 35-cell multi-source shared state | Broad candidate coverage, but corrected external provenance reduced full passes | `docs/research/findings/MULTISOURCE_SHARED_STATE_35_CELL_FINDINGS_20260902.md` |
| Masked generalization and source repair | Several readout, mixture, ensemble, source-cohort, and robust variants tested; no universal promotion | `docs/research/findings/MASKED_GENERALIZATION_FINDINGS_20260903.md`, `docs/research/findings/MASKED_SOURCE_COHORT_REPAIR_FINDINGS_20260903.md`, `docs/research/findings/MASKED_ROBUST_READOUT_FINDINGS_20260903.md` |
| Masked distributional expert | Candidate-only; distribution modeling did not remove cross-domain instability | `docs/research/findings/MASKED_DISTRIBUTIONAL_EXPERT_FINDINGS_20260904.md` |
| Organ exchange | Dense, sparse, permuted, and no-exchange variants tested; cross-organ routing remains edge-specific and candidate-only | `docs/research/findings/ORGAN_EXCHANGE_FINDINGS_20260906.md` |
| Input support parity | Support and observation-contract differences audited; parity alone did not create stable transfer | `docs/research/findings/INPUT_SUPPORT_PARITY_FINDINGS_20260906.md` |

## Persistent State And Cascade

| Experiment | Decision | Evidence |
|---|---|---|
| Persistent whole-body physiology state | Architecture and point gates tested; candidate-only where full external gate failed | `docs/research/findings/PERSISTENT_PHYSIOLOGY_STATE_FINDINGS_20260828.md`, `docs/research/findings/PERSISTENT_PHYSIOLOGY_STATE_POINT_GATE_FINDINGS_20260828.md` |
| Corrected persistent-state promotion | Corrected state semantics retained only where exact promotion criteria passed | `docs/research/findings/PERSISTENT_PHYSIOLOGY_STATE_CORRECTED_PROMOTION_FINDINGS_20260829.md` |
| Cross-target nowcast | A small subset of same-anchor completion targets passed strict leakage-safe gates | `docs/research/findings/CROSS_TARGET_NOWCAST_FINDINGS_20260827.md` |
| Nowcast-to-forecast cascade | No general promotion; nowcast is not automatically allowed to feed future forecasts | `docs/research/findings/NOWCAST_FORECAST_CASCADE_FINDINGS_20260828.md` |

## Routing, Personalization, And Calibration

| Experiment | Decision | Evidence |
|---|---|---|
| Kidney state-conditioned / utility-conditioned routing | State-dependent signal exists, but complete routing promotion remains limited | `STATE_CONDITIONED_KIDNEY_FINDINGS_20260904.md`, `UTILITY_CONDITIONED_KIDNEY_FINDINGS_20260904.md` |
| Distributional kidney risk | Relative and absolute risk variants tested; not a universal fix for persistence selection | `docs/research/findings/DISTRIBUTION_RISK_KIDNEY_FINDINGS_20260904.md`, `DISTRIBUTION_RISK_KIDNEY_RELATIVE_FINDINGS_20260904.md` |
| Expected-error routing | Correct decision target tested; gains remained small and domain-dependent | `docs/research/findings/EXPECTED_ERROR_ROUTING_FINDINGS_20260905.md` |
| Forecast-aware routing | Added forecast context did not solve cross-database amplitude and state mismatch | `docs/research/findings/FORECAST_AWARE_ROUTING_FINDINGS_20260905.md` |
| Pre-anchor context routing | POE, bedside, treatment, and history context partially repaired selector input; no broad promotion | `PREANCHOR_CONTEXT_ROUTING_FINDINGS_20260905.md`, `PREANCHOR_CONTEXT_ROUTING_FINDINGS_20260905_V2.md` |
| Amplitude calibration | Tested correction of under/over-sized change; candidate-only | `docs/research/findings/AMPLITUDE_CALIBRATION_FINDINGS_20260905.md` |
| Change-behavior evaluation | Separated stable, dynamic, direction, magnitude, and uncertainty behavior | `docs/research/findings/CHANGE_BEHAVIOR_FINDINGS_20260906.md` |

## Future Change And Representation

| Experiment | Decision | Evidence |
|---|---|---|
| Prospective change head | No new promotion; point improvement did not establish early-change recognition | `docs/research/findings/PROSPECTIVE_CHANGE_FINDINGS_20260907.md` |
| Quantile change and CQR | No new promotion; more quantiles did not improve the learned median physiology | `docs/research/findings/QUANTILE_CHANGE_FINDINGS_20260907.md` |
| Observation representation repair | Explicit masks/ages, raw inputs, frozen latent, and matched placebos all tested; no complete point/risk/interval pass | `docs/research/findings/REPRESENTATION_REPAIR_FINDINGS_20260907.md` |
| Cross-variable pre-anchor history | Measurement-pure 1-24h histories failed before the encoder for glucose/MAP at 3h; no end-to-end retraining justified for these targets | `docs/research/findings/CROSS_VARIABLE_HISTORY_FINDINGS_20260907.md` |
| Whole-catalog cross-variable history screen | Corrected current/history placebos and support-balanced hospital folds evaluated 37/39 continuous cells; 0 earned observation-aware encoder retraining eligibility, 2 platelet cells lacked hospital support | `docs/research/findings/CROSS_VARIABLE_HISTORY_CATALOG_FINDINGS_20260908.md` |
| Near-miss portfolio and state routing | Four predeclared heterogeneous near-miss cells had strong outcome-oracle upper bounds, but fixed convex portfolios and legal pre-anchor expected-error routing both passed 0/4; errors were highly correlated and source choice was not prospectively identifiable | `docs/research/findings/NEAR_MISS_PORTFOLIO_FINDINGS_20260908.md` |
| Anchor-aligned cross-database signal (interpretation superseded) | Original results retained, but temporal-only placebo, 2-10h actual labels and mixed release features did not isolate the claimed causes; no promotion | `docs/research/findings/ANCHOR_ALIGNED_SIGNAL_FINDINGS_20260908.md` |
| Corrected timing and block attribution | Actual 5-7h primary; separate current/temporal/full-block controls and pure latent. Latent beats swapped-block placebo in pooled source/external tests, yet MIMIC persistence remains better; no full diagnostic pass or runtime change. Broad-window sensitivity also run | `docs/research/findings/ANCHOR_ALIGNED_SIGNAL_CORRECTED_FINDINGS_20260908.md`, `artifacts/anchor_aligned_signal_corrected_narrow_20260908.json`, `artifacts/anchor_aligned_signal_corrected_broad_20260908.json` |
| Exact measurement-anchor target-only ablation | Same eICU/MIMIC patients, actual creatinine measurement anchors, exact 5-7h labels and equal-capacity target-only/whole-body frozen-JEPA heads. Fresh anchoring lowers all methods' error, but whole-body has no stable incremental or external gain and MIMIC persistence remains best; no promotion | `docs/research/findings/MEASUREMENT_ANCHOR_ENCODER_FINDINGS_20260909.md`, `artifacts/measurement_anchor_encoder_20260909.json` |
| Prequential continual self-learning | Frozen patient state, naive online updates, short/long replay and uniform replay tested with equal update budgets on exact creatinine anchors. Replay improved the frozen adapter and reduced source forgetting, but never beat persistence and retained order-dependent MIMIC harm; candidate-only | `docs/research/findings/CONTINUAL_SELF_LEARNING_FINDINGS_20260909.md`, `artifacts/continual_self_learning_20260909_v2.json` |
| Continual learning on a validated cell | Exact `glucose@6h` anchors tested whether self-learning can improve a source that already transfers. Patient-scoped residual state improved repeat predictions by 2.27 mg/dL in 7/7 orders without unrelated-patient harm, but did not significantly beat persistence on the repeat slice. All global weight-update/replay arms caused 7/7 retention, hospital and MIMIC harm; no promotion | `docs/research/findings/CONTINUAL_VALIDATED_CELL_FINDINGS_20260909.md`, `artifacts/continual_validated_glucose_6h_20260909_v2.json` |
| Event-aware patient-state validity | Same frozen `glucose@6h` JEPA compared fixed history, unconditional resolved-error updates and an anchor-known measurement/treatment event gate. Event-aware passed only 1/7 eICU hospital and 0/7 MIMIC comparisons versus frozen, never beat state-only or shuffled-event controls, and its eICU treatment-transition direction gain did not transfer. Wrong-patient state was harmful, so identity matters but event-conditioned signed correction is not stable; candidate-only | `EVENT_AWARE_PATIENT_STATE_PROTOCOL_20260909.md`, `EVENT_AWARE_PATIENT_STATE_FINDINGS_20260909.md`, `artifacts/event_aware_patient_state_glucose_6h_20260909_v3.json` |
| External patient-state replay | Required a genuinely resolved prior outcome, selected time decay/trend on eICU, calibrated on disjoint eICU patients, and replayed each MIMIC patient prospectively. Time-aware state narrowed MIMIC intervals but point correction harmed changing patients and lost to frozen/simple predictions; coverage was slightly above the accepted band. Candidate-only, no runtime change | `docs/research/findings/PATIENT_STATE_EXTERNAL_FINDINGS_20260909.md`, `artifacts/patient_state_external_glucose_6h_20260909.json` |
| Patient-state failure localization | Corrected inconsistent placebo support and causal donor timing, deduplicated shared future measurements, ablated level/decay/slope, isolated uncertainty with a fixed point source, and compared target history/raw cross-history/JEPA latent on identical roles. Personal level did not transfer; decay caused significant external harm; personal scale was narrower but not reliably calibrated; raw and latent cross-body information did not stably beat target history. Diagnostic-only, no runtime change | `docs/research/findings/PATIENT_STATE_DIAGNOSTIC_FINDINGS_20260909.md`, `artifacts/patient_state_diagnostics_glucose_6h_20260909.json` |
| Complete recurrent event-state loop | Frozen `glucose@6h` JEPA plus a patient-scoped recurrent state directly decoded signed residuals from anchor-known measurements, freshness, changes, pre-anchor treatment and resolved outcomes. It beat persistence 7/7 in eICU hospital-heldout but 0/7 in MIMIC, beat frozen JEPA 0/7 in both, and did not beat reset-state or shuffled-event controls. Changed-row source benefit traded against stable-row harm and did not transfer; candidate-only | `docs/research/protocols/RECURRENT_EVENT_STATE_LOOP_PROTOCOL_20260909.md`, `docs/research/findings/RECURRENT_EVENT_STATE_LOOP_FINDINGS_20260909.md`, `artifacts/recurrent_event_state_glucose_6h_20260909_v1.json` |

## Raw Lab Identity and Availability (2026-09-10)

Follow-up to duplicate-ingestion sensitivity, using native glucose labIDs and
recorded availability on the same cached patients. Exact source-version dedup
removed zero role rows and retained equal-value distinct IDs. All old scored
current/future raw pairs matched, but availability was after anchor in644/846
source,614/797 hospital and2349/2349 MIMIC rows. With strict availability, MIMIC
has zero pairs: seven-seed accuracy training did NOT start and no improvement
or promotion can be claimed. Native-event dedup/query separation implemented
and unit-tested; not a whole-body raw-event rebuild. eICU final revised entry
time is conservative, not original provisional availability. Next reconstruct
anchors, inputs and future labels on availability time together. No runtime
change. `docs/research/findings/RAW_LAB_IDENTITY_FINDINGS_20260910.md`,
`artifacts/raw_lab_identity_20260910.json`.

## Update Invariance and Feedback Attribution (2026-09-10)

Seven source-fitted own-feedback heads, identical patients and fixed weights.
Duplicating encoded events without new information changed later predictions:
4 copies shifted MIMIC predictions by 1.520 mg/dL mean absolute, with significant
MAE harm in 5/7 seeds. This localizes duplicate-ingestion sensitivity, not actual
duplicate prevalence or the sole transfer-failure cause. Withholding only newly
resolved feedback did not help (0/7 significant benefit); mean MIMIC MAE worsened
30.940 ->30.963. Accumulated historical-error harm remains unisolated. Treatment
snapshot subgroup is observational and the MIMIC no-change subgroup unsupported.
No promotion, runtime or interval change. See `docs/research/findings/UPDATE_INVARIANCE_FINDINGS_20260910.md`
and `artifacts/update_invariance_20260910.json`. Next distinguish real event
identities/available time from re-documentation before changing state updates.

## Event-State Contract Diagnosis (2026-09-10)

- Question: do similar frozen/recurrent scores establish a shared bottleneck?
- Evidence: refreshed CodeGraph, existing seven-seed glucose report, and three
  synthetic probes (no new patient training or external evaluation).
- Results: focal input changes with unrelated patients; repeated measurements
  are not counted as new; rate-only changes leave treatment encoding unchanged.
  Update streams are filtered by future labels; feedback is frozen-source error.
- Decision: diagnostic-only, not proof of one root cause or improved MAE. The
  earlier "complete loop" wording is superseded by
  `docs/research/notes/EVENT_STATE_CONTRACT_DIAGNOSIS_20260910.md`.
- No release weights, registry, intervals or serving behavior changed.

## Event-State Contract Repair (2026-09-10)

Before further continual-learning proposals, see the completed repair:
`docs/research/findings/EVENT_STATE_REPAIR_FINDINGS_20260910.md`. Two final 7-seed eICU/MIMIC runs
compared scored-only versus all cached-anchor updates, frozen-error versus
candidate-error feedback, reset and shuffled controls. Input invariants were
repaired, but both learned variants beat frozen in 0/7 in both domains. Full
MIMIC MAE 30.985/30.940 versus frozen 28.944; no promotion, interval or runtime
change. Final artifacts: `artifacts/recurrent_event_state_repaired_full_20260910_final.json`
and `artifacts/recurrent_event_state_repaired_scored_20260910_final.json`.
Raw event availability and matched update-frequency replay remain untested.

## Waveform Experiments

| Experiment | Decision | Evidence |
|---|---|---|
| Raw waveform JEPA | No promoted continuous forecast cells | `docs/research/findings/RAW_WAVEFORM_JEPA_FINDINGS_20260826.md` |
| Waveform event JEPA | Event-oriented direction tested; remains research-only | `docs/research/findings/WAVEFORM_EVENT_JEPA_FINDINGS_20260826.md` |
| Waveform promotion methods | Multi-scale and alternative summaries tested; no release promotion | `docs/research/findings/WAVEFORM_PROMOTION_METHODS_FINDINGS_20260827.md` |
| VitalDB waveform lockbox | External waveform result recorded; does not establish broad ICU waveform gain | `docs/research/findings/VITALDB_WAVEFORM_LOCKBOX_FINDINGS_20260827.md` |

## Important Negative Findings

- Larger latent capacity did not produce a monotonic improvement.
- More readout capacity, more quantiles, clustering variants, and unrestricted
  whole-body mixing did not solve generalization.
- Same-anchor nowcast success does not authorize forecast feedback.
- Marginal 90% coverage does not compensate for point failure, care-unit
  miscalibration, or excessive width.
- A source that wins in eICU can lose in MIMIC-IV or NWICU; source ordering is
  not invariant across databases.
- Personal target history often matches or beats the frozen shared latent.
- Sparse tests may contain current-state information without supporting a
  reliable fixed-horizon future value.
- Combining near-miss candidates can enlarge an outcome-aware oracle while
  producing no deployable gain. In the tested cells, candidate errors were
  highly correlated and legal pre-anchor routing agreed with the oracle only
  about 19-27% of the time.

## Still Unfinished

- End-to-end observation-aware encoder training with a prespecified source-side
  gain. Existing catalog readouts failed eligibility; corrected narrow timing
  shows partial latent placebo evidence, not proof that tabular information is
  absent. Own-target-only latent ablation and same-patient timing/freshness
  diagnostics remain needed before attributing that evidence to other organs.
- Exact organ-store supervision after a complete unit-normalized intake/output
  flux ledger exists.
- A new untouched external lockbox for any method tuned after MIMIC-IV and
  NWICU results were viewed.
- General cross-organ future dynamics with stable no-edge ablation evidence.
- Reliable irregular-event forecasts for sparse hormones, troponin, INR, tumor
  markers, and similar high-level tests.
- Longitudinal healthy and chronic-disease histories outside one hospitalization.
- Database-invariant prequential continual adaptation that beats persistence
  without retention or external harm. Patient-scoped residual state is the
  closest current result; global online weight updates are rejected.
- Causal treatment-effect estimation; RCT work remains a separate chapter.

## Update Template

For each new completed experiment, add:

1. Question and frozen protocol.
2. Development, calibration, validation, and lockbox roles.
3. Comparator and capacity-matched placebo.
4. Patient, hospital/care-unit/time, interval, and negative-transfer results.
5. Decision: promoted, partial, candidate-only, or rejected.
6. Aggregate artifact and fitted-weight paths.
7. Whether runtime weights, registry, calibrator, or serving behavior changed.
# Availability-Anchored Glucose (2026-09-10)

`AVAILABILITY_ANCHOR_PROTOCOL_20260910.md` and
`docs/research/findings/AVAILABILITY_ANCHOR_FINDINGS_20260910.md`: completed source-trained current-only
versus legal-history ridge versus persistence after rebuilding queries, inputs
and future labels at result availability. MIMIC0 old strict pairs becomes6,616
new pairs/889 patients. History beats persistence7/7 seeds in eICU hospital and
MIMIC; incremental history vs current-only passes7/7 and3/7 respectively.
Aggregate: `artifacts/availability_anchor_20260910.json`. No neural retraining,
interval calibration, full promotion or runtime change. Full whole-body
availability reconstruction is still untested; do not repeat this bounded
glucose diagnostic as if it were that experiment.

## Availability-Safe Full JEPA Schema (2026-09-10)

`AVAILABILITY_JEPA_PROTOCOL_20260910.md` / `docs/research/findings/AVAILABILITY_JEPA_FINDINGS_20260910.md`:
re-encoded own/full labs with true masks/ages and old-mask control using frozen
JEPA; all use identical newly rebuilt glucose queries/labels.42/40 lab variables
supported, not complete bedside/treatment data. Frozen truthful full head beats
persistence in both domains, but whole vs own fails external significance.
Seven source-refitted311-feature ridge full heads beat own/frozen0/7 in each
domain and strongly harm MIMIC. Scale/support drift documented separately.
Artifacts `artifacts/availability_jepa_20260910_v2.json` (final unit aliases),
`artifacts/availability_jepa_20260910.json` (v1 retained),
`artifacts/availability_jepa_shift_20260910.json` (diagnostic only).
No encoder retraining, new interval, full gate or runtime change. This tests
frozen encoder compatibility, not whether end-to-end availability-aware JEPA
can learn. Do not repeat a frozen readout sweep as that untested experiment.

## Availability-Aware End-to-End Pilot (2026-09-10)

`AVAILABILITY_ENCODER_PROTOCOL_20260910.md` / `docs/research/findings/AVAILABILITY_ENCODER_FINDINGS_20260910.md`:
three-seed four-arm own/full x frozen/trainable context trial completed,8 epochs.
Real context gradients,77 changed context tensors in selected end-to-end weights,
EMA future-delta supervision. Whole-vs-frozen significant1/3 per domain;
whole-vs-own0/3; no stable incremental whole-body gain. Direct neural head,
not external311 readout. Report `artifacts/availability_encoder_20260910.json`.
Candidate checkpoints retained locally with hashes. No runtime/promotion/interval
change. End-to-end warm-start is now tested; from-scratch, full-budget and formal
seven-seed gate are not. Do not describe all encoder retraining as untried.

## Direct Encoder Bottleneck Diagnosis (2026-09-10)

`docs/research/findings/ENCODER_BOTTLENECK_FINDINGS_20260910.md`: fixed pilot weights, same data,
no refit. Direct glucose decoder has a mathematical30.023mg/dL delta cap;
35.35%-46.29% of validation truths exceed it. Whole MIMIC predictions saturate
more and correlate less well than own. Same whole weights with own-only input
improve only1/3 MIMIC seeds, excluding a universal input-deletion remedy.
Selected-checkpoint source gradient probes find whole JEPA norms5.3%-7.8%
of value norms, positive cosine. Cap is proven; sole cause of generalization
failure is not. Report `artifacts/encoder_bottleneck_20260910.json`.
No training, calibration, formal gate or runtime changes.

## Factorial Decoder / Gradient Repair (2026-09-10)

`ENCODER_REPAIR_PROTOCOL_20260910.md` / `docs/research/findings/ENCODER_REPAIR_FINDINGS_20260910.md`:
24 source-fit trainings, same availability-safe glucose cohort, own/full x
tanh/identity x standard/source-gradient-balanced JEPA. Own identity improves
bounded3/3 hospital and MIMIC; whole identity3/3 hospital but1/3 MIMIC.
Whole combined beats persistence3/3 both; none establishes incremental whole
benefit over matching own. Cap is a confirmed repairable contributor, not full
transfer explanation. Report `artifacts/encoder_repair_20260910.json`;
candidate checkpoints local with activation metadata. No release changes,
interval calibration, seven-seed gate, other-target retention or promotion.

## Source-Supported Input Repair (2026-09-10)

`ENCODER_SOURCE_SUPPORT_PROTOCOL_20260910.md` and findings of the same prefix:
fixed six identity-decoder candidates; source-fit patient-support1/20 rules,
not external variable selection. Source-seen restriction changes only MIMIC
ionized_calcium, improves both modes3/3, eICU identical. Confirms a concrete
input/scaler/support mismatch after the amplitude repair. Whole remains worse
than own. `artifacts/encoder_source_support_20260910.json`; local protected
candidate checkpoints retain activation and support metadata. No release or
interval changes. Do not repeat this as an untested global-mask hypothesis.

## Training Context Dropout (2026-09-10)

Source-fit follow-up: `docs/research/findings/ENCODER_FIT_GAP_FINDINGS_20260910.md`, artifact
encoder_fit_gap_20260910.json. Same selected original32epoch weights,
own/wholeidentity7seeds. Whole fits training patients better6/7, source-tune
worse7/7(direction),6/7significant.250fitpatients per seed; source-tune exact
reproduction. Diagnostic only, not an independent gate; no runtime change.
Failure is present before MIMIC transfer, not just external calibration.

Related follow-up: `ENCODER_HISTORY_MASK_PROTOCOL_20260910.md` and findings.
28fits wire existing observation-step mask into history filter. Empty-prefix
invariance fixed; forecast not repaired, whole-vs-own0/7both domains. Own
identity harms3/7hospital,4/7MIMIC. Three encoder_history_mask artifacts with
20260910 suffix;300tests, runtime unchanged. Do not repeat empty-step masking
as an untested explanation of the whole-context gap.

`ENCODER_CONTEXT_DROPOUT_PROTOCOL_20260910.md` and findings:28 matched fits,
non-target variable/history blocks missing with fixed probability0.5 during
training only. Own-control logs/predictions exact. Whole identity median
hospital38.202->37.479,MIMIC33.368->33.128; significant improve3/7 both,
MIMIC harm1/7. Whole-vs-own0/7 both domains. TSICU and2014-2016 remain0/7
vs persistence. No uniform regularization repair;299 tests. Three artifacts
encoder_context_dropout, comparison and strata with20260910 suffix. No
runtime, calibration or promotion changes.

## Source Numerical Rescaling (2026-09-10)

Related follow-up: `docs/research/notes/ENCODER_TEACHER_DIAGNOSTIC_20260910.md` checked all14
original32-epoch identity teachers on a fixed glucose delta grid. Each
distinguishes sign and amplitude; centered numerical rank6/6. Normalization
does not erase all delta amplitude information. Not a clinical data audit,
new fit, or promotion; no runtime changes.

`ENCODER_RESCALE_PROTOCOL_20260910.md` and findings.28 matched32-epoch fits,
source-fit history-only non-glucose median/std transform. Own-control logs and
predictions exact. Whole identity improves hospital3/7,MIMIC3/7; MIMIC harm1/7;
whole-vs-own0/7. No uniform transfer repair. Artifacts encoder_rescale,
encoder_rescale_comparison,encoder_rescale_strata with suffix20260910.json.
No runtime or calibration change;298 tests. Affine source rescaling now tested.

## Component Sensitivity (2026-09-10)

`ENCODER_COMPONENT_PROTOCOL_20260910.md` and findings. Seven fixed32-epoch
whole weights x12 groups x value/age neutralization in two domains. Global
numeric neutralization keeping observation pattern improves MIMIC7/7, hospital
0/7. Exact-age impact tiny. Heme numeric result differs from whole-block
removal; prior group harm cannot be credited solely to numeric physiology.
Artifact `artifacts/encoder_components_20260910.json`. No serving of artificial
values, no new fitted repair or promotion.297 tests pass.

## Fixed Context Group Localization (2026-09-10)

`ENCODER_GROUP_PROTOCOL_20260910.md` and findings, artifact
`artifacts/encoder_groups_20260910.json`. Seven32-epoch whole identity weights,
12 group removals across two domains; full predictions reproduce, glucose
history preserved. Acid-base removal harms hospital4/7, helps MIMIC3/7;
blood/coagulation removal harms5/7 both. No stable deletion repair. Masking
does not isolate values from observation process; no causal/promoted edge,
new weights, runtime or calibration change.

## Extended Budget (2026-09-10)

`ENCODER_BUDGET_PROTOCOL_20260910.md` and findings:28 matched32-epoch fits,
exact reproduction of every first8-epoch log. Own identity improves short
version7/7 hospital and5/7 MIMIC; whole1/7 both, with MIMIC harm4/7. No
whole-vs-own benefit. More training alone does not repair context transfer.
Artifacts encoder_budget, encoder_budget_comparison, encoder_budget_strata
under artifacts with suffix20260910.json. No runtime or promotion.296 tests.

## Seven-Seed Supported Decoder (2026-09-10)

`ENCODER_CONFIRM_PROTOCOL_20260910.md`, separate descriptive strata protocol,
and `docs/research/findings/ENCODER_CONFIRM_FINDINGS_20260910.md`:28 fits, own/whole x bounded/identity.
Both identity-vs-bounded and identity-vs-persistence7/7 pooled both domains;
whole-vs-own0/7. MIMIC TSICU and some whole care-unit/time strata still fail.
No promotion/calibration/retention or runtime change. Artifacts:
`artifacts/encoder_confirm_20260910.json`,
`artifacts/encoder_confirm_strata_20260910.json`.296 tests pass. Do not describe
cap repair as only three-seed evidence or declare pooled7/7 universal.

## BNP / NT-proBNP Identity (2026-09-10)

`BNP_IDENTITY_PROTOCOL_20260910.md` / `docs/research/findings/BNP_IDENTITY_FINDINGS_20260910.md`:
dictionary-confirmed50963 NTproBNP was mislabeled BNP. Corrected extractor,
canonical name and standard cache; archive preserves original evidence.
Fixed supported whole candidates,3seeds x2modes: no significant glucose benefit
or harm, all paired CIs cross0. `artifacts/bnp_identity_repair_20260910.json`.
No promotion, new weights or calibration. Other calculated/measured method
mixtures were inventoried but not yet experimentally isolated.
# Source Learning Curve: 1,000 Patients (2026-09-10)

`ENCODER_SOURCE1000_FINDINGS_20260910.md`, source-expansion and learning-curve
protocols. Same glucose availability-safe task, same original tune/hospital/
MIMIC patients; add750 independent source fit patients, four arms x7seeds32epochs.
Whole identity improves previous250-fit version7/7hospital,6/7MIMIC; MAE
37.116/32.286. Still whole-vs-own0/7both, source-tune worse7/7. Expansion helps,
not complete repair; count and composition not isolated, fixedcompute.
Artifacts encoder_source_expansion,encoder_source1000,encoder_source1000_comparison,
encoder_source1000_strata with suffix20260910.json. 2,250stage unrun.
No runtime/promotion/calibration;302tests pass. MIMIC is validation, not lockbox.
# Source Learning Curve: 2,250 Patients (2026-09-10)

`ENCODER_SOURCE2250_FINDINGS_20260910.md`:prespecified28fits32epochs, same136
tune patients and original evaluation queries. Whole37.108/32.223MAE,
own36.998/31.214; whole-vs-own0/7both. Whole improves250-fit6/7each; marginal
median change from1000small, not a paired significance/convergence claim.
TSICU0/7 persists. Fixedcompute exposure and composition confounds remain.
Artifacts encoder_source2250,encoder_source2250_comparison,encoder_source2250_strata
with suffix20260910.json. No runtime, interval or promotion changes.
# Expanded Cohort Exposure and64Epoch Budget (2026-09-10)

`docs/research/findings/ENCODER_EXPANDED_BUDGET_FINDINGS_20260910.md` and protocol. Source-only exact
sampler replay shows decreasing supervised coverage withfixedcompute;2250fit
21.60%rows/24.07%futureevents,7median draws/patient.28matched64epochfits;
first32logs identical. Whole36.796hospital/32.037MIMIC, improves32epoch2/7both;
whole-vs-own0/7both. Source-tune worse6/7. Partial optimization gain, not
complete context generalization repair. Artifacts encoder_exposure,
encoder_source2250_64,encoder_source2250_64_comparison,encoder_source2250_64_strata
with suffix20260910.json. No runtime/interval/promotion changes;303tests.
# Requested Time Diagnostic (2026-09-10)

`docs/research/findings/ENCODER_QUERY_TIME_FINDINGS_20260910.md`: source-only14fixed2250/64checkpoints,
5/6/7/actual horizon queries, reproduced nominal tune scores. Actual versus
nominal improvements0/7both views,~0.005/~0.012MAE change. Added explicit
validated query_horizons input; default6h unchanged. Not actual-time retraining
or prospective measurement-time prediction. Artifact encoder_query_time_20260910.json.
No runtime/calibration/promotion;303tests pass.
# Actual-Time Retraining (2026-09-10)

`docs/research/findings/ENCODER_ACTUAL_TIME_FINDINGS_20260910.md` and protocol:28matched2250/64fits,
actual5-7h source training/tune queries, samephysiology andvalidationlabels.
Primary fixed6h whole36.799/32.067MAE, improves nominal-trained0/7both;
harm1/7hospital5/7MIMIC. Whole-vs-own0/7both. Negative timing-repair result,
not a claim that all event-time learning fails. Separate actual-query scores
descriptive, no next-measurement forecast. Artifacts encoder_actual_time,
encoder_actual_time_comparison,encoder_actual_time_strata suffix20260910.json.
303tests;no runtime/calibration/promotion.

# Executed Encoder Path (2026-09-10)

`docs/research/notes/ENCODER_PATH_DIAGNOSIS_20260910.md`:14 actual-time identity checkpoints,
synthetic gradient and separate-organ perturbation probes. Both value/JEPA
losses connect to non-target inputs, belief filter and both mixers; no frozen
context-path defect. Separate organ arguments do not affect future state when
context is fixed: the loaded class is base JointWholeBodyJEPAWorldModel, not
persistent organ transition. Organ information does enter shared context.
Glucose-only supervision is not a full multi-organ state-learning trial.
Artifact `artifacts/encoder_path_20260910.json`; one new regression passes.
Structural evidence only, no real-data accuracy, weight or runtime change.

# Source Auxiliary Supervision (2026-09-10)

`AUXILIARY_SUPERVISION_PROTOCOL_20260910.md`,
`AUXILIARY_ENCODER_PILOT_PROTOCOL_20260910.md`,
`docs/research/findings/AUXILIARY_ENCODER_FINDINGS_20260910.md`:
49,823 unchanged glucose queries/2386development patients,14 auxiliary targets
meet source-only support thresholds. Glucose labels reproduced.9matched8epoch
fits, real auxiliary improves glucose-only0/3 and shuffled0/3 significantly.
Original single-target first8logs exactly reproduce. Budget-boundary selections
prevent convergence/complete hypothesis rejection. No external data evaluation,
weights/calibrator/runtime promotion.309tests pass. Artifacts auxiliary_supervision
and encoder_auxiliary_pilot with suffix20260910.json; row data stays private.

# Longer Auxiliary Supervision (2026-09-11)

`AUXILIARY_ENCODER_BUDGET_PROTOCOL_20260910.md` and
`docs/research/findings/AUXILIARY_ENCODER_BUDGET_FINDINGS_20260911.md`:21 matched64epoch fits,7seeds,
source-only. Exact first8pilot and all64single-target reproduction. Primary
glucose improves single-target0/7 and shuffled0/7; shuffled harm1/7.
Secondary diagnostic: potassium persistence6/7, shuffled4/7; other auxiliary
targets improve their unmaintained heads but not all controls. Sparse tune
support explicitly separated from failures. Shared-context gradient conflict
5/21 sampled fit batches; no universal conflict claim. Auxiliary learning does
not establish beneficial cross-task transfer. No external/runtime/calibration
promotion. Artifacts encoder_auxiliary_budget_20260910.json and
encoder_auxiliary_learning_20260911.json. Private checkpoints only.
# Expanded insulin-potassium diagnostic (2026-09-11)

Question: does repairing subgroup support recover incremental sequence benefit?
Frozen outcome-blind hospital/patient roles, 21 fixed tree fits: own, real
insulin, donor-patient placebo. Source gain1/7, hospital0/7; sequence hospital
numerical harm7/7. Both roles now support patient bootstrap. Not merely N;
no full-JEPA, interval, runtime or promotion change. See
`docs/research/findings/EXPANDED_INSULIN_POTASSIUM_FINDINGS_20260911.md` and
`artifacts/expanded_insulin_potassium_20260911.json`.
# Rate transition collision (2026-09-11)

Synthetic counterexample: old insulin summaries identical for different
internal changes; new observed-transition features distinguish them. Future
and duplicate invariance tested. Real-data accuracy remains untested; do not
claim repaired forecast. See `docs/research/notes/RATE_TRANSITION_COLLISION_20260911.md`.
# Transition accuracy contrast (2026-09-11)

Fourteen matched real/placebo fits with transition features; frozen own and
original all-group scores reproduce. Original improvement0/7source/hospital,
both all/sequence. No accuracy repair, JEPA or runtime promotion. See
`docs/research/findings/TRANSITION_ACCURACY_FINDINGS_20260911.md` and machine report
`artifacts/expanded_insulin_transitions_20260911.json`.
# Insulin same-mask control (2026-09-11)

Seven fixed fits preserving exact treatment missingness, replacing finite
content with zero. Frozen real reproduced. Numeric vs support0/7gain all roles,
hospital1/7harm. See `docs/research/findings/INSULIN_SUPPORT_CONTROL_FINDINGS_20260911.md`.
No new lockbox, shared-JEPA or runtime promotion.
# Increment positive control (2026-09-11)

Artificial independent +/-0.5 future-label increment with same frozen patient
roles/tree settings and two-column base, 14fits. Detected7/7source/hospital.
Not real efficacy or complete-pipeline validation. See
`docs/research/findings/INCREMENT_SENSITIVITY_FINDINGS_20260911.md`; no runtime modification.
# Rate versus current draw timing (2026-09-11)

Pre-anchor timing-only audit:554/3373sequence windows show a rate change after
current draw, hospital43/358. Narrows interpretation of prior context tests;
not effect absorption or causal explanation. No fitting or runtime changes.
See `docs/research/findings/RATE_MEASUREMENT_ORDER_FINDINGS_20260911.md`.
# Frozen post-draw subgroup (2026-09-11)

No fits; frozen scores reproduced. Post-draw change group has40hospital patients,
real-vs-own significance0/7 despite numerical6/7. No subgroup routing promotion.
See `docs/research/findings/POSTDRAW_RATE_FINDINGS_20260911.md`; runtime unchanged.
# Observed rate-change bracket (2026-09-11)

Timing audit distinguishes pairs straddling current draw from entirely-after
pairs. Hospital strict support9patients, not40. No known actual transition time
or treatment availability inferred. `docs/research/findings/RATE_CHANGE_BRACKET_FINDINGS_20260911.md`.
Three rate tests pass; no forecast/runtime promotion.
# Rate-triggered query support (2026-09-11)

Native unequal-rate-record queries with rebuilt as-of potassium/future pairs;
fixed original patient roles. Hospital strict146patients, not9; restoration of
evaluation support, not forecast improvement. See `docs/research/notes/RATE_ANCHORED_SUPPORT_20260911.md`.
No training or runtime changes; raw caches private.
# Record-anchor accuracy (2026-09-11)

Twenty-one fixed real/own/support fits at rebuilt rate-record queries. Hospital
bracket-after real-vs-own2/7gain, support1/7, persistence7/7. Partial positive,
not promotion or causal timing repair. See `docs/research/findings/RATE_ANCHORED_PREDICTION_FINDINGS_20260911.md`.
No runtime or interval modifications.
# Rate-anchor influence (2026-09-11)

Frozen weights exactly reproduce; hospital bracket group benefit7/7numerical,
51.4-59.6%patients and no single-patient mean reversal. Still2/7significant,
not independent trials. `docs/research/notes/RATE_ANCHORED_INFLUENCE_20260911.md`. No fitting/runtime.
# Simultaneous snapshot contract (2026-09-11)

Latest equal-time unequal rates selected arbitrarily by ID; fixed to unknown.
Actual native-query support1/5526windows/1patient. Not main-cause proof or
accuracy repair. See `docs/research/findings/SIMULTANEOUS_RATE_FINDINGS_20260911.md`.
# Snapshot repair accuracy (2026-09-11)

Twenty-one fixed refits, exact old/own control reproduction. Repaired conflict
handling gain0/7, harm0/7 all roles/groups. No main-cause or promotion claim.
`docs/research/notes/SNAPSHOT_REPAIR_ACCURACY_20260911.md`; runtime unchanged.
# Rate-query whole-lab input cache (2026-09-11)

Restored40observed variables into existing53-slot encoder schema on unchanged
5526pairs. Current K reproduces. Allows neural comparison, not yet performed.
`docs/research/notes/RATE_ANCHOR_BODY_INPUTS_20260911.md`; no runtime or promotion change.
# Rate-anchor shared JEPA (2026-09-11)

Fourteen32epoch own/whole neural fits. Source whole-own5/7all,6/7bracket,
hospital1/7both. Both persistence7/7allgroups. Not treatment effects or full
promotion. `docs/research/findings/RATE_ANCHOR_JEPA_FINDINGS_20260911.md`; runtime unchanged.
# Rate JEPA hospital influence (2026-09-11)

Frozen scores reproduce;15sites/193patients only1site meets30patient minimum.
Mean gain7/7, hospital removal reverses2/7. No hospital selection or promotion.
See `docs/research/findings/RATE_JEPA_HOSPITAL_FINDINGS_20260911.md`; runtime unchanged.
