# E21: State Content Retention Intervention

E20 locates known-content readability gaps at assimilation and later state.
The actual continuation trains only SmoothL1 on one masked target per context.
That omission is confirmed; its role in failure is a hypothesis, not a bug proof.
This intervention changes the state learning objective, not another frozen reader.

Use all five E19 fold roles, identical outcome-blind hash contexts (2048 fitting,
1024 selection patients per fold). Report all original context targets and all53
observed variables with support; repeated fold losses averaged per patient.
These are reused development data, not an untouched lockbox or full-cohort test.

Keep epoch10 event encoder frozen and identical. Cache its event-body, organs,
presence and legal actual event times after original target/sibling masking.
Reproduce original final body and native masked prediction on every context
before training. Replay original advance/update equations, no raw input skip.

Three arms, same layer/readout initialization, cached inputs, schedule and budget:
primary-only, real-content, shuffled-content. Each has the same new Linear64→53
content head. One fixed pass trains only this head on fitting last-observed
values. Primary-only trains the head on real labels throughout, but detaches
the state for auxiliary loss so only the primary loss updates its state;
real uses real labels; shuffled rotates fitting patients' values independently
within each observed variable, preserving the observation mask and marginal.
Variables with fewer than2 fit observations are excluded from auxiliary training
and scoring for every arm. Known zero remains valid. Original hidden target and
algebraic siblings are never auxiliary labels for that context.

Then exactly2 passes update the original state layer, masked readout and auxiliary
head with AdamW(lr1e-4, weight_decay1e-2), clip_norm1, original SmoothL1 plus
0.25*mean observed-content SmoothL1 (detached state for primary-only). This gives
all auxiliary heads equal training exposure without changing the primary-only
state objective. Freeze all event
encoder weights, no dropout, fixed initialization seed20261003 and identical
hash-independent schedules. Do not tune weight/budget after looking at selection.
Content head warmup uses AdamW(lr1e-3, weight_decay1e-2),1pass. No old optimizer
state is reused: matched primary-only controls the restart and extra training.

Content target is the last known measurement, not a claim of its current true
value. This objective may conflict with evolving current physiology. A content
improvement alone is not a repair. Score the original unknown masked target
alongside known-content MAE; keep the original frozen native baseline. Compare
real against primary-only and shuffled with patient/component paired bootstrap.
At least30 patients and3 source components, both CI95 upper<0 and one-sided
p<.05/(53*2*2) for descriptive per-variable screens (2tasks x2comparisons).
Report all53 slots, unsupported included. For masked task also compare every arm
to frozen original; those differences are descriptive, not a new promotion gate.
No promotion: unmasked copying, average gain, or this bounded diagnostic cannot
replace full original masked inference, cross-source and runtime gates.

If real retains content but worsens masked inference, reject it as repair, do not
raise its weight or suppress damaged targets. If neither changes, limited budget
cannot prove universal impossibility. If both improve, larger independent-role
validation is still required. No forecasting/causal/clinical claim.

Private caches, patients and trained weights remain ignored. Preserve aggregate
reports, settings, hashes and unit tests. On role/hash/parity/nonfinite mismatch,
stop; do not treat implementation failure as physiological evidence.
