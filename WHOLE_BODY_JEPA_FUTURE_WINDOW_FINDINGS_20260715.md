# Whole-Body JEPA Future-Window Target Findings

Date: 2026-07-15

## Question

Does a true JEPA target improve when the target encoder represents a causal
future trajectory prefix rather than one sparse future row?

## Implementation

The target side now receives, for the same stay and anchor, the available
future-label prefix on the horizon grid:

`1h -> 3h -> 6h -> 12h -> 24h -> 48h`

For a task at horizon `h`, target points beyond `h` are masked. Rows are joined
only when they share the same stay and anchor timestamp. The target encoder
encodes each available future state with the EMA target body encoder and uses a
masked mean over the available prefix. This prevents a later label from
leaking into an earlier task and avoids treating one sparse row as the entire
future state.

The implementation is selectable with:

`--future-target-mode window|single`

The default is `window`; `single` is a matched ablation that preserves the old
one-row JEPA target.

## Matched Smoke

Both runs used the same:

- 5 modules: sepsis, AKI, respiratory, electrolyte/acid-base, cardiovascular instability
- 3,465 rows, 56 subjects, 39 hospitals
- as-of alignment, horizon-scaled age policy, 1 training epoch
- identical masking, batch size, cross-system loss, EMA momentum and 7 patient seeds
- hospital-held-out evaluation

The future-window run is:

`whole_body_joint_gate_smoke_true_jepa_future_window_canonical_20260715.json`

The matched one-row run is:

`whole_body_joint_gate_smoke_true_jepa_single_target_20260715.json`

| Target-horizon | Window median delta vs persistence | Single median delta vs persistence | Window value passes | Single value passes | Window cross passes | Single cross passes |
|---|---:|---:|---:|---:|---:|---:|
| glucose@6h | -0.210579 | -0.210144 | 4/7 | 4/7 | 6/7 | 6/7 |
| potassium@6h | +0.005170 | +0.005224 | 2/7 | 2/7 | 5/7 | 5/7 |
| bicarbonate@12h | -0.005244 | -0.005228 | 4/7 | 4/7 | 2/7 | 2/7 |
| MAP@1h | +0.160552 | +0.161250 | 2/7 | 2/7 | 6/7 | 6/7 |
| creatinine@24h | -0.018599 | -0.018600 | 1/1* | 1/1* | 1/1* | 1/1* |

`*` The creatinine cell had only one eligible patient split in this small
smoke and is not a promotion result.

## Result

Both models remained `candidate_only`; neither produced a validated target-
horizon cell. The future-window target is numerically almost identical to the
single-row target under this small, one-epoch budget. Therefore this experiment
does **not** prove that the future-window target improves forecasting.

It does establish that:

1. the future trajectory target is implemented without future-horizon leakage;
2. the target contract and single-row ablation are now directly comparable;
3. the target-window change alone is not enough to promote the joint JEPA.

The result is not evidence that JEPA cannot learn future trajectories. The
smoke still has sparse labels, only one epoch, and a target encoder that
aggregates the available prefix with a simple masked mean. A larger matched
training audit would be needed before judging the representation itself.

## Full Matched Audit

The comparison was repeated with the full available matched budget:

- 65,077 rows, 917 subjects, 154 hospitals
- 15 body-system modules
- 2 training epochs, identical seeds and gate settings
- future-window and single-row targets trained in separate runs

The window and single-row results remained effectively indistinguishable:

| Target-horizon | Window median delta | Single median delta | Window value passes | Single value passes | Window cross passes | Single cross passes |
|---|---:|---:|---:|---:|---:|---:|
| glucose@6h | +3.219116 | +3.214878 | 2/7 | 2/7 | 3/7 | 3/7 |
| potassium@6h | +0.003060 | +0.003118 | 2/7 | 2/7 | 4/7 | 4/7 |
| bicarbonate@12h | +0.046736 | +0.046582 | 2/7 | 2/7 | 2/7 | 2/7 |
| MAP@1h | +2.313111 | +2.318116 | 0/7 | 0/7 | 2/7 | 2/7 |
| creatinine@24h | +0.003142 | +0.003023 | 3/7 | 3/7 | 1/7 | 1/7 |

Both reports contain zero validated target-horizon cells and both remain
`candidate_only`:

- `whole_body_joint_gate_1k_true_jepa_future_window_20260715.json`
- `whole_body_joint_gate_1k_true_jepa_single_target_20260715.json`

This full audit is stronger than the small smoke: the trajectory target did
not change the promotion decision, patient split pass counts, cross-system
pass counts, or conformal pass counts. The current bottleneck is therefore not
simply the number of future rows supplied to the target encoder. The next
architectural question is whether a trajectory target needs an order-aware
temporal representation and a loss that gives that representation enough
training signal; changing only the target aggregation is not sufficient.

## Sequence-Level JEPA Revision

The next revision was implemented and tested:

- the predictor now emits `predicted_trajectory_latent` with one latent per
  horizon;
- the target side emits `target_trajectory_latent` and a per-step availability
  mask;
- the JEPA loss is computed over every available future step, not over one
  pooled latent;
- horizon time features are supplied separately to each predicted step;
- the existing numerical decoder remains current-anchored residual prediction;
- the old one-row output fields remain available for compatibility.

The implementation passed all 246 repository tests.

The full 1,000-stay, 15-module audit was then repeated:

`whole_body_joint_gate_1k_true_jepa_sequence_20260715.json`

Compared with the previous pooled-window target:

| Target-horizon | Sequence median delta | Previous pooled-window delta | Sequence value passes | Previous value passes | Sequence cross passes | Previous cross passes |
|---|---:|---:|---:|---:|---:|---:|
| glucose@6h | +3.226624 | +3.219116 | 2/7 | 2/7 | 3/7 | 3/7 |
| potassium@6h | +0.003179 | +0.003060 | 2/7 | 2/7 | 4/7 | 4/7 |
| bicarbonate@12h | +0.047124 | +0.046736 | 2/7 | 2/7 | 2/7 | 2/7 |
| MAP@1h | +2.306492 | +2.313111 | 0/7 | 0/7 | 2/7 | 2/7 |
| creatinine@24h | +0.002937 | +0.003142 | 3/7 | 3/7 | 1/7 | 1/7 |
| lactate@3h | +0.192891 | +0.193340 | 0/7 | 0/7 | 4/7 | 3/7 |

The revision produces small target-specific changes and one improved
cross-system lactate count, but it does not change the promotion decision:
validated target-horizon cells remain `0`, and the model remains
`candidate_only`. This is a real architectural improvement in the training
contract, not yet a validated accuracy improvement.

## Measurement-Pure Delta Target

The next fix removed a major source of target contamination. Future targets
are now encoded as `future_state - anchor_state`. The future measurement mask
is used only for step eligibility, not as a target feature, and a future step
must contain at least two observed variables before contributing to JEPA loss.
This prevents the target latent from learning the future measurement schedule
instead of physiology.

The full audit was repeated with 65,077 rows, 917 subjects, 154 hospitals and
15 modules:

`whole_body_joint_gate_1k_true_jepa_measurement_pure_delta_20260715.json`

| Target-horizon | Pure-delta median delta | Sequence median delta | Pure value passes | Sequence value passes | Pure cross passes | Sequence cross passes |
|---|---:|---:|---:|---:|---:|---:|
| glucose@6h | +3.215302 | +3.226624 | 2/7 | 2/7 | 4/7 | 3/7 |
| potassium@6h | +0.003079 | +0.003179 | 2/7 | 2/7 | 4/7 | 4/7 |
| bicarbonate@12h | +0.055873 | +0.047124 | 2/7 | 2/7 | 2/7 | 2/7 |
| MAP@1h | +2.333121 | +2.306492 | 0/7 | 0/7 | 3/7 | 2/7 |
| creatinine@24h | +0.005468 | +0.002937 | 3/7 | 3/7 | 1/7 | 1/7 |
| heart_rate@3h | +0.418107 | +0.454645 | 0/7 | 1/7 | 6/7 | 4/7 |
| lactate@3h | +0.198885 | +0.192891 | 0/7 | 0/7 | 4/7 | 4/7 |

This partially improves cross-system stability for glucose, MAP and heart
rate, confirming that measurement-process noise was real. However, validated
cells remain `0` and the model remains `candidate_only`. The remaining
limitation is now the quality and identifiability of the future physiological
target itself, not merely whether the future mask is exposed to the target
encoder.

## Predict-Update Belief Filter and Paired-Delta Supervision

The next revision addressed two concrete signal-path problems:

- a neural `predict-update` belief filter now processes the history encoder;
  its update decay is driven by the observed stay-relative time feature, and
  it exposes both `belief_state` and `belief_uncertainty` for auditing;
- the input metadata offset was corrected so module-presence masking no longer
  overwrites the onset feature;
- future JEPA deltas now require both an observed anchor and an observed future
  value.  A missing anchor is no longer silently replaced by the normalized
  median and treated as a zero physiological change;
- target-side token pooling uses only paired measured deltas.  Future masks
  never enter the context path, so the predictor cannot copy the future
  measurement schedule.

The implementation passed all 246 repository tests. A matched 60-stay smoke
was run before the full audit; it remained fail-closed. The full 1,000-stay,
15-module audit used 65,077 rows, 917 subjects and 154 hospitals:

`whole_body_joint_gate_1k_true_jepa_belief_paired_delta_20260715.json`

| Target-horizon | Paired-delta median delta | Previous pure-delta delta | Paired value passes | Paired cross passes |
|---|---:|---:|---:|---:|
| glucose@6h | +3.823635 | +3.215302 | 1/7 | 4/7 |
| potassium@6h | +0.000223 | +0.003079 | 3/7 | 4/7 |
| bicarbonate@12h | +0.035439 | +0.055873 | 3/7 | 6/7 |
| MAP@1h | +2.143926 | +2.333121 | 0/7 | 5/7 |
| creatinine@24h | +0.003437 | +0.005468 | 3/7 | 4/7 |
| heart_rate@3h | +0.618419 | +0.418107 | 0/7 | 7/7 |
| lactate@3h | +0.208401 | +0.198885 | 0/7 | 4/7 |

The paired-delta target improves several cross-system pass counts, especially
bicarbonate, MAP and heart rate, but it does not produce any fully validated
target-horizon cell. Glucose also regresses in the patient-held-out aggregate.
Therefore the neural belief filter and paired-delta supervision are retained
as research candidates, not promoted as a replacement for the current
fail-closed router. The remaining bottleneck is robust identification of
future physiological state under sparse, target-specific observation, not
simply model capacity or a missing recurrent layer.

## Target-Specific Temporal Adapters

To address target heterogeneity directly, the decoder now keeps the shared
body latent but applies one horizon-conditioned residual adapter per target.
Each adapter receives the shared latent, the target's current value and the
four horizon features, then produces a bounded residual state for that target's
forecast and observation heads. The final adapter layers are zero-initialized,
so the new model starts as the previous decoder rather than changing the
forecast contract at initialization. The runtime exposes
`target_temporal_latent` and `target_temporal_gate` for auditing.

The implementation passed all 246 repository tests. A matched 60-stay smoke
was run with the same training and gate settings as the paired-delta model:

`whole_body_joint_gate_smoke_true_jepa_target_adapters_20260715.json`

| Target-horizon | Adapter median delta | Paired-delta median delta | Adapter value passes | Adapter cross passes |
|---|---:|---:|---:|---:|
| glucose@6h | +0.140354 | -0.270012 | 1/7 | 3/7 |
| potassium@6h | +0.001394 | +0.006265 | 3/7 | 2/7 |
| bicarbonate@12h | +0.000621 | +0.012054 | 3/7 | 4/7 |
| MAP@1h | +0.030095 | +0.019120 | 2/7 | 5/7 |
| creatinine@24h | +0.002555 | -0.012892 | 0/7 | 1/7 |
| heart_rate@3h | +0.142519 | +0.035730 | 2/7 | 3/7 |

The adapter creates the expected target differentiation, but this smoke does
not show robust improvement: glucose and heart-rate regress, and the complete
promotion gate remains closed. The adapter is therefore a candidate-only
research component. This result also identifies the next real requirement for
this architecture: target-specific future latent supervision or stronger
regularization, rather than adding more unconstrained decoder capacity.

## Per-Target Future Latent JEPA Supervision

The next revision implemented that requirement directly:

- each target has a separate future-latent predictor conditioned on the shared
  context latent, its current value and the horizon features;
- each future target latent is encoded from that target's own paired delta,
  rather than from one pooled future body delta;
- the JEPA loss is computed over available `(horizon, target)` cells;
- the target-specific predicted latent is also used by the corresponding
  numerical forecast head.

The implementation passed all 246 repository tests. The matched smoke report
is:

`whole_body_joint_gate_smoke_true_jepa_per_target_future_latent_20260715.json`

| Target-horizon | Per-target median delta | Decoder-adapter median delta | Value passes | Cross passes |
|---|---:|---:|---:|---:|
| glucose@6h | +0.142139 | +0.140354 | 1/7 | 3/7 |
| potassium@6h | +0.001446 | +0.001394 | 3/7 | 2/7 |
| bicarbonate@12h | +0.000958 | +0.000621 | 3/7 | 4/7 |
| MAP@1h | +0.031724 | +0.030095 | 2/7 | 5/7 |
| creatinine@24h | +0.002381 | +0.002555 | 0/7 | 1/7 |
| heart_rate@3h | +0.143178 | +0.142519 | 2/7 | 4/7 |

This is a clean negative result for the hypothesis that target-specific future
latent supervision alone will unlock the joint model. It changes the target
representations and slightly improves some cross-system counts, but glucose
and heart-rate remain worse than the paired-delta baseline and no cell passes
the full promotion gate. The target-specific path remains candidate-only; the
remaining limitation is not merely a shared-latent bottleneck, but weak and
heterogeneous future supervision after target-specific conditioning.

## Target-Horizon Balanced JEPA Loss

Because sparse targets might otherwise receive fewer gradient updates, the
JEPA losses were changed to average each active `(target, horizon)` cell first
and then average the active cells. This gives sparse and dense targets equal
cell-level weight without changing the data or the model capacity.

The matched smoke report is:

`whole_body_joint_gate_smoke_true_jepa_per_target_balanced_20260715.json`

The result was effectively unchanged from the unbalanced per-target loss:
glucose@6h `+0.142838` vs `+0.142139`, potassium@6h `+0.001379` vs
`+0.001446`, bicarbonate@12h `+0.000962` vs `+0.000958`, and heart-rate@3h
`+0.143135` vs `+0.143178`. The promotion gate remained at zero validated
cells. This rejects loss reweighting as the main fix; the remaining issue is
the quality, timing and identifiability of future labels, not target-gradient
imbalance.

## Safety Boundary

No checkpoint was promoted. Non-validated cells remain fail-closed. This is a
factual research experiment only: no causal or clinical claim is allowed.

## Raw-Event Future-Label Repair

The future-label contract was then rebuilt from the raw eICU measurement-event
tables rather than trusting a nominal transition row. The rebuild applies a
target-specific strict tolerance and records, for every target:

- the raw future event time;
- the actual positive `delta_t` from anchor to event;
- strict, near, or missing quality;
- a training weight and the paired current-observation requirement.

Only strict labels with a real post-anchor event, a positive actual `delta_t`,
and an observed anchor are eligible for JEPA supervision. Near and missing
labels remain available to coverage audits but receive zero training weight.
Measurement timing is kept separate from the physiological value task. The
model receives the target-specific actual `delta_t`; it does not silently
pretend that every label occurred exactly at the nominal 1/3/6/12/24/48-hour
horizon.

The repaired eICU tree contains 68 transition files and 1,128 target-horizon
records. `whole_body_future_label_quality_audit.py` reports
`all_contracts_pass: true`, including timing, post-anchor, positive-delta,
weight, and paired-label checks. Respiratory numeric measurements were also
restored to the raw-event path, so values such as FiO2, PaCO2, PEEP and tidal
volume are audited as numeric targets rather than as presence-only events.

This repairs the future-supervision contract; it does not by itself prove that
the JEPA beats persistence or justify promotion. The patient-held-out,
hospital-held-out, conformal, and cross-system gates remain mandatory, and the
model remains `candidate_only` until those gates pass.

## Target-Time-Regime Balanced Candidate

To test whether mixed future regimes were hiding useful signal, a candidate
loss now balances active supervision across three keys:

`target × actual_delta_t_bin × anchor_regime`

The actual-delay bins are 0--1.5, 1.5--4.5, 4.5--9, 9--18, 18--36,
36--72, and above 72 hours. Anchor regimes are target-specific quantile bins
computed from the training rows only; future values are never used to define
them. The same strata are reported diagnostically on held-out rows, while the
existing promotion registry remains target-by-horizon and fail-closed.

The candidate was run on a matched 20-stay repaired-label smoke with
respiratory, AKI, and sepsis modules, 1/3/6-hour horizons, a true JEPA target,
and all seven patient seeds. It produced 443 non-empty diagnostic strata, but
`validated_target_horizon_count` remained `0` and `promotion_status` remained
`candidate_only`. This is a valid implementation and a useful diagnostic
decomposition, not evidence of a global accuracy improvement. A full-budget
comparison is still required before deciding whether regime balancing should
be retained.
