# Whole-Body Joint Model Problem Resolution

Date: 2026-07-14

This report resolves the seven scientific problems raised for the whole-body
joint JEPA.  It records what was fixed, what was tested, and what remains
fail-closed.  It does not grant causal, clinical, or runtime decision
authority.

## 1. Joint latent promotion

The joint model is still one shared latent with cross-system variable tokens;
it is not split into independent organ models.  The gate now supports
target-horizon promotion instead of an unrealistic all-or-nothing model
promotion:

- complete joint gate: `candidate_only` unless every eligible cell passes;
- partial gate: only cells passing patient-heldout, hospital-heldout,
  conformal, and cross-system ablation may move;
- every other cell falls back to the validated source.

The final corrected horizon-scaled re-audit validated **zero** joint cells.
An earlier intermediate run reported one cell, but it was discarded after a
module-presence bug was found: a same-named variable from another module could
incorrectly mark the current module as present.  The final run uses
module-local presence after age filtering.  Therefore the whole-body latent is
not promoted as a universal forecaster.

## 2. Unstable cross-organ forecasts

Instability is handled by a target-horizon registry, not by averaging unstable
targets into a global score.  A cell must pass all independent gates before it
can use the shared latent.  Unsupported cells remain fallback/persistence.

The strict 1h source re-audit increased validated factual-router cells from
13 to 19 after repairing the future-label contract.  This is an extractor
repair, not evidence that the joint latent itself is universally valid.

## 3. Sparse future labels

Short-horizon future labels now use the target-specific tolerance rule
`min(2h, max(0.25h, horizon/2))`.  The strict 1h eICU re-extraction and raw
event-time audit found:

- 0 future labels without a post-anchor raw event;
- 0 inconsistent target-horizon tasks;
- 3 honest source-coverage gaps: CK-MB index, troponin T, serum ketones.

Those three gaps remain missing, not imputed.  This removes false future labels
without inventing future measurements.

## 4. Arbitrary six-hour as-of age

The joint loader now supports two policies:

- `fixed`: the legacy six-hour cap;
- `horizon_scaled`: `min(6h, max(0.5h, horizon/2))`.

The age-policy sensitivity audit compares global caps from 0.5h through 12h,
horizon-scaled alignment, and a conservative target-specific profile. The
support frontier is explicit: global 0.5h/1h/2h/3h/6h/12h produces 332,246 /
435,886 / 599,706 / 752,511 / 987,885 / 1,005,327 rows; horizon-scaled
produces 689,899 rows; target-specific conservative produces 912,032 rows.
Observed-value fraction ranges from 7.80% to 16.58%. This is a support
sensitivity result, not proof that a looser policy forecasts better.
The new policy is scientifically safer, but its full gate produced fewer
validated joint cells; it is therefore retained as the strict default, not
promoted as a performance improvement.

Per-variable age overrides are also supported through the loader and CLI.

## 5. Asynchronous stale measurements

Age is now filtered per variable before module presence and module count are
computed.  A module is present only when at least one of its variables survives
the age rule.  A stale value cannot silently keep a whole module alive.

Regression tests cover causal backward alignment, bounded future labels, and
rejection of a one-hour target driven by a one-hour-old input.

## 6. Measurement-process confounding

The joint model now has a `neutralized` measurement-process mode.  It keeps
normalized physiological values but removes mask, age, and module-presence
signals.  A matched 1k-stay, two-mode gate was run with the same age policy and
training budget:

- observed process: 0 validated target-horizon cells (65,077 rows);
- neutralized process: 0 validated target-horizon cells (the same 65,077 rows).

This does not prove that measurement process has no influence.  It does prove
that the current data do not justify promoting it as physiological coupling.
The process comparison remains an audit, and both modes stay research-only.

## 7. Multi-hop cross-organ coupling

The re-audit tested `sepsis -> MAP -> renal` against a direct sepsis-to-renal
baseline and a capacity-matched placebo:

- 24h creatinine: patient 7/7 and hospital-heldout significant against both;
- 24h BUN: patient 7/7 and hospital-heldout significant against both;
- 6h creatinine: patient 6/7, so near-miss;
- 48h creatinine/BUN: unstable hospital result;
- urine output: not validated.

The new conformal re-audit calibrates only on discovery OOF residuals. All
tested active target/scope combinations reached the interval coverage gate on
the seven patient splits, but that does not rescue unstable predictive gain:

- 24h creatinine and BUN: gain 7/7 and conformal 7/7;
- 6h creatinine: gain 6/7 and conformal 7/7;
- 48h creatinine/BUN: gain 5/7 and conformal 7/7;
- urine output: gain 0-1/7 despite conformal 7/7.

Therefore the 24h creatinine/BUN cells are research candidates for a multi-hop
router, not a promoted universal latent or a causal pathway. Conformal
coverage describes interval honesty; it does not create predictive signal.

## 8. Shared-latent repair attempt

The original joint training path had three structural weaknesses: the gate did
not pass a non-zero cross-forecast loss by default, variables had no explicit
module identity, and whole-module masking left a presence side-channel. These
made it possible for the model to optimize local reconstruction or the
measurement process without proving cross-organ prediction.

The repair now includes:

- module-aware variable embeddings and a two-stage mixer: variable tokens are
  first mixed within the body representation, then pooled into module tokens
  and mixed across modules;
- a hybrid body pool that retains both variable-level context and balanced
  module-level context, preventing large modules from dominating the latent;
- signature-balanced batches by actual observed module combination;
- complete module masking that removes current values, history values, module
  presence flags, and history presence flags;
- cross-forecast supervision with a curriculum, so ordinary factual
  forecasting is learned before the harder masked cross-organ objective;
- a corrected self-only ablation that keeps only the target-owning module
  presence flags, rather than leaking the whole body topology;
- hospital-heldout conformal coverage is now an explicit promotion condition,
  in addition to patient-heldout conformal coverage.

The 1k smoke gate remained `candidate_only` after the structural repair. The
full 683,911-row eICU run showed real but target-specific signal:

- glucose@6h and potassium@6h passed patient factual value on 7/7 splits;
- bicarbonate@12h passed factual value on 6/7 and cross-system ablation on
  7/7;
- MAP cross-system ablation passed 7/7, but its factual value gate did not;
- no target-horizon cell passed every patient, hospital, conformal, and
  cross-system condition, so `validated_target_horizon_count = 0` and the
  shared latent remains `candidate_only`.

This is a meaningful repair, not a promotion: it removes training/evaluation
shortcuts and recovers stable cross-organ evidence in selected targets, but a
universal shared latent is still not demonstrated. Unsupported cells continue
to fall back to their validated target-specific source.

The listed full-run JSON was generated before the final explicit
hospital-conformal field and the later cross-forecast curriculum were added
to the registry/training path. It is retained as a full-cohort diagnostic;
its zero-promotion result is not a promotion claim for the newer code. A
release artifact should rerun the full gate with both final changes before
any checkpoint is considered for use.

An additional attention-level missingness-mask variant was tested on the 1k
smoke cohort and rejected: glucose@6h fell to 0/7 factual-value passes and
the run remained `candidate_only`. That variant is not retained in the main
architecture; missingness remains an audited input, not a hard attention
shortcut.

A factorized local-head/shared-head variant was also tested and rejected. It
did not improve the cross-system gate and worsened glucose, MAP, and
creatinine on the 1k smoke cohort. The retained model therefore keeps the
validated single target head over the hybrid body latent, rather than adding
capacity that has not demonstrated generalization.

A zero-initialized shared-residual variant was tested with the same 1k gate.
It preserved the `candidate_only` outcome and did not improve the stable
cross-system cells, so it was also rejected rather than promoted.

A target/horizon/patient-conditioned shared router was then tested to model
heterogeneity explicitly. It also remained `candidate_only` and reduced
factual stability for bicarbonate, MAP, and potassium on the 1k gate. It is
not retained. This rules out the simple explanation that adding conditional
gates alone is sufficient; the remaining heterogeneity is not recoverable by
capacity or routing without stronger, better-supported cross-organ labels.

## 9. Independent MIMIC-IV validation

The full MIMIC-IV v3.1 transition cohort was rerun with patient splits,
care-unit holdout, chronological holdout, and conformal forecast checks:

- 640,164 rows, 63,307 subjects, 90,190 stays, 17 care units;
- 59 eligible numeric targets;
- 3 validated nowcast targets;
- 16 validated forecast targets;
- 9 validated forecast-interval targets.

This is independent factual evidence for selected targets, not evidence that
the eICU joint latent generalizes universally. MIMIC-IV has no hospital
holdout across institutions in this artifact, so external multi-hospital
generalization remains open.

## Final status

The practical model contract is now:

1. one shared body latent and cross-system mixer;
2. strict event-time future-label contract;
3. horizon-scaled and variable-specific observation age filtering;
4. measurement-process neutralization audit;
5. target-horizon promotion with fallback;
6. multi-hop features allowed only in cells that pass their own held-out and
   conformal gate;
7. independent MIMIC-IV validation kept separate from eICU promotion.

The remaining scientific limitation is not hidden by the code: a universal
whole-body latent has not passed. The system can now use validated shared
signals where they exist and refuse to move everywhere else. All outputs remain
factual/observational and research-only.

## Artifacts

- `whole_body_joint_age_policy_audit.json`
- `whole_body_joint_age_sensitivity_audit_20260714.json`
- `whole_body_joint_gate_full_10k_horizon_scaled_v2.json`
- `whole_body_joint_gate_1k_horizon_scaled_observed_v2.json`
- `whole_body_joint_gate_1k_horizon_scaled_neutralized_v2.json`
- `whole_body_joint_measurement_process_comparison_v2.json`
- `whole_body_joint_event_time_audit_strict_1h.json`
- `eicu_body_system_multihop_coupling_audit_recheck_20260714.json`
- `eicu_body_system_multihop_coupling_audit_conformal_20260714.json`
- `mimiciv_cross_database_coverage_audit_20260714.json`
- `whole_body_joint_gate_1k_hierarchical_module_20260714.json`
- `whole_body_joint_gate_1k_hierarchical_fair_20260714.json`
- `whole_body_joint_gate_1k_hybrid_joint_20260714.json`
- `whole_body_joint_gate_1k_curriculum_20260714.json`
- `whole_body_joint_gate_1k_factorized_heads_20260714.json`
- `whole_body_joint_gate_1k_attention_masked_20260714.json`
- `whole_body_joint_gate_1k_zero_shared_residual_20260714.json`
- `whole_body_joint_gate_1k_heterogeneous_router_20260714.json`
- `whole_body_joint_gate_full_10k_hybrid_joint_20260714.json`
