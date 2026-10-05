# Persistent Whole-Body State: Corrected External Findings

Date: 2026-08-30

## Decision

The persistent whole-body state remains a research candidate. Its unrestricted
`glucose@3h` path beats the old baselines, but it does not beat a fair
owner-organ JEPA baseline after both receive the same target-head
specialization. The external ablations show a narrower rule:

```text
all causal measurements -> shared persistent state for reconstruction
target forecast          -> target owner-organ state by default
cross-organ influence    -> only through a separately validated sparse adapter
```

The active release is unchanged. No candidate in this report has an untouched
third-database lockbox, and unsupported cells continue to use the validated
router or persistence.

## Defects corrected

The investigation found and fixed four model/evaluation defects:

1. Absolute recurrent body state had been mixed directly with a JEPA
   future-change latent.
2. A near-zero residual gate starved the new physiology path of gradient.
3. The persistent path replaced the trained base future latent with a random
   projection. It now starts as an exact zero residual around the trained base
   JEPA trajectory.
4. Label reports counted future values that already had zero supervision
   weight. Reports now count only effective measured labels.

The masked-nowcast pretraining path also contained an uninitialized
`domain_loss` branch. It is fixed and covered by the ordinary training path.

## Urine-output hypothesis was rejected

The original 200-stay `urine_output@3h` point result did not survive the full
development cohort. It was a bounded pilot, not a release result.

Two stronger label definitions were tested:

- weight-normalized urine output using only a prior measured body weight;
- interval-aware 3-hour urine totals requiring at least two-thirds temporal
  coverage.

Neither transferred from eICU to MIMIC-IV better than persistence. The
remaining mismatch is not explained by body size or a simple point-label bug.
Complete intake, quantitative diuretic exposure, catheter workflow, and a
consistent collection-interval contract are still missing. Urine output stays
on fallback.

## External results

All values below are patient-equalized normalized MAE on MIMIC-IV after
training on eICU. Lower is better.

| Cell and input scope | Candidate | Base JEPA | Cell expert | Persistence | Result |
|---|---:|---:|---:|---:|---|
| glucose@3h, unrestricted all organs | 0.8792 | 0.9148 | 1.8502 | 0.8917 | point and conformal pass |
| glucose@3h, owner organ only | **0.8608** | 0.9148 | 1.8502 | 0.8917 | passes all four paired point comparisons |
| bicarbonate@3h, unrestricted all organs | 0.4280 | 0.4537 | 1.6209 | 0.4201 | loses to persistence |
| bicarbonate@3h, owner organ only | **0.4165** | 0.4537 | 1.6209 | 0.4201 | point estimate wins, CI crosses zero |

The table above is the input-scope ablation before owner-organ target-stage
training. A second, fair training run specialized both persistent and base
target heads on the same owner-organ view:

| Cell | Persistent owner head | Base owner head | Persistence | Incremental persistent-state result |
|---|---:|---:|---:|---|
| glucose@3h | 0.8608 | **0.8440** | 0.8917 | persistent state is significantly worse than owner base |
| bicarbonate@3h | **0.4170** | 0.4177 | 0.4201 | no significant difference from owner base or persistence |

Canonical artifacts:

- `artifacts/persistent_physiology_state_glucose_owner_gate_3h_external_20260830.json`
- `artifacts/persistent_physiology_state_bicarbonate_owner_gate_3h_external_20260830.json`
- `artifacts/persistent_physiology_state_glucose_owner_trained_3h_external_20260830.json`
- `artifacts/persistent_physiology_state_bicarbonate_owner_trained_3h_external_20260830.json`
- `artifacts/persistent_physiology_state_urine_per_kg_1000_external_20260829.json`
- `artifacts/persistent_physiology_state_windowed_urine_1000_external_20260829.json`

For glucose, owner-organ input significantly beat the unrestricted whole-body
input by `0.0184` normalized MAE, 95% CI `[-0.0289, -0.0083]`, and beat
persistence by `0.0310`, 95% CI `[-0.0423, -0.0200]`.

For bicarbonate, owner-organ input significantly beat the unrestricted model,
but its advantage over persistence was `0.0036`, 95% CI
`[-0.0126, 0.0053]`. It is therefore a near-pass, not a promotion.

## What the shared state has proved

The persistent candidate can recover bicarbonate from the rest of the causal
history when the complete bicarbonate history is hidden. It beats both the
base JEPA and the population-center placebo in that all-history masked test.
This is evidence that the representation contains shared current-state
physiology.

It has not proved unrestricted cross-organ future transfer:

- disabling graph message passing produces no practical gain;
- masking non-owner organs improves both glucose and bicarbonate;
- glucose still fails masked recovery against the population-center placebo.

The fair owner-stage experiment further shows that the recurrent persistent
transition has not added future-prediction value beyond an owner-organ JEPA.
The correct interpretation is that shared-state reconstruction is real, while
future dynamics remain target- and organ-specific. More inputs are not
automatically more physiology; unvalidated organs add domain-shift noise.

## Target-specific residual activation result

The final experiment reserved 20% of eICU development patients before any
model fitting. The recurrent graph was identity by default, each target used
its own actual-time tolerance, and every persistent correction was a
target-specific zero-start residual around the owner-organ JEPA.

| Cell | eICU gate residual | Base | Relative gain | Activated | MIMIC raw residual | MIMIC safe output | Persistence |
|---|---:|---:|---:|---|---:|---:|---:|
| glucose@3h | 0.7804 | **0.7756** | -0.61% | no | 0.8650 | **0.8512** | 0.8917 |
| bicarbonate@3h | 0.5048 | **0.5033** | -0.29% | no | 0.4134 | **0.4128** | 0.4201 |

Both eICU paired patient bootstraps failed, so both residuals were disabled
before reading the MIMIC result. The safe external prediction was therefore
exactly the owner-organ JEPA. Glucose still significantly beat persistence;
bicarbonate retained a favorable point estimate but its confidence interval
crossed zero. This is not a persistent-state promotion. It proves that the
activation boundary prevents a weak shared-state correction from degrading a
useful owner model.

The bicarbonate shared representation still passed the stricter all-history
masked-recovery test on MIMIC (0.9326 versus base 0.9963 and population center
0.9494). The body state therefore contains recoverable current physiology,
but the learned transition from that state to an incremental future residual
is not stable enough to activate.

## Direct state-change decoder experiment

The failed shared-future-latent to target-latent conversion was removed. The
new decoder maps body-state change, owner-organ change, actual target elapsed
time, anchor value, and observation mask directly to a zero-start value delta.
Everything else remained fixed.

With the original 1,000-stay development budget, this corrected the direction
of the eICU signal but did not make it statistically stable:

| Cell | Direct decoder | Owner base | Relative gain | 95% paired CI | Activated |
|---|---:|---:|---:|---:|---|
| glucose@3h | **0.7753** | 0.7756 | +0.05% | [-0.0043, 0.0034] | no |
| bicarbonate@3h | **0.4995** | 0.5033 | +0.76% | [-0.0148, 0.0078] | no |

The full 4,559-stay eICU cohort tested whether this was merely low power. It
was not: glucose was 0.7023 versus base 0.7008, and bicarbonate was 0.4721
versus base 0.4661. Both residuals remained disabled. Full-cohort MIMIC raw
residuals were 0.8644 for glucose and 0.4365 for bicarbonate, versus safe owner
outputs 0.8663 and 0.4222 respectively. The opposite target behavior across
databases confirms target- and domain-dependent residual heterogeneity rather
than a remaining decoder-capacity problem.

Canonical direct-decoder artifacts:

- `artifacts/persistent_direct_delta_glucose_3h_external_20260830.json`
- `artifacts/persistent_direct_delta_bicarbonate_3h_external_20260830.json`
- `artifacts/persistent_direct_delta_glucose_full_eicu_3h_external_20260830.json`
- `artifacts/persistent_direct_delta_bicarbonate_full_eicu_3h_external_20260830.json`

Canonical artifacts for this final experiment:

- `artifacts/persistent_residual_dev_gated_glucose_3h_external_20260830.json`
- `artifacts/persistent_residual_dev_gated_bicarbonate_3h_external_20260830.json`

The target-specific future labels use measured `actual_delta_t`: fast targets
have predeclared tolerances rather than sharing one nominal-horizon window.
Anchor-pre treatment context is also fail-closed. Both databases contain
treatment evidence, but the current eICU cohort has no observed dose or rate
for insulin, dextrose, fluids, bicarbonate, or ventilation, while MIMIC does.
Using those quantitative fields would encode database identity. The external
run therefore uses only the administration-presence contract common to both
databases. Complete dose/rate context remains an extractor/data-parity gap,
not a completed feature.

## Architecture consequence

The long-term model keeps one persistent patient state and anatomical organ
states, but changes the forecast boundary:

1. the shared state learns coherent current-body reconstruction and recurrent
   evolution;
2. each target head reads its anatomical owner organ by default;
3. a non-owner organ can affect that target only through a predeclared sparse
   adapter that wins its own patient- and hospital-held-out ablation;
4. the release persistence anchor remains outside the candidate and cannot be
   removed by a structural test.

This is not a return to unrelated expert models. Every owner head still reads
states generated by the same patient trajectory. The restriction applies to
which organ evidence may move a target, not to whether the patient state is
shared.

## Promotion verdict

`glucose@3h` passes external point accuracy and candidate-specific conformal
coverage against the older unrestricted baselines. The owner-organ version is
stronger than the unrestricted persistent version, but a fair owner-organ base
is stronger still. Therefore no cell has proved incremental future value from
the persistent transition itself. The architecture remains `candidate_only`
because:

- masked recovery does not beat the population-center placebo;
- no cross-organ edge has earned promotion;
- MIMIC-IV has already informed architecture development;
- no untouched HiRID, AmsterdamUMCdb, or equivalent lockbox is available in
  the workspace.

`bicarbonate@3h` proves masked shared-state information but remains a point
near-pass. Urine-output candidates are rejected. The implemented candidate now
keeps shared state for reconstruction while making recurrent future
corrections zero-start, target-specific, and development-gated against the
owner-organ base. Neither tested correction earned activation. All claims are
factual only;
`causal_claim_allowed` remains `false`.
