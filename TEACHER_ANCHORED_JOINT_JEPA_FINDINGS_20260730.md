# Teacher-Anchored Joint JEPA Findings

Date: 2026-07-30

## Question

Can whole-body JEPA accuracy improve without forcing one shared latent to
replace the already validated target-specific factual router?

## Candidate architecture

The experiment kept the current validated ridge/persistence router as a fixed
teacher anchor and allowed JEPA to learn only a bounded residual:

```text
validated factual router
        +
zero-initialized bounded JEPA residual
        =
candidate factual prediction
```

The residual candidate contained:

- a shared slow whole-body context latent;
- a target-specific fast future latent;
- a learned target/horizon fast-versus-slow mixture;
- an optional patient/target/horizon-specific cross-organ route;
- an optional PCGrad optimizer across target losses;
- persistence and teacher fallback outside validated cells.

The adapter is exactly equal to the teacher at initialization. It cannot
silently worsen a forecast before learning evidence for a residual.

## Data and comparison

- 63,228 canonical event-ledger rows;
- 480 patients;
- 12 hospitals;
- 11 organ-system modules;
- six targets: glucose, bicarbonate, creatinine, potassium, MAP, heart rate;
- six horizons: 1h, 3h, 6h, 12h, 24h, 48h;
- seven patient-heldout seeds;
- strict hospital, care-unit, and forward-time holdouts;
- untouched calibration rows for split-conformal intervals.

Three variants used identical data and splits:

1. `hierarchical`: shared slow + target-specific fast latent;
2. `dynamic_route`: hierarchical plus learned cross-organ routing;
3. `dynamic_route_pcgrad`: dynamic routing plus PCGrad.

## Bounded seven-seed result

Out of 36 evaluated target-horizon cells:

- `hierarchical` beat both the teacher and persistence in 15 cells in all
  seven patient splits;
- `dynamic_route` did the same in 15 cells;
- `dynamic_route_pcgrad` did the same in 14 cells.

The repeatable gains concentrated in dense targets, including multiple
glucose, MAP, heart-rate, and potassium horizons.

PCGrad did what it was designed to do at the optimizer level:

- mean negative target-gradient fraction before projection: 0.110;
- after projection: 0.000.

It did not improve the number of robust factual cells. Gradient conflict was
real, but it was not the main accuracy bottleneck.

## Cross-organ routing ablation

The dynamic route had to beat the identical model with its explicit organ
route disabled.

- no target-horizon cell passed this ablation in 7/7 splits;
- the best PCGrad route result was potassium at 48h, passing 5/7;
- the best ordinary route results passed at most 4/7;
- the highest-weight source organ changed across patient splits.

Therefore explicit dynamic cross-organ routing remains `candidate_only`.
No new cross-organ interpretation is allowed.

## External factual gate

The plain hierarchical candidate was then retrained and evaluated on:

- held-out hospitals;
- held-out ICU care units;
- later within-stay time points;
- module-balanced patient subsets;
- independent 90% split-conformal calibration.

Three cells passed every external point-performance and uncertainty gate:

- `heart_rate@6h`;
- `heart_rate@12h`;
- `map@3h`.

Held-out hospital MAE improvements over teacher/persistence were:

- heart rate at 6h: 9.398 to 9.165;
- heart rate at 12h: 11.020 to 10.755;
- MAP at 3h: 11.489 to 11.240.

All three also passed care-unit and forward-time patient bootstrap checks.

## Direction, delta, and latent stability

Across all seven patient splits:

- direction accuracy ranged from about 0.595 to 0.641;
- delta correlation ranged from about 0.27 to 0.38;
- patient-level bootstrap intervals were below zero;
- learned effective rank ranged from 12.65 to 18.34;
- fit/test rank ratio ranged from 0.82 to 1.16;
- principal-subspace overlap ranged from 0.795 to 0.981.

The learned representation was active, stable, and non-collapsed.

## Final conformal boundary

The point model passed, but strict patient-split uncertainty stability did not:

- `heart_rate@6h`: conformal band passed 5/7 splits;
- `heart_rate@12h`: passed 6/7;
- `map@3h`: passed 5/7.

Failures were over-coverage, not under-coverage. The intervals were slightly
too wide in the failing splits, with coverage approximately 0.931-0.947
instead of the required 0.87-0.93.

The compiled v2 runtime registry therefore validates zero cells and remains
`candidate_only`. No threshold was relaxed.

## Method verdict

### Keep

- validated-router teacher anchor;
- zero-initialized bounded residual;
- shared slow plus target-specific fast JEPA latents;
- per-cell validation and persistence fallback.

These produced robust point-accuracy gains.

### Keep as research diagnostics

- PCGrad: removes gradient conflict but did not improve final accuracy;
- dynamic routing: contains weak increments but no 7/7 stable organ edge.

### Remaining work

The nearest remaining problem is interval calibration, not point prediction
or latent collapse. A future candidate may use calibration-only adaptive or
normalized conformal methods, but it must select interval parameters without
using held-out test outcomes and must rerun the same seven patient and three
external holdouts.

## Safety boundary

- factual observation and forecasting only;
- no treatment-effect or counterfactual claim;
- no clinical recommendation;
- no cross-organ attribution from this candidate;
- current validated router remains active;
- every unvalidated cell falls back to persistence or missing.

Verification: the complete `medical_jepa` unit suite passed 311/311 tests
after this implementation.

## Calibration follow-up (2026-07-31)

The interval boundary was tested rather than relaxed.

Two calibration-only methods were implemented first:

1. patient-disjoint adaptive nominal-coverage selection;
2. patient-grouped cross-fit normalized conformal.

Both were rejected. The first reduced the effective calibration sample and
made the final quantile unstable. The second learned an error-scale model
that varied too much across patient splits, producing both under- and
over-coverage. Neither method was used for promotion.

The original seven development splits showed that fixed nominal 90%
conformal produced mean empirical coverage of 91.85%. A second, disjoint set
of seven split seeds tested nominal 88% and showed that it was too narrow.
Those two development batches fixed the final nominal coverage at 89%.

A third set of previously unused patient split seeds
(`191, 211, 223, 239, 251, 269, 281`) was then run without further tuning.
The result was:

- `heart_rate@6h`: point accuracy 7/7 versus both teacher and persistence,
  conformal coverage 7/7 within 0.87-0.93;
- `heart_rate@12h`: point accuracy 7/7, conformal 5/7;
- `map@3h`: point accuracy 7/7, conformal 4/7.

Only `heart_rate@6h` proceeded to the external gate. It passed:

- held-out hospital: MAE 9.398 to 9.163, coverage 0.896;
- held-out care unit: MAE 8.293 to 8.116, coverage 0.878;
- forward time: MAE 9.327 to 9.238, coverage 0.872.

All three external patient-level bootstrap intervals excluded zero in favor
of the candidate. Direction accuracy and delta correlation passed all seven
final patient splits, and latent SVD stability remained 7/7.

The fail-closed v2 registry therefore promotes exactly one factual research
cell:

```text
heart_rate@6h -> teacher_anchored_hierarchical_joint_jepa
```

Every other target-horizon cell remains on the validated router or
persistence fallback. This is a target-gated research promotion, not a
whole-checkpoint, causal, cross-organ-attribution, or clinical promotion.

Canonical registry:
`teacher_anchored_joint_jepa_validated_registry_20260731.json`.

Verification on 2026-07-31: the complete `medical_jepa` unit suite passed
315/315 tests.

## Full-cohort patient-cluster follow-up (2026-07-31)

The earlier 480-patient audit was not the largest available eICU cohort. The
canonical ledger already contained 4,564 encounters from 3,293 patients; the
smaller result came from `--max-stays 500`. The ledger and event examples were
rebuilt from the original eICU files into a persistent artifact directory:

- 5,000 extracted stays and 3,347 patients;
- 10,698,009 timestamped events, including 114,057 medication events;
- 581,976 target-specific event examples;
- 3,293 patients, 4,564 eligible encounters, and 2,053,001 strict future
  labels before the joint-loader organ-presence filter;
- 566,376 final joint rows from 3,290 patients and 12 hospitals.

Conformal calibration was also changed from row-weighted to target/horizon
patient-cluster calibration. Every patient contributes total calibration
weight one, so long and densely sampled ICU stays cannot dominate interval
width. Coverage is judged by patient-equalized coverage; row coverage remains
diagnostic only.

Across seven new patient splits (`307, 331, 353, 379, 401, 421, 443`), nine
target-horizon cells passed all three internal requirements: 7/7 versus the
validated teacher, 7/7 versus persistence, and 7/7 patient-cluster conformal.
The external hospital, care-unit, forward-time, module-balanced, bootstrap,
direction, delta-correlation, and conformal gates retained three:

```text
creatinine@6h
creatinine@12h
heart_rate@3h
```

External MAE and patient-equalized 90% coverage were:

| Cell | Holdout | Candidate | Teacher/persistence | Coverage |
|---|---|---:|---:|---:|
| creatinine@6h | hospital | 0.2294 | 0.2316 | 0.9229 |
| creatinine@6h | care unit | 0.2410 | 0.2457 | 0.9255 |
| creatinine@6h | forward time | 0.2076 | 0.2116 | 0.9038 |
| creatinine@12h | hospital | 0.2528 | 0.2549 | 0.9220 |
| creatinine@12h | care unit | 0.2615 | 0.2659 | 0.9214 |
| creatinine@12h | forward time | 0.2101 | 0.2140 | 0.9067 |
| heart_rate@3h | hospital | 7.2237 | 7.3318 | 0.8764 |
| heart_rate@3h | care unit | 7.1654 | 7.2645 | 0.9077 |
| heart_rate@3h | forward time | 7.4380 | 7.5129 | 0.8712 |

Direction accuracy was 0.581-0.647 and delta correlation was 0.084-0.325
across the retained cells and seven patient splits. Latent SVD stability
passed 7/7: effective rank was 9.21-15.13 and principal-subspace overlap was
0.988-0.998.

This resolves two earlier uncertainties:

1. the shared whole-body representation can add stable factual value beyond
   the target router, but only in selected target/horizon cells;
2. the earlier interval instability was partly a small-cohort and row-weighted
   calibration problem, not evidence that the JEPA point signal was false.

It does not validate universal cross-organ routing. The dynamic organ route
still failed its own ablation, so the three cells are target-gated factual
research outputs with no cross-organ attribution, causal, or clinical claim.
Every other cell continues to use the validated router or persistence.

Long-running audits now write one persistent checkpoint per seed or external
split and support `--resume`; a runtime restart can no longer discard all
completed evidence.

Verification after the full-cohort update: the complete `medical_jepa` unit
suite passed 318/318 tests.
