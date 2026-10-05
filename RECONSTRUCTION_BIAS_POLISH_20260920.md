# All-Target Source-Fit Bias Polish

## Fixed Repair

Frozen bounded7 body state and all non-bias weights. Full source10,737patients,
247,611accepted fitting pairs across33trained targets. For each target solve
the convex one-dimensional SmoothL1(beta1) bias optimum, using expected bounded
site weights and equal selected/broad group mass. All33corrections frozen and
saved before loading evaluation predictions. No per-target acceptance selected
from evaluation; six untrained targets retain their original weights.

Expected target-equal fitting SmoothL1 mean1.258742271->1.246812201. Each target
loss non-increase and root derivative tolerance checked. Direct candidate versus
parent tensor comparison confirms ONLY readout.bias changed. Independent scalar
biases do not change shared body state, ordering of predictions, or other rows
of the readout. Bias-only algebra verified with a focused test.

## Reused Evaluation

These are the prior fitting-disjoint evaluation cohorts, NOT the90/3086source
selection pairs and NOT a new untouched external lockbox. Patient identity
exclusion, exact original per-target errors, and input hashes verified.

| Group | Target-equal MAE before | After | Numerical gains /33 | Incremental patient+site passes |
|---|---:|---:|---:|---|
| Selected |2.909265049|2.895305785|22|bicarbonate|
| Broad |1.280860946|1.278736557|18|bun|

Twelve targets improve numerically in BOTH groups: albumin,alkaline_phos,alt,
bicarbonate,bun,hematocrit,lipase,magnesium,o2sat,respiratory_rate,total_protein,wbc.
No target passes the incremental gate in both groups. Gate retains>=30patients,
>=3sites, negative patient/site means, and both one-sided pvalues<.05/39.
This is NOT a full population/placebo/interval promotion gate.

BUN selected176patients/11sites: MAE.943422->.880081; patientp.000366,
sitep.09672. Broad3318/7sites:.958874->.903870; sitep.000770, passesincremental.
Thus selected evaluation also has an average benefit, unlike the earlier90person
selection result, but cross-site stability remains unproven. Do not conflate
these cohorts or infer one stable opposite physiological regime.

Bicarbonate selected174/11sites:1.016914->.936788, incrementally passes.
Broad2947/7sites:.904271->.895926 but equal-site mean delta+.003388, fails.
Creatinine slightly worsens in both groups, despite optimized training loss.

## Decision

This repairs a real training-data readout optimization gap without changing
shared-state architecture, but does not solve conditional ranking or cross-site
generalization. Retain the candidate privately; no release weights, registry,
intervals or serving behavior changed. No new promotion. Avoid selecting only
the favorable outputs using these evaluation outcomes. A legitimate next step
must address representation/conditional error on source data, not more scalar
calibration freedom on reused evaluation patients.

Three focused tests pass; full run exit0, all data-loss roots and exactbaseline
assertions pass. Patient predictions cached in ignored private directory for
future diagnostics. Evidence:
artifacts/native_reconstruction_bias_polish_20260920.json.
