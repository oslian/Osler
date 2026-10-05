# Complete Seven-Split Shared-State Reconstruction

All prespecified seeds7/19/31/43/59/71/83 completed matched real, process-only
and placebo training and evaluation. This is not a forecast promotion study.
Machine summary: `artifacts/native_reconstruction_seven_split_summary_20260917.json`.

## Last Split And Recovery

Seed83 completed eight epochs in all arms. Real selects8 (tune2.103858), process5
(2.216308), placebo0 (2.259817). Per epoch25,994 distinct fit pairs and35,492
updates; total283,936updates per arm. Anion gap source tune28 is unsupported.
There were13,784 heldout pairs. Nine targets meet all nominal comparisons in
both roles; no nominal harms versus the three references in this split.

Temporary cleanup removed the query file during report serialization, after
all predictions were saved. Restored missing files from the previously verified
durable backup without overwriting surviving files. The recovery finalizer checks
original source hashes, every shard, patient-target keys, truths, fit medians and
completed checkpoint bindings before calculating the unchanged paired statistics.
No retraining, new selection, changed source hash or discarded predictions.
Seed83 checkpoints/predictions have a durable Git-ignored copy.

## Full Catalog, Not Only Winners

Success here means real beats population, trained process-only and placebo with
nominal95% paired patient intervals in BOTH evaluation and hospital roles.

| Target | Successful splits /7 |
|---|---:|
| Albumin, AST, total protein | 7 |
| ALT | 6 |
| Bicarbonate, heart rate | 4 |
| Alkaline phosphatase, calcium, PTT | 3 |
| Chloride, creatinine | 2 |
| Hematocrit, hemoglobin, phosphate, platelets | 1 |
| Remaining24 catalog targets | 0 |

Zero is not evidence of no information. Some targets lack support, some beat
population but not the process comparator, and others vary across trained fits.
Population and selected epoch0 placebo are identical in several splits, not
independent evidence. The full summary retains each comparator's mean and CI,
and explicit unsupported cases, rather than counting unavailable tests as passes.

## What This Locates

Complete exposure and correct numerical/process wiring allow shared-state
reconstruction to learn repeatable incremental numerical information for a
subset. A universal empty or unusable shared-state claim is not supported.
It has not established broad stable gain across the full catalog. Repeated-split
success is descriptive: the same hospital group is reused, patients overlap
across fits, and many hypotheses are tested. Multiplicity and untouched external
evidence remain unresolved. These counts must not be called runtime promotion.

The experiment changes patient membership AND state initialization/sampling
together. Therefore it cannot identify which source drives unstable targets.
Equal eight-epoch budgets do not prove convergence; several models select the
last epoch. The next diagnostic must hold patient roles fixed while varying
optimization randomness, or hold initialization/sampling policy fixed across
patient roles, and compare per-target source/heldout behavior. This is distinct
from repeating another architecture sweep or selectively extending failed seeds.

All targets remain in the goal. Future trajectories, multi-step consistency,
calibrated intervals and broader external validation remain separate unfinished
requirements. No release weights, registry, intervals or serving changes.
