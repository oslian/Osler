# Expected-error routing experiment

Frozen before running outcomes: five kidney-related 6h targets; existing frozen
JEPA encoder; one common expert per target fit only on expert-fit eICU patients.
Router labels are losses on disjoint router-fit patients. Same corrected context
is used for all arms (chart availability uses storetime; infusion snapshots expire
after 60 minutes and only visible snapshots may end an earlier segment).

Methods: existing probability plus q25-benefit rule, probability alone (q25 ablation),
two conditional mean absolute-error regressors, and direct conditional mean benefit.
Expected-error heads use sklearn HistGradientBoostingRegressor with squared error,
80 iterations, 15 leaves, minimum 40 samples, L2=1; no hyperparameter sweep.
References: always-persistence, always-expert, old rule, matched shuffled-context,
and no-context variants. Each comparator tunes its threshold on the same eICU
selection patients. Margins: 0, .02, .05, .1, .2 times the target dynamic threshold.
Probability cutoffs follow the existing grid. Model selection minimizes patient-equal
MAE; stable/dynamic harms are reported separately without using future state at inference.

The former eICU tune partition is split by patient hash into decision-selection and
interval-calibration partitions. No labels from eICU patient/hospital validation or
MIMIC are used for choice or interval calibration. The primary candidate is selected
between two-risk and mean-benefit on eICU decision-selection MAE before external data.

Report seven patient evaluation resamples (not seven independent training reruns),
paired patient bootstrap, care-unit-balanced and late-stay gates, CRPS, 90% coverage
in .87-.93, per-care-unit coverage, stable/dynamic errors, movement, and diagnostic
oracle choice (uses future truth ONLY as an upper-bound diagnostic). Model gate must
beat all mandatory references and pass CRPS/coverage on all three roles. Independent
runtime promotion is not authorized by this diagnostic: MIMIC has been repeatedly
inspected; base-encoder training independence is not established by this experiment.
No runtime edits, next-measurement task, or causal claims. Artifacts remain candidate-only.

Pre-fit correction: the first run stopped at the patient-overlap assertion before
any expert or selector fit. Patients with any held-out-hospital encounter are now
excluded from every development role, including encounters at other hospitals.
All arms use this corrected split. This supersedes older hospital reports that did
not enforce cross-hospital patient exclusion; they are not clean comparators here.
