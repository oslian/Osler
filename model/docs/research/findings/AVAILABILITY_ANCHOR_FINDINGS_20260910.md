# Availability-Anchored Glucose Findings

## Decision

Reconstruction and seven source fit/tune-seed accuracy runs completed. The
glucose-only history ridge beats persistence in both evaluation datasets in
7/7 paired patient bootstrap comparisons. This is a bounded timing/signal
diagnostic, NOT full JEPA retraining, personalization promotion or a runtime
replacement. Release weights, registry, intervals and serving are unchanged.

## What changed

Queries now occur at raw laboratory result availability. Current glucose is the
latest measured known result, not simply the last arriving result. Historical
inputs use only versions available by the query; true same-valued remeasurements
survive. Future truth is newly selected 5-7h after this new query, nearest 6h.
Old cached feature vectors, frozen predictions and old future labels are not used
in fitting or scoring. The old preparation function supplies patient roles only.

All queries remain in the local reconstruction even without a future label.
Scored current values must be no older than6h. Future labels use final recorded
versions retrospectively. eICU revised-entry time is a conservative proxy for
availability, not a complete history of provisional results.

| Role | Queries | Scored pairs | Patients | Actual future hours min/median/max |
|---|---:|---:|---:|---|
| eICU development | 14,929 | 7,140 | 386 | 5 / 5.90 / 7 |
| eICU held-out hospital | 12,994 | 6,077 | 364 | 5 / 5.93 / 7 |
| MIMIC | 20,835 | 6,616 | 889 | 5 / 5.85 / 7 |

Both timestamp checks have zero violations: current available after anchor,
and future measurement at/before anchor. Prior unchanged-anchor MIMIC support
was zero; this reconstruction restores a prospective query task.

## Accuracy

Median across seven source fit/tune seeds, patient-equal MAE in mg/dL:

| Evaluation | Persistence | Current + age ridge | Current + legal history ridge |
|---|---:|---:|---:|
| eICU held-out hospital | 45.878 | 40.318 | 38.874 |
| MIMIC | 37.449 | 35.378 | 34.900 |

History versus persistence: significantly lower error 7/7 in each dataset.
History versus current-only: 7/7 eICU, only3/7 MIMIC. Thus much of the MIMIC
gain is available from current-state correction, not proven incremental
personal dynamics. The history/current-only comparison is an ablation, not
an equal-capacity shuffled-history placebo. Do not interpret it as promotion.

Models and alpha selection use eICU only. Seeds change source fit/tune patient
partitions; external patients remain the same, so these are not seven independent
external replications. MIMIC is repeatedly inspected validation, not a lockbox.

## What this establishes and does not

Established: the availability-time contract can be rebuilt and supports
source-trained forecasts better than persistence on this glucose slice.
Not established: repair improves the old JEPA. Old scores concern different
anchors, labels and selected rows and are not a paired before/after comparison.
No neural recurrent self-learning, cross-organ input, treatment availability,
care-unit/time gate or conformal calibration was tested here. The fixed cohort
was inherited from prior eligibility; no general-population claim is warranted.

The next unresolved implementation is reconstructing other variables and
treatments on their own availability clocks before encoding a new JEPA input.
Do not carry old cached values to the new anchor or use an old calibrator.

## Evidence

- `artifacts/availability_anchor_20260910.json`: seed-level errors, patient CIs,
  support and code/protocol bindings.
- `whole_body_jepa/availability_anchors.py`: query/history/label reconstruction.
- `scripts/audit_availability_anchors.py`: fixed source-only diagnostic.
- `tests/test_availability_anchors.py`: delayed labels, revisions, duplicate
  identity and late-old-measurement regression checks.
- Raw patient reconstructions remain in local temporary storage, outside Git.
