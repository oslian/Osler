# Raw Lab Identity and Availability Findings

## Decision

Raw-event audit completed. The planned seven-seed accuracy comparison did NOT
start: the unchanged-anchor MIMIC evaluation has zero availability-valid scored
pairs. Source fit/tune support also fails some predefined seed thresholds. This
is `blocked_by_available_time_support`, not a negative accuracy result or model
promotion. No release weights, registry, intervals or serving code changed.

Scope: glucose@6h cached-anchor cohort from the recurrent repair experiments,
not all deployed targets or all MIMIC patients. A raw-lab residual module and
matched once/re-consume control are implemented, but not fitted on real patients
in this experiment. The frozen JEPA baseline's other inputs remain unverified
for available time, so even a later pass here would not prove a clean whole-body
deployment pipeline.

## Identity audit

Retrieved 67,278 eICU and 20,843 MIMIC unit-qualified raw glucose laboratory rows
for the prior patient roles. Source ID and conservative available timestamp are
preserved. All inspected development/hospital/MIMIC role rows had distinct
event versions: exact source-version deduplication removed zero rows.

Consecutive equal-valued readings with different event IDs: 272 eICU development,
214 eICU hospital, 452 MIMIC. These were retained. Equal numbers do not establish
one duplicated measurement or justify dropping a true remeasurement.

Latest-available event reuse across cached queries was 0/1717 development,
2/1500 eICU hospital and 142/6988 MIMIC query transitions with a known event.
This is snapshot reuse, not duplicate source rows. The prior synthetic duplicate
stress failure is real, but does not establish widespread raw duplicates as
the explanation of the old transfer gap.

## First observed contract failure: result availability

For every previously scored row, the exact cached current value and its
measurement timestamp matched a native raw event; the future value/time also
matched. Thus the rejected rows below are not a value-mapping or event-ID
matching failure. The final recorded value was available after the old anchor.

| Role | Old scored rows | Matched current recorded after anchor | Median late delay | Strict scored rows remaining | Primary patients |
|---|---:|---:|---:|---:|---:|
| eICU development | 846 | 644 (76.1%) | 49 min | 202 | 44 |
| eICU hospital | 797 | 614 (77.0%) | 47 min | 183 | 45 |
| MIMIC | 2349 | 2349 (100%) | 64 min | 0 | 0 |

Primary additionally requires a previous matched outcome to have become available.
Its rows are112 development,100 hospital and0 MIMIC. Some eICU source partitions
also fall below50 fit or30 tune patients; thresholds were not relaxed.

The eICU revised offset records entry of the revised value. Using it conservatively
does not assert that no earlier provisional result was available: the historical
earlier version is not supplied in this bounded extraction. In MIMIC, storetime
documents laboratory result availability; charttime usually corresponds to
specimen acquisition. These are different clocks.

References: [eICU lab schema](https://eicu.mit.edu/eicutables/lab/) and
[MIMIC labevents schema](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html).

## Interpretation

Surface: a requested deduplication accuracy comparison cannot be supported on
the old strict scoring anchors.

Direct trigger: requiring available_at <= prediction_time excludes current
results which were only entered later. Old eligibility used measurement age.

Upstream cause established for this slice: observation time and information
availability were conflated. The canonical extraction fabricates source IDs
from row positions and discards available times, so the cache cannot enforce
original-event identity/availability. This does not prove all previous research
is invalid: retrospective prediction conditional on sampled values is a different
task from prospective bedside forecasting. It does mean this slice's previous
scores cannot establish availability-safe online glucose prediction.

Other causes remain unisolated (representation, patient heterogeneity, treatment
capture). No evidence here that deduplication alone improves MAE, and no reason
to discard all JEPA results based on this one bounded audit.

## Repair boundary

Implemented event-version deduplication and observation/query separation pass
synthetic invariants. The once path is invariant to duplicate raw rows and extra
queries; true equal-valued measurements with new IDs and later revisions survive.
The matched control deliberately re-consumes the last event after a query.
Both use the same raw-event stream and neural capacity.

To obtain an honest accuracy comparison, rebuild prediction anchors at actual
information availability, reconstruct all pre-anchor inputs on their own
available clocks, then rebuild future labels from that NEW anchor. Merely
moving an anchor while retaining the old 6h label changes its effective horizon.
Keeping old frozen JEPA features or early error feedback would leave leakage.
This full-cache reconstruction was not completed in this bounded trial.

## Artifacts

- `artifacts/raw_lab_identity_20260910.json`: aggregate result and support failures;
  `training_started:false`, `accuracy_comparison_complete:false`.
- `RAW_LAB_IDENTITY_PROTOCOL_20260910.md`: fixed comparison and acceptance criteria.
- `whole_body_jepa/raw_lab_replay.py`: native-event replay and read-only query head.
- `scripts/audit_raw_lab_identity.py`: original-source audit and support-gated trial.
- `tests/test_raw_lab_replay.py`: identity, revisions, availability and query invariants.

Raw patient rows and IDs remain only in local temporary caches, outside Git.
Full regression: 280 tests passed. `git diff --check` passed; CodeGraph synced.
