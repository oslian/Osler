# Interval-Normalized Urine Output Findings (2026-08-02)

## Question

Why did urine-output forecasts repeatedly approach promotion but fail to
generalize across patients, hospitals, and care units?

The failure was traced to measurement semantics rather than JEPA capacity.
The previous extractors treated every charted urine volume as an instantaneous
hourly value. In eICU, these values usually summarize irregular collection
intervals and therefore are not comparable without time normalization.

This work is factual research only. It does not support causal, treatment, or
clinical claims.

## Extraction Repair

The raw eICU events showed a median collection interval of 2 hours and a 90th
percentile of 8.65 hours. The old extraction also admitted non-volume records
such as urine counts, occurrences, unmeasured voids, and mixed urine/stool
labels. One DKA path additionally divided every value by an arbitrary constant
of four.

The canonical repair now:

- retains only genuine urine-volume labels and paths;
- excludes counts, occurrences, unmeasured voids, and mixed outputs;
- aggregates duplicate records at the same stay and timestamp;
- divides volume by the actual preceding collection interval;
- excludes first events and intervals longer than 24 hours;
- emits `urine_output` in `mL/hour` with collection interval and quality
  provenance.

The repaired distribution changed from a median charted volume of 300 mL to a
median interval-normalized rate of 67.16 mL/hour. The 90th percentile changed
from 700 mL to 214.29 mL/hour. Current/future correlation fell from 0.430 to
0.188, confirming that the old feature partly encoded collection practice
rather than a stable physiological rate.

## Cohort And Separation

The repaired event ledger contains 5,402,168 events from 2,520 encounters and
1,841 patients. It produced 309,258 canonical examples from 2,388 encounters
and 1,802 patients.

Formal evaluation recreated the original 1,000-stay development sample and
then removed every stay belonging to those patients. The resulting holdout
contains 144,672 rows, 1,079 stays, and 957 patients, with zero patient overlap
with development.

## Patient-Heldout Result

The same core-12 sparse-distributional JEPA configuration was evaluated over
seven patient-disjoint seeds. `urine_output@3h` passed every patient gate in
all seven splits:

| Requirement | Result |
|---|---:|
| Beats validated teacher | 7/7 |
| Beats persistence | 7/7 |
| Beats capacity-matched no-expert ablation | 7/7 |
| Patient-equal conformal coverage | 7/7 |
| Median MAE delta vs teacher | -0.5191 mL/hour |
| Median MAE delta vs persistence | -0.5308 mL/hour |

Across the seven splits, direction accuracy ranged from 0.646 to 0.686 and
delta correlation from 0.288 to 0.692. Latent SVD stability passed in all
seven splits.

## External Gate

The fixed candidate was retrained inside each external split and evaluated on
held-out hospital, care-unit, and forward-time cohorts. A 45% patient-disjoint
calibration allocation was used because the small care-unit calibration group
did not provide enough finite-sample resolution at 20% or 30%. Test labels
were never used to fit the model or conformal quantile.

| Holdout | Candidate MAE | Persistence MAE | Delta | Coverage |
|---|---:|---:|---:|---:|
| Hospital | 81.8618 | 82.2529 | -0.3911 | 0.8773 |
| Care unit | 76.0753 | 76.1407 | -0.0654 | 0.9272 |
| Forward time | 82.8300 | 83.1812 | -0.3512 | 0.8953 |

All three domains passed teacher, persistence, no-expert, module-balanced,
paired-bootstrap, and conformal requirements. Direction/delta scoring and the
SVD audit then completed the materialized promotion gate.

Because calibration allocation was iterated after inspecting earlier external
coverage results, this remains a validated factual research cell, not a
product or clinical validation. A fresh database or untouched future cohort is
still required for a stronger external claim.

## Decision

`urine_output@3h` is now a validated target-gated joint-JEPA research source.
It may move only when all of the following match:

- registry schema `whole_body_promotion_registry.v2`;
- input contract `joint_asof_whole_body_state_interval_urine.v1`;
- urine-output unit `mL/hour`;
- checkpoint status `validated_target_gated_joint_research_only`;
- target/horizon cell `urine_output@3h`.

Every mismatch falls back to persistence. The old raw-volume urine artifacts
are incompatible and must not be routed into this candidate.

## Meaning

This result identifies one concrete reason other target cells can remain
candidate-only: the model cannot learn a stable physiological relation when
the target silently mixes amount, duration, and charting workflow. Repairing
the measurement object produced a new validated cell without enlarging the
latent or weakening any gate.

It does not prove that every remaining failure is an extraction bug. It does
show that future target work should audit units, collection intervals,
aggregation windows, and workflow labels before changing model architecture.

## Artifacts

- `osler_jepa/urine_output.py`
- `interval_urine_validated_registry_20260802.json`
- `whole_body_runtime_candidate_catalog_v2.json`
- aggregate event and gate reports under the external artifacts directory

Verification on 2026-08-02: the complete `medical_jepa` unit suite passed
344/344 tests. A runtime smoke test confirmed that the matching contract and
`mL/hour` unit can move the validated cell, while legacy contract/unit input
falls back to persistence with explicit rejection reasons.

## Aligned-Treatment Addendum (2026-08-02)

A subsequent audit found that canonical eICU measurements used a 1970
synthetic origin while medication, infusion, and procedure events used 2100.
The measurement-only result above is still valid for its own contract, but it
did not contain aligned treatment history.

After rebuilding every event kind on the same synthetic timeline, the
treatment-aware candidate validated `urine_output@1h`, not 3h. The two cells
belong to different input contracts and must not be merged or substituted.
See `EICU_EVENT_TIME_ALIGNMENT_FINDINGS_20260802.md`.
