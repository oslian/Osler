# MAP Selection Repair Training

## Design

Seven splits x old/expanded fit cohorts x own/full lag histories. Tree settings
unchanged; source-tune rows remain the OLD MAP tune set for both fit policies.
Thus the main intervention is training eligibility, not test or tuning labels.
Source patient weights recomputed in each fit cohort; source channel support
may expand. Old controls reproduce saved predictions exactly on all old
eligible rows. No neural training or new external database evaluation.

## Same Old Scoring Rows

Median patient-equal MAE, mmHg:

| Model | Source evaluation | Held-out hospitals |
|---|---:|---:|
| Old fit own | 8.99978 | 10.39641 |
| Expanded fit own | 8.94096 | 10.15531 |
| Old fit full | 9.42906 | 10.59785 |
| Expanded fit full | 9.14915 | 10.41079 |

Common20k patient bootstrap: expanded-full versus old-full gain4/7source,
1/7hospital, no harm. Expanded-own versus old-own gain2/7source,0/7hospital.
Expanded-full versus expanded-own gain0/7both, sourceharm1/7.

## All MAP-Evaluable Rows

Expanded-full versus old-full gain5/7source,1/7hospital; no harm. Median MAE
source9.15242->8.89262, hospital10.38585->10.21392. Expanded-full versus
persistence gain7/7both, but versus expanded-own gain0/7both. Newly recovered
rows are also reported separately in the aggregate artifact. Patient-equal
means across all/recovered/old subsets are not row-weighted averages, and
patients may contribute to both old and recovered subsets.

## Conclusion

Unrelated future-label selection limited part of source learning under this
controlled tree experiment. Removing it partially improves predictive error;
it does not establish cross-organ incremental value or repair native JEPA.
More repeated rows and somewhat more patients are combined in the intervention;
their independent effects are not isolated. Do not claim data volume alone is
the sole root cause. Native event inputs/retraining remain uncompleted for the
expanded queries. No calibration/runtime/promotion.

Evidence: `artifacts/map_selection_training_20260912.json`,
`scripts/train_map_selection_repair.py`. Patient predictions remain private.
