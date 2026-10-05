# Transition information restored, accuracy not improved

Fourteen fixed fits complete, seven seeds. All frozen own/original MAEs reproduce
within1e-10 in all role/groups. Same rows, labels, weights and partitions.

| Role/group | Original | Added transitions | Own | Persistence |
|---|---:|---:|---:|---:|
| Source/all | .422354 | .422908 | .424584 | .529313 |
| Source/sequence | .386444 | .394722 | .402279 | .436196 |
| Hospital/all | .366079 | .366660 | .365562 | .464061 |
| Hospital/sequence | .341000 | .343361 | .336886 | .400376 |

Significant gain against original0/7 in every row above. Significant harm1/7
source all/sequence and hospital all,0/7hospital sequence. Against own0/7gain
everywhere, hospital1/7harm both groups. Hospital sequence vs placebo0/7gain,
1/7harm. Persistence hospital7/7both, source all7/7, sequence3/7.

The synthetic summary collision was real, but adding change/reversal timing
does not repair this comparator's incremental prediction. It is not evidence
that all exact event sequences lack information, or a sole cause established.
Do not infer accuracy benefit from invariant tests. No JEPA retraining,
calibration, runtime or promotion change. Reused cohorts, overlapping seeds,
donor-support limitation and unverified treatment availability remain.

Artifact: `artifacts/expanded_insulin_transitions_20260911.json`.
