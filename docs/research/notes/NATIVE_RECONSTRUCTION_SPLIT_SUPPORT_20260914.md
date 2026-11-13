# Repeated-Split Support Preflight

Report: `artifacts/native_reconstruction_split_support_20260914.json`.
All frozen sample shards and original cache bindings verified. Seven existing
partitions each contain the same1763patients, with no within-split role overlap.
The same194hospital patients are held out every time. Repeating these splits
is robustness evidence, not seven independent external studies.

The previously frozen39target catalog was retained for audit. Thirty-eight
have >=30fit/tune patients in every split. Anion gap support is:

| Split | Fit | Tune | Evaluation | Hospital |
|---|---:|---:|---:|---:|
| 7 | 104 | 38 | 30 | 89 |
| 19 | 117 | 30 | 25 | 89 |
| 31 | 105 | 26 | 41 | 89 |
| 43 | 102 | 30 | 40 | 89 |
| 59 | 101 | 26 | 45 | 89 |
| 71 | 93 | 39 | 40 | 89 |
| 83 | 112 | 28 | 32 | 89 |

Source eligibility fails the existing30patient rule in31/59/83; evaluation
support fails in19. These are unsupported cases, not accuracy failures. Do not
replace those seeds or lower the rule to manufacture7/7. Fibrinogen, LDH and
TSH retain their previously documented hospital support limitations.

Existing examples can be reused by mapping each patient's frozen shard to its
new role, NEVER by keeping the old shard role as the optimization assignment.
Anchor sampling was fixed before this repartition and must not be redrawn based
on error. Recompute fit medians and train weights separately per split; weights
from seed7 cannot count as newly heldout when testing another partition.

This preflight performs no training, forecast evaluation, promotion or runtime
change. Next execute matched real/process/placebo training under these explicit
role and support constraints, retaining unsupported targets in the audit.
