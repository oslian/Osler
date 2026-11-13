# Equal-seed averaging result

Fixed seven-member mean completed; common hospital role verified identical
and disjoint from every member's fit/tune patients. Source predictions were
not pooled, avoiding cross-seed patient leakage. Hash-bound cached forecasts.

| Hospital group | Whole mean | Own mean | Persistence | Whole-own CI95 |
|---|---:|---:|---:|---|
| All,193patients | .381275 | .399965 | .482238 | [-.039246,.002784] |
| Bracket-after,146patients | .409151 | .434379 | .523661 | [-.049176,.000706] |

Whole ensemble significantly beats persistence in both groups, but not the
matched own ensemble. Compared with individual whole members, significant
gain2/7all and1/7bracket, harm0/7; these are descriptive multiple comparisons.
It does not establish superiority to every member or isolate initialization
from different source training subsets. Small average improvement is not a
complete repair. Do not loosen CIs because an upper endpoint is near zero.

Aggregate `artifacts/rate_jepa_ensemble_20260911.json`. No fitting, runtime,
interval or promotion changes; reused hospital set, pretraining independence
uncertified. This is a candidate ensemble within the same JEPA family, not an
externally confirmed new whole-body capability.
