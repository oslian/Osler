# Expanded Source Selection Replay Results

All epoch0-8 scores completed on3323broad source-selection patients, including
the original90 and additional3233. Original90 per-target MAEs reproduce every
recorded epoch within1e-7. Source-role disjointness, shard hashes, epoch bindings
and complete unique prediction pairs checked. Two summary tests pass.

No model weights updated. Only the frozen expanded sequential real trajectory
was replayed, not all candidate families or newly trained process/placebo models.
Six targets without trained heads remain pending;33trained targets scored.

| Source selection scope | Epoch0 macro | Epoch7 macro | Epoch8 macro | Best epoch |
|---|---:|---:|---:|---:|
| Original90 broad | 1.935442 | 1.806171 | 1.819073 | 7 |
| Additional3233 broad | 1.675188 | 1.550719 | 1.549965 | 8 |
| All3323 broad | 1.681707 | 1.557181 | 1.556772 | 8 |

Combining the expanded broad selection with unchanged original selected-group
selection at equal group weight changes bestepoch7->8:1.686014->1.685488,
only.0312%relative improvement. This is a source selection ranking, not a
statistical accuracy gain or independently confirmed promotion. Additional
patients change composition as well as N; original90 is not a random subsample
experiment establishing a sole small-sample cause.

## What This Changes

The broad cohort's current reconstruction is more favorable than the original
small selection sample suggested. On all3323, epoch8 beats epoch0 population
initialization numerically for31/33targets, macro-7.43%. Triglycerides and
troponin remain above initial reference. These are source-selection means,
not patient/site significance, numeric-versus-process attribution or forecast
evidence. Do not count31new promotions.

Glucose illustrates the interpretation change: original90 .713449prior versus
.731145epoch8 (harm); all3323 .848504prior versus.830379epoch8 (gain). Different
cohort composition/support changes the conclusion before any parameter update.
Troponin also has a major distribution difference: prior1.475809original versus
11.875525additional in normalized units. Its original18examples were not a
reliable description of the larger source selection population's error scale.
This does not establish unit error; identical schema and extractor were retained.

## What This Does Not Repair

Epoch8 improves17/33full-broad targets overepoch7, while16worsen. BUN
.004777 andcreatinine+.006949normalized MAE worsen; glucose-.008488 improves.
Thus a larger source selection cohort does not remove shared-checkpoint tradeoffs.
The previously inspected evaluation cohorts were not used or rerun here, and
their old failures are not overturned. Bestatlast epoch is not convergence proof.

Next separate adequate source exposure from selection uncertainty: the fixed
eight-epoch training still leaves most dense broad-fit patients unvisited.
A predeclared longer/full-coverage training comparison may use these source
selection roles but must preserve all original comparisons and report additional
compute, sparse-target repetition and old-cohort retention. Do not choose future
experiments from evaluation labels or retroactively weaken gates.

Evidence: artifacts/native_reconstruction_expanded_selection_20260919.json.
Private epoch predictions are hash-bound under ignored data storage. No runtime,
registry, release weights, intervals, causal claims or promotion changed.
