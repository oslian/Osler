# Failure Localization and Executed Path

## Scope

This audit diagnoses the availability-safe glucose@6h experiment, not every
runtime target. It does not train, promote, or change serving. The new test
uses synthetic inputs on all14 actual-time identity checkpoints; it is a
structural test, not clinical evidence or a new accuracy evaluation.

## First Reproducible Failure

The original32epoch experiment was better with whole context on fitting
patients in6/7 splits, but worse on source-tune patients in7/7 (6/7 significant).
Thus the extra-context failure is already present before database transfer.
Tune data selected checkpoints and is not a pristine test. This is evidence
of a fit/generalization reversal, not proof of one unique mechanism.

Expanded2250patient/64epoch actual-time training still gives whole-vs-own0/7
in both validation roles. Whole MAE36.799hospital/32.067MIMIC versus own
36.319/31.061. These same-task results must not be compared as a paired benefit
against older, availability-unsafe cohorts.

## New Executable Checks

`scripts/audit_encoder_path.py` reloads14 candidate state dictionaries with
identity activation and observed-step history semantics. On the same synthetic
probe, both value loss and JEPA loss have nonzero gradients into non-target
history/current values, the belief filter, variable mixer and module mixer.
EMA teacher parameters remain frozen as intended. This rejects a completely
disconnected context path or universally frozen encoder as the explanation.
It does not establish real-data gradient adequacy or useful information.

`load_release` constructs `JointWholeBodyJEPAWorldModel`, not its persistent
organ-state subclass. Its future predictor uses context latent plus query time;
separate organ arguments are accepted but unused in that transition. Holding
context fixed and adding10 to every separate organ token changes the future
body output by exactly0 in all14 probes. Organ information IS mixed into the
context earlier, so this is NOT absence of cross-variable input. It is absence
of a separate evolving organ-state transition in this particular trial.

The training script supervises glucose value and its encoded future delta.
Other variables supply inputs but not their own future losses in this trial.
Consequently it tests single-target prediction with shared inputs, not jointly
supervised, persistent, multi-organ physiology. This is a scope limitation,
not experimental proof that enabling organ dynamics will improve MAE.

## Cause Hierarchy

| Level | Evidence | Interpretation |
|---|---|---|
| Symptom | Whole context fails to outperform own history | Incremental whole-context benefit not established |
| Earliest measured trigger | Fit benefit reverses on new source patients | Learned additional associations are not stable across patients |
| Proven contributors | Direct residual cap; source-unseen calcium scaling | Controlled repairs improved some errors; neither solved the residual gap |
| Correctness defects | Availability, empty history and analyte identity | Necessary semantic fixes, not all accuracy improvements |
| Remaining hypotheses | Weak conditional information, single-target supervision, inherited representation and heterogeneous trajectories | Not separated into a unique root cause by current evidence |

More patients helped; more optimization helped partly; narrow-window actual
time retraining did not. None establishes a universal data ceiling. A new
architecture or relaxed gate is not justified as a proven cure.

Before a new full-state trial, its acceptance test must assert the exact class,
trained paths and label support. Any joint supervision comparison must use
identical primary glucose queries and source-only auxiliary selection, retain
the matched own-history control, and remain candidate-only. Such a comparison
would test supervision scope; it must not simultaneously change every encoder,
cohort and gate and attribute any gain to organ structure.

Evidence: `artifacts/encoder_path_20260910.json`,
`docs/research/findings/ENCODER_FIT_GAP_FINDINGS_20260910.md`,
`docs/research/findings/ENCODER_ACTUAL_TIME_FINDINGS_20260910.md`.
Regression: `tests/test_encoder_path.py` passes. No runtime, interval or weights
changed. MIMIC remains repeatedly inspected validation, not a new lockbox.
