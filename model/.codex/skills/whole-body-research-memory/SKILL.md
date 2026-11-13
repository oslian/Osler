---
name: whole-body-research-memory
description: Load and maintain the WholeBody-JEPA project state and experiment history before planning, diagnosing, changing, testing, or reporting on this repository.
---

# Whole-Body Research Memory

Use this skill for every task in the WholeBody-JEPA repository.

## Start Every Task

1. Read the applicable `AGENTS.md` files completely.
2. Read `PROJECT_STATE.md` completely.
3. Read the relevant section of `EXPERIMENT_INDEX.md` and the linked findings or protocol before proposing an experiment.
4. If `.codegraph/` exists, use CodeGraph first for code structure, symbol ownership, and call paths. Use `rg` for prose, reports, and exact text searches.
5. When current release counts matter, run `python -m scripts.validate_release`; do not infer them from older prose.

## Evidence Rules

- Treat machine-readable reports as authoritative for exact numbers.
- Distinguish the active runtime from candidates, partial passes, near misses, and rejected experiments.
- A candidate may move one target only after that target passes its complete prespecified gate.
- MIMIC-IV and NWICU have already been inspected repeatedly. Do not call a new method tested on either an untouched external lockbox.
- Keep every output factual and observational. Preserve `causal_claim_allowed: false` unless a separate causal evidence contract is satisfied.
- Do not commit raw patient rows, patient identifiers, or local data paths containing protected data.

## Finish Every Experiment

Update the memory in the same commit:

- Add the question, data roles, fixed comparison, result, decision, and artifact paths to `EXPERIMENT_INDEX.md`.
- Update `PROJECT_STATE.md` when the active release, current bottleneck, validated fact, or next legitimate experiment changes.
- Record negative results. Never silently repeat a rejected method under a new name.
- State whether release weights, registry, intervals, or serving behavior changed.

The goal is not a longer diary. The goal is a compact, auditable map of what is known, what failed, what is active, and what remains untested.
