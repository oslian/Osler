# Adjacent Three-Level JEPA Expanded Findings

Date: 2026-08-09

## Purpose

Evaluate the strict `system <-> organ <-> body` hierarchy beyond the original
six-target screen, compare it with the previous joint whole-body model, and
identify whether any improvement survives patient, hospital, care-unit, and
forward-time splits.

## Architecture Under Test

```text
validated expert cells -> system layer
                           <->
                         organ layer
                           <->
                          body layer
```

- No direct system/body edge exists.
- The validated router remains the system-layer anchor.
- The system stage is trained first and frozen.
- The hierarchy stage then learns only the remaining residual.
- A matched no-route ablation tests whether organ/body information adds value.
- One total adapter epoch budget is split across stages for fair comparison
  with the one-stage baseline.

## Expanded Data Coverage

The canonical event data contain:

- 124,890 aligned rows in the 1,000-stay comparison subset;
- 916 subjects;
- 12 hospitals;
- 15 body-system modules mapped to 11 canonical organ groups;
- 53 canonical targets;
- six horizons: 1h, 3h, 6h, 12h, 24h, 48h.

Forty-four targets had enough measured future support for at least one formal
horizon. Deep sparse targets such as cortisol, free T4, myoglobin, and
ionized calcium remained support-limited or too unstable for the formal gate.

## Same-Split, Equal-Budget Discovery

The first expanded comparison used all 53 targets, seed 7, identical rows,
identical teacher forecasts, and equal total adapter epochs.

| Metric | Adjacent three-level | Previous one-stage hierarchy |
| --- | ---: | ---: |
| Evaluable target-horizon cells | 189 | 189 |
| Beat validated teacher | 133 | 131 |
| Beat persistence | 133 | 133 |
| Passed conformal gate | 91 | 90 |
| Beat matched no-route | 128 | Not an identifiable hierarchy test |

The new model did not create a large one-split point-accuracy jump. Its main
advance was architectural identifiability: for 128 cells, removing the
organ/body route made the same trained model worse. The previous model mixed
all information before prediction and could not provide this layer-specific
evidence.

Sixty-nine cells passed all four discovery gates, distributed across 23
targets. This was treated only as screening evidence and was not promoted.

## Seven-Seed Expanded Gate

All 23 targets with at least one discovery core-pass cell were then rerun over
seven patient seeds and all six horizons. This produced 138 evaluated cells.

| Gate | Cells passing all 7 seeds |
| --- | ---: |
| Beat validated teacher | 60 / 138 |
| Beat persistence | 56 / 138 |
| Beat matched no-route | 46 / 138 |
| Passed conformal calibration | 10 / 138 |
| Passed all four gates | 3 / 138 |

The three complete patient-split cells were:

| Cell | Median MAE delta vs teacher | Median MAE delta vs no-route |
| --- | ---: | ---: |
| heart_rate@12h | -0.5032 | -0.2156 |
| map@3h | -0.4086 | -0.1759 |
| map@12h | -0.5787 | -0.2667 |

Negative delta means lower MAE.

## Near-Pass Directions

Several directions showed stable point-forecast and hierarchy value but did
not pass conformal calibration in all seven splits:

- respiratory rate at 3h, 6h, and 24h: conformal 6/7;
- MAP at 48h: conformal 6/7;
- calcium at 1h: conformal 6/7;
- bicarbonate at 12h: conformal 6/7;
- anion gap at 24h: conformal 6/7;
- glucose at 6h and 48h: conformal 5/7;
- creatinine, magnesium, sodium, urine output, and WBC had reproducible point
  improvements but less stable uncertainty coverage.

These remain candidate-only. The expanded failure is therefore not simply
"no physiological signal": 46 cells repeatedly benefited from hierarchy, but
their uncertainty or split stability was not reliable enough.

## External Split Results

The three patient-split core cells were evaluated on hospital, care-unit, and
within-stay forward-time splits using three model seeds.

### Hospital Held-Out

- `heart_rate@12h`: all four core gates passed 3/3.
- `map@3h`: teacher, persistence, and no-route passed 3/3; conformal passed
  2/3.
- `map@12h`: the factual, route, and conformal gates each passed 2/3.

### Care-Unit Held-Out

No cell passed every gate 3/3.

- `heart_rate@12h`: each core gate passed 2/3.
- `map@3h`: factual and no-route passed 3/3, conformal passed 2/3.
- `map@12h`: factual and no-route passed 2/3, conformal passed 3/3.

### Forward-Time Held-Out

- `heart_rate@12h`: all four core gates passed 3/3.
- MAP retained negative median deltas, but one or more factual/no-route gates
  passed only 2/3.

The most stable expanded candidate is therefore `heart_rate@12h`, but its
care-unit result still blocks promotion.

## Difference From The Previous Whole-Body Model

The previous canonical whole-body gate used 563,736 rows, 3,286 subjects, and
53 variables, yet promoted 0 joint runtime cells.

For the same three targets that passed the new patient gate:

- Previous `heart_rate@12h` beat persistence and its self-only ablation, but
  beat the validated router in 0/7 bootstrap splits.
- Previous `map@3h` and `map@12h` passed the cross-system ablation in only 5/7
  splits and beat the validated router in 0/7 bootstrap splits.
- New `heart_rate@12h`, `map@3h`, and `map@12h` beat the validated teacher,
  persistence, and the trained no-route model in 7/7 patient splits while
  also passing patient-cluster conformal calibration.

The improvement did not come primarily from more parameters. It came from:

1. preserving the validated expert as the system anchor;
2. forbidding the direct system/body shortcut;
3. training the system layer before the hierarchy;
4. measuring hierarchy value against a trained system-only ablation.

## Current Boundary

Status: `candidate_only`.

The experiment establishes partial but real adjacent-layer coupling. It does
not establish a universal whole-body predictor. Care-unit heterogeneity and
uncertainty calibration are now the dominant blockers, not the absence of
cross-organ point-prediction signal.

No runtime registry or checkpoint was promoted. All unsupported cells continue
to use their validated expert or persistence source.

```text
causal_claim_allowed: false
clinical_promotion_allowed: false
```

## Canonical Artifacts

- `adjacent_three_level_all_targets_discovery_20260809.json`
- `adjacent_three_level_expanded_23target_7seed_20260809.json`
- `adjacent_three_level_core_hospital_3seed_20260809.json`
- `adjacent_three_level_core_careunit_3seed_20260809.json`
- `adjacent_three_level_core_time_3seed_20260809.json`
- `whole_body_joint_gate_full_canonical_5000_20260808.json` (previous model)
