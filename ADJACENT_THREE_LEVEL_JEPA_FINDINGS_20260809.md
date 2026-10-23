# Adjacent Three-Level JEPA Findings

Date: 2026-08-09

## Question

Can the existing validated target-horizon experts be retained while a new
whole-body model learns only adjacent physiological interactions?

The tested hierarchy is:

```text
validated target expert cells
            |
            v
         system
          <->
          organ
          <->
           body
```

Direct `system -> body` and `body -> system` parameters are forbidden. Body
information can change a target only after passing through an organ token.
Existing expert forecasts enter at the system layer and remain the prediction
anchor, so this experiment does not discard the validated router.

## Implementation

- `osler_jepa/adjacent_hierarchy.py` implements the strict adjacency contract.
- `teacher_anchored_joint_jepa_audit.py` evaluates the hierarchy against the
  validated teacher, persistence, and a matched no-route ablation.
- Missing expert forecasts are represented by the normalized population
  median plus an explicit availability bit. They are never converted to a
  physical zero and cannot propagate `NaN`.
- The residual is bounded and zero-initialized, so an untrained hierarchy is
  exactly the existing expert forecast.
- Training is staged:
  1. train the system-local residual with hierarchy disabled;
  2. freeze the system stage;
  3. train only the `system <-> organ <-> body` residual.
- The seven-seed result below used three epochs for each stage. Subsequent
  expanded comparisons divide one configured adapter epoch budget between the
  two stages so one-stage and staged candidates receive equal total updates.

This staging is essential. Joint training allowed the system-local head to
consume the available signal before the hierarchy learned anything.

## Data And Gate

- 124,890 event-aligned rows
- 916 held-out-capable subjects
- 12 hospitals
- 15 body-system modules
- targets: glucose, bicarbonate, creatinine, potassium, MAP, heart rate
- horizons: 1h, 3h, 6h, 12h, 24h, 48h
- 36 target-horizon cells
- seven patient split seeds: 7, 19, 31, 43, 59, 71, 89

The hierarchy must beat:

1. the validated expert/router teacher;
2. persistence;
3. the same trained model with the organ/body route disabled;
4. the patient-cluster conformal coverage gate.

The no-route comparison is the direct test of whether organ/body coupling adds
value beyond the trained system layer.

## Iteration Results

| Variant | Stable cells beating no-route | Interpretation |
| --- | ---: | --- |
| One-stage shared feedback | 0 / 36 (3 seeds) | System-local prediction consumed the signal. |
| Target-specific feedback | 0 / 36 (3 seeds) | More target capacity did not solve competition. |
| Staged system then hierarchy | 11 / 36 (7 seeds) | Adjacent coupling now adds stable information. |

For the final seven-seed run:

- 21 / 36 cells beat the validated teacher in all seven splits.
- 20 / 36 beat persistence in all seven splits.
- 11 / 36 beat the matched no-route ablation in all seven splits.
- 9 / 36 passed conformal calibration in all seven splits.
- 3 / 36 passed all four core factual gates in all seven splits.

## Core-Pass Cells

| Cell | Median MAE delta vs teacher | Median MAE delta vs no-route |
| --- | ---: | ---: |
| heart_rate@12h | -0.5053 | -0.2310 |
| map@3h | -0.4095 | -0.1790 |
| map@12h | -0.5870 | -0.2719 |

Negative delta means lower MAE. These three cells provide the first stable
evidence in this experiment that an adjacent organ/body path adds information
beyond both the expert forecast and the trained system-only model.

Additional cells repeatedly benefited from the organ/body route but did not
pass every calibration or factual gate. Examples include glucose at 6h/48h,
heart rate at 6h/24h/48h, MAP at 48h, and potassium at 1h/3h. They remain
candidate-only.

## Interpretation

The original premise is partly supported:

- A single undifferentiated whole-body predictor was too competitive and
  unstable across targets.
- Preserving validated experts at the system layer avoids throwing away known
  good cells.
- Restricting information flow to adjacent layers is auditable and prevents a
  direct whole-body shortcut.
- Separating system learning from hierarchy learning is what made the organ
  and body representations useful.

The result does **not** show that every physiological target benefits from
whole-body coupling. Benefit is target- and horizon-specific, strongest here
for perfusion and heart-rate dynamics.

## Promotion Boundary

Status: `candidate_only`.

No runtime source was changed. Before any of the three cells can replace an
existing source, they still require the established hospital/care-unit/time
held-out audit and final artifact audit. Failure at any additional gate keeps
the current expert or persistence fallback.

This is factual observation and forecasting only:

```text
causal_claim_allowed: false
clinical_promotion_allowed: false
```

## Artifacts

- `adjacent_three_level_staged_1000_7seed_20260809.json` is the canonical
  seven-seed result.
- `adjacent_three_level_staged_1000_3seed_20260809.json` is the discovery run.
- Earlier one-stage files are retained only as negative experimental evidence;
  they are not runtime artifacts.
