# Three-Level External Expert Anchor Findings

## Question

Can the 11 MIMIC-to-eICU externally validated expert cells serve as system-layer anchors, with the strict `system <-> organ <-> body` model learning an additional externally generalizing residual?

## Setup

- Expert fit database: MIMIC-IV only.
- External scoring database: eICU only.
- Continuous expert anchors: 9 cells.
- Retained binary respiratory event experts: 2 cells (`hypoxemia_event@1h`, `hypoxemia_event@3h`).
- Binary probabilities were not mixed into the continuous value loss.
- Dataset: 54,111 rows, 8,166 patients before loader eligibility filtering.
- Model audit: 48,509 eligible rows, three seeds (`7,19,31`).
- Anatomy: 14 canonical organs only; no functional system was represented as an organ.
- Training order: system, then organ, then body.
- Evaluation: MIMIC training and calibration, frozen eICU external scoring.

## Result

| Cell | Beats expert | Beats persistence | Beats no-route | Beats no-organ | Beats no-body | Conformal | Stable coupling path |
|---|---:|---:|---:|---:|---:|---:|---|
| heart_rate@6h | 3/3 | 0/3 | 3/3 | 0/3 | 3/3 | 3/3 | system only |
| heart_rate@24h | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 1/3 | no |
| heart_rate@48h | 3/3 | 3/3 | 3/3 | 2/3 | 1/3 | 1/3 | system only |
| hemoglobin@12h | 1/3 | 3/3 | 0/3 | 1/3 | 0/3 | 1/3 | no |
| map@24h | 0/3 | 3/3 | 0/3 | 0/3 | 2/3 | 1/3 | system only |
| potassium@12h | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 0/3 | no |
| potassium@24h | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 | system only |
| respiratory_rate@3h | 0/3 | 0/3 | 0/3 | 1/3 | 2/3 | 2/3 | system only |
| sodium@24h | 0/3 | 3/3 | 0/3 | 1/3 | 0/3 | 3/3 | system only |

## Interpretation

The expert-anchor design is useful as a system-layer starting point. It produces repeatable system-local residual improvements for several cells, especially heart rate and potassium. However, no cell passed all external point, path-specific ablation, and interval gates.

The result does not establish stable organ/body coupling. Path selection usually reverted to `system_only`; occasional organ or body path selections did not repeat across seeds. `potassium@12h` and `potassium@24h` are the closest candidates, but the organ ablation and conformal gates remain incomplete.

## Decision

- Keep the original 11 externally validated experts unchanged.
- Do not promote any three-level residual from this experiment.
- Keep the two hypoxemia event experts as separate respiratory probability outputs.
- Treat the three-level model as `candidate_only` until one exact cell passes all seven patient seeds plus hospital, care-unit, time, path-ablation, and conformal gates.
- The experiment is factual and research-only; `causal_claim_allowed: false`.
