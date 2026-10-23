# Anatomical Organ Coupling Findings (2026-08-09)

## Taxonomy correction

The previous implementation incorrectly called functional groups such as
`cardiovascular_perfusion`, `acid_base_electrolyte`, and
`hematologic_coagulation` organs. They are physiological systems, not
anatomical organs. Results produced with that taxonomy remain useful as
functional-group coupling evidence, but they are not evidence of anatomical
organ coupling.

The canonical hierarchy is now:

```text
validated target-horizon expert cells
                 |
functional system layer
                 | explicit many-to-many mapping
anatomical organ layer
                 | sparse anatomical edges
whole-body layer
```

There is no direct system/body path.

## Corrected schema

The anatomical layer now contains 14 explicit nodes:

- heart;
- blood vessels;
- lungs;
- kidneys;
- liver;
- pancreas;
- stomach;
- intestines;
- thyroid;
- adrenal glands;
- bone marrow;
- spleen;
- brain;
- skeletal muscle.

Functional systems map to one or more organs. For example, acid-base and
electrolyte control maps to kidneys and lungs; endocrine/metabolic control maps
to pancreas, liver, thyroid, and adrenal glands. Observation ownership remains
disjoint so no measured variable is silently copied into multiple organ
tokens.

## Bounded anatomical-organ experiment

Artifact:
`adjacent_three_level_strict_anatomical_organs_renal_1000_7seed_20260809.json`

- eICU canonical event ledger;
- 1,000 stays and 41,450 rows;
- creatinine at 3h and 12h;
- seven patient-held-out seeds;
- fixed six-epoch adapter budget split system -> organ -> body;
- patient-cluster conformal audit;
- validated router/persistence teacher retained as the prediction anchor.

### Patient-held-out result

| Cell | Full vs teacher | Full vs persistence | Body-mediated vs system-only | Direct organ vs system-only | Direct organ graph vs disabled | Conformal |
|---|---:|---:|---:|---:|---:|---:|
| creatinine@3h | 7/7 | 7/7 | 6/7 | 6/7 | 2/7 | 6/7 |
| creatinine@12h | 7/7 | 7/7 | 7/7 | 5/7 | 2/7 | 4/7 |

Median full-model normalized-MAE improvements versus the teacher were
`-0.0104` at 3h and `-0.0112` at 12h. The body-mediated improvements versus
system-only were `-0.0082` and `-0.0090`, respectively.

Across all seven seeds, the most heavily weighted neighboring organs for the
kidney/creatinine target were consistently liver and heart, followed by either
blood vessels or lungs. This is plausible anatomical attribution, but attention
weight alone is not validation.

## Interpretation

The corrected experiment supports a bounded statement: anatomical organ
states plus the whole-body mediator improve creatinine forecasts beyond the
system-only expert in six of seven 3h splits and seven of seven 12h splits. It does **not** yet prove
that direct organ-to-organ edges add independent predictive value, because the
matched no-organ-coupling ablation passed only two of seven splits.

No cell is promoted. The path selector was not stable across all seeds and
conformal coverage did not pass seven of seven splits. Hospital, care-unit,
and forward-time gates therefore remain closed.

## Safety boundary

- status: `candidate_only`;
- factual observation/forecast only;
- `causal_claim_allowed: false`;
- failed cells remain on the validated router or persistence;
- the pre-correction functional-group results must not be cited as anatomical
  organ-coupling validation.
