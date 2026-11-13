# Expected-error persistence routing: completed matched experiment

## Decision

Candidate-only: **0/5 targets passed the full experimental gate**. The running
release remains unchanged (41 cells, 27 movable, 14 persistence fallbacks).
This experiment tests routing between one fixed eICU-trained expert and persistence,
not a newly trained whole-body encoder or a replacement for the validated runtime router.

The q25 rule was overly restrictive for some targets, but removing it did not solve
individual selection or cross-database transfer. The result does not establish that
all useful pre-anchor information has been exhausted.

## Matched protocol

- Five targets, all at 6h; one frozen shared JEPA representation and one common
  distributional expert per target, identical across routing arms.
- Expert-fit and selector-fit patients are disjoint. Expert fitting uses 5,268-6,677
  eICU patients per target; selector fitting uses 3,440-4,442.
- Separate eICU selection and conformal-calibration patients; no MIMIC tuning.
- Compare old probability-plus-q25, probability alone, two conditional expected
  absolute-error heads, and a direct conditional mean-benefit head.
- For expected-risk methods, also fit no-context and capacity-matched shuffled-context
  controls. Always-persistence and always-expert are mandatory references.
- Select the primary expected-risk method on eICU selection MAE. Mean-benefit is
  selected for creatinine, BUN, urine and potassium; two-error heads for bicarbonate.
- Each MIMIC target has 12,000 evaluated patients, one anchor per patient. These
  target cohorts overlap; this is not 60,000 distinct patients.

See `EXPECTED_ERROR_ROUTING_PROTOCOL_20260905.md` and the frozen development artifact.

## MIMIC results

MAE is in each target's native transition units. Lower is better. This table is
descriptive; statistical results and failures are distinguished below.

| Target @6h | Persistence | Always expert | Old q25 rule | New selected rule | Old move rate | New move rate |
|---|---:|---:|---:|---:|---:|---:|
| Creatinine | 0.18556 | 0.20270 | 0.18557 | 0.19797 | 0.008% | 72.87% |
| BUN | 3.02925 | 3.09671 | 3.02925 | 3.05742 | 0.000% | 63.95% |
| Urine output | 111.91949 | 102.94314 | 110.79825 | 104.69012 | 0.392% | 70.43% |
| Potassium | 0.42890 | 0.38448 | 0.42876 | 0.38493 | 0.225% | 97.80% |
| Bicarbonate | 1.91184 | 1.88599 | 1.91147 | 1.88293 | 0.058% | 78.68% |

### What changed, and what did not

1. The old rule suppressed movement. Removing only q25 also raises movement to
   50.5-84.7%, so this is not solely an effect of the expected-error objective.
2. Urine and potassium pass the MIMIC point comparison against persistence and the
   old rule, including seven patient resamples, unit-balanced and late-stay checks.
   However urine is significantly worse than always-expert (MAE delta +1.747,
   patient bootstrap 95% CI [1.338, 2.162]); potassium does not significantly improve
   on always-expert (delta +0.000453, CI [-0.000090, 0.000997]).
3. Creatinine and BUN are significantly worse than persistence in MIMIC:
   creatinine delta +0.01241, CI [0.01020, 0.01470]; BUN +0.02817,
   CI [0.00844, 0.04831]. They do improve average MAE in eICU, so the preferred
   prediction source does not transfer uniformly between these cohorts.
4. Bicarbonate improves pooled MIMIC MAE vs persistence (delta -0.02891,
   CI [-0.04117, -0.01665]), but fails the late-stay check. Its improvement over
   always-expert is not significant (delta -0.00305, CI [-0.00796, 0.00181]).
5. No primary method passes the complete no-context and shuffled-context comparison.
   Individual-benefit AUC in MIMIC is only 0.478-0.525. AUC measures ranking of
   individual wins, not benefit magnitude; it is a diagnostic, not the objective.

Thus gains on urine/potassium mostly recover the value of using the expert more
often; they do not demonstrate reliable patient-specific routing. The experiment
does not support either blanket claim: "the forecasts are fine, only routing is
wrong" or "the data contains no signal."

## Interval and distribution results

Intervals were calibrated on independent eICU calibration patients, without external
recalibration. All five pass the configured eICU patient/hospital coverage checks;
all five fail the configured MIMIC coverage check.

| Target | MIMIC patient-equal coverage | MIMIC unit-equal coverage | CRPS vs empirical persistence |
|---|---:|---:|---|
| Creatinine | 96.66% | 95.79% | Pass |
| BUN | 95.53% | 96.14% | Fail |
| Urine output | 95.57% | 95.05% | Pass |
| Potassium | 93.67% | 91.12% | Pass |
| Bicarbonate | 95.71% | 94.56% | Fail |

Nominal coverage is 90%, with the existing 87-93% gate unchanged. These failures
are overcoverage, not undercoverage: eICU-calibrated intervals are generally too
conservative on MIMIC under this protocol. No external coverage-based retuning
was performed. CRPS evaluates the unscaled expert/persistence sample distributions;
coverage evaluates their separately conformal-scaled intervals, not the same score.
Individual care-unit sample sizes and coverage are in `report.json`; small units
can have very noisy coverage. Unit-equal coverage is not proof of per-unit validity.

## Correctness fixes made before the completed run

- eICU infusion history now expires a snapshot after 60 minutes and computes its
  endpoint only from snapshots visible at the anchor. A future snapshot cannot
  retroactively prolong past infusion exposure. Explicit zero stops a stream;
  concurrent streams are summed. The TTL is a data-handling assumption, not proof
  that a real infusion stopped after 60 minutes.
- MIMIC bedside features become available at the later of charttime and storetime;
  missing storetime is excluded.
- The first run stopped before fitting because calibration/hospital patients
  overlapped. Source metadata contained 196 patients seen on both sides of the
  hospital partition (37,119 total patients). Any patient with a held-out-hospital
  encounter is now excluded from all development roles, even at another hospital.
  All methods were rerun with this rule. Older audits using the old split must not
  be treated as clean patient-independent hospital comparisons without rerunning.

## Remaining limits

- Seven seeds here are evaluation resamples, not seven independently trained models.
- The temporal check is late-in-stay, not calendar-time-heldout.
- MIMIC has been repeatedly inspected and is transfer validation, not a new lockbox.
- Prior training exposure of the frozen shared encoder was not re-audited.
- Historical treatment features inherited from transition files still have incomplete
  availability provenance. Bedside/POE categories remain coarse; common names do not
  guarantee equal clinical semantics or capture across databases.
- Full runtime promotion additionally requires comparison with the validated release
  router and an untouched external cohort. No clinical or causal claim is permitted.

The next unresolved question is whether richer, availability-valid patient context
can identify conditional benefit beyond these coarse features. Changing the decision
objective alone did not do so here. Do not tune the MIMIC threshold to hide this result.

## Reproduction and artifacts

```sh
python -m unittest discover -s tests -q
python -m scripts.audit_expected_error_routing --output-dir artifacts/expected_error_routing_reproduction
python -m scripts.validate_release
```

Use the project's installed dependency environment and existing local raw data paths,
or pass `--eicu`, `--mimic`, `--eicu-root`, and `--mimic-root` explicitly.
The output directory must be new. This is a local-data research audit, not a patient
data upload workflow.

Completed outputs: `artifacts/expected_error_routing_20260905_v2/`:
`development.json`, `candidate_heads.joblib`, `report.json`, `FINDINGS.md`.
The development decisions were persisted before MIMIC outcomes were evaluated.
