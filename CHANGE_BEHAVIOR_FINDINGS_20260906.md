# Change-Behavior Audit: What the Pooled Score Hides

## Scope and Status

Completed the frozen release replay across 39 continuous eICU cells (six horizons)
and seven continuous NWICU 3h cells, each with 500 distinct patients and one
hash-selected eligible anchor per patient. These are 23,000 patient-cell cases,
NOT 23,000 distinct patients. No fitting, calibration, source selection or promotion.

The two hypoxemia interval-event cells were not evaluated: a single future SpO2
measurement cannot substitute for an event-window label. NWICU urine lacks valid
measured pairs. Other NWICU horizons were unavailable in the selected strict input.
No MIMIC claim is added: the recorded exclusion cache is missing and the located
transition files lack the per-row strict future quality/time fields required here.

Both evaluated databases were previously inspected; eICU training overlap is not
excluded. This is a descriptive behavior replay, NOT new independent validation.

## Main Finding

The model can reduce errors on retrospectively changing cases while adding error
on stable cases. Overall coverage can also hide poor coverage in changing cases.
This supports separate reporting, not outcome-based routing or relaxed promotion.

| Database / cell | Stable MAE difference | Changed MAE difference | Overall coverage | Stable coverage | Changed coverage |
|---|---:|---:|---:|---:|---:|
| eICU glucose@6h | +5.96 | -13.60 | 89.6% | 99.6% | 80.1% |
| eICU MAP@6h | +0.98 | -3.33 | 84.8% | 100.0% | 69.7% |
| eICU urine_output@3h | +4.25 | -33.04 | 82.5% | 98.5% | 66.3% |
| NWICU glucose@3h | +6.28 | -7.23 | 88.2% | 100.0% | 75.0% |
| NWICU MAP@3h | +0.47 | -2.54 | 89.8% | 100.0% | 71.8% |

MAE difference is model minus persistence, in each target's physical units. MAE
uses paired rows with a prediction. Coverage uses issued intervals only. Urine
3h issued predictions for 401/500 cases; its missing cases are not treated as
correct predictions. Full availability, uncertainty and patient bootstrap results
are in the machine-readable report.

For NWICU glucose@3h, changing-case MAE difference has nominal 95% patient bootstrap
CI [-10.39, -4.03] mg/dL; stable-case difference is [+4.67, +7.98]. Overall difference
is -0.10 with CI [-1.90, +1.85], so this replay does NOT show an overall improvement.

## Detecting Change Is Still Weak

At the predeclared research thresholds (glucose 20 mg/dL, MAP 10 mmHg):

- NWICU glucose@3h: stable false-move rate 13.6%; signed change recall 25.0%.
- NWICU MAP@3h: stable false-move rate 4.7%; signed change recall 11.6%.
- eICU glucose@6h: stable false-move rate 13.9%; signed change recall 23.0%.

Signed change recall requires the forecast to move at least the research threshold
in the correct direction. A smaller correct-direction forecast does not count as
detection, even if it improves MAE. Abstention counts as missed detection. These
are NOT sensitivities for a clinical deterioration diagnosis.

Persistence-only cells correctly have zero false moves but also zero change recall.
This is serving behavior, not evidence that every rejected expert is incapable of
predicting changes. This audit does not force those rejected heads into runtime.

## Interpreting the Intervals

Overall 90% calibration does not imply 90% in every subgroup, especially subgroups
defined by the outcome. In this replay NWICU glucose@3h's changing-case coverage
is 75.0%, with nominal bootstrap CI [69.1%, 80.5%]; MAP@3h is 71.8% [65.2%, 78.5%].
Stable coverage near 100% is not by itself proof of useless or incorrect intervals.
It shows why nominal marginal coverage cannot be sold as protection for every
future-change subgroup.

Half/double-threshold sensitivity retains the difference. NWICU glucose changing
coverage is 82.4% / 60.8%; MAP is 84.0% / 30.5%. Large realized changes naturally
stress tail coverage. These are descriptive sensitivities, not tuned definitions.

## What Was Implemented

- `whole_body_jepa/change_evaluation.py`: patient-equal paired bootstrap, stable
  false moves, signed/magnitude errors, change recall, availability, stratified
  coverage, width, central interval score, and prediction-width-based selection.
- `scripts/audit_change_behavior.py`: source-bound release replay, strict label
  eligibility, all numeric catalog statuses, care-unit slices, sensitivity and hashes.
- `tests/test_change_evaluation.py`: zero/missing distinction, missing label fields,
  NaT and empty future columns, denominators, patient equalization, invalid intervals,
  no input mutation, and numeric-cell parity with the actual serving API.
- Full test suite: 212 passed. No release artifact or promotion registry was changed.

## Next Decision, Not a New Promise

Do not use the realized future group to decide whether to use persistence. That
would create an oracle and leakage. Any prospective selector must infer risk from
pre-anchor information, and its false moves, missed changes and conditional
coverage must be tested on patients not used to choose it. Outcome-conditioned
results alone do not prove those groups are prospectively distinguishable.

The diagnosis is therefore not simply "the model is good and only the selector is
wrong". We observe a stable/change error tradeoff, low large-change recall, and
insufficient conditional uncertainty in this replay. Previous studies of selectors
remain relevant; changing the scoreboard has not fixed those modeling problems.

## Reproduction and Files

```bash
python -m unittest discover -s tests
python -m scripts.audit_change_behavior --max-patients 500 --bootstrap 1000 \
  --output artifacts/change_behavior_20260906_complete
```

Use a new output directory when repeating. Source data/model paths are CLI options.
Completed aggregate results: [table](artifacts/change_behavior_20260906_complete/FINDINGS.md)
and [full report](artifacts/change_behavior_20260906_complete/report.json).
The earlier `change_behavior_20260906` directory is an incomplete failed attempt
(`complete: false`), not the result. Its empty NWICU urine dtype failure was fixed
and the complete run reran every cell without changing thresholds or weights.

No row-level patient data are written. No causal, clinical, or next-measurement-time
claim is introduced. Research thresholds are not disease diagnosis/alert cutoffs.
