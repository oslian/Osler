# Forecast-aware routing repair: results and failure decomposition

## Status

Completed; **0/5 primary candidates pass the full gate**. No runtime promotion,
checkpoint replacement, interval activation or causal claim. The existing release
remains 27 movable cells and 14 persistence fallbacks.

One concrete design omission was repaired: the old selector read only the first
160 shared/context-belief dimensions plus 59 coarse care-context features. It did
not directly receive the expert's proposed delta, quantiles or explicit current
target. This is not proof that its latent contained no such information; rather,
the selector had to reconstruct information already available from the expert.

We added eight explicit forecast-evidence features and tested both direct mean
benefit and cost-sensitive choice. The latter weights each expert/persistence win
label by its absolute error difference; it does not mistake win frequency for
expected error reduction. Its score is not a calibrated medical-event probability.

This uses held-out stacking, following the established pattern described in the
[scikit-learn stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html).
Existing HGB estimators were sufficient; no dependency or parallel body model was added.

## Matched evaluation

The five 6h experts and shared encoder are exactly the previous run's frozen
weights. Patient roles, pre-anchor care features, calibration data and external
sampling are unchanged. The script checks release hashes, patient disjointness,
and exact reference MAE reproduction (absolute tolerance 1e-8).

Each target selects between two real-evidence objectives using eICU selection
patients only. Separate eICU patients calibrate intervals. Those choices and weights
are saved in `development.json` and `selector_heads.joblib` before MIMIC evaluation.
Each MIMIC evaluation contains 12,000 patients, one anchor per patient. The five
target cohorts overlap and must not be summed into 60,000 distinct patients.

Controls: previous primary selector; always-persistence; always-expert; same-objective
blind selector; same-capacity shuffled added evidence; no-care-context selector.
The added evidence is jointly shuffled across patients, not merely filled with zeros.

## MIMIC primary results

MAE is in native transition units; lower is better. "Previous" means the previous
expected-error experiment's eICU-selected primary, not the runtime release router.

| Target @6h | Persistence | Previous selector | Repaired selector | Always expert | Primary objective |
|---|---:|---:|---:|---:|---|
| Creatinine | 0.18556 | 0.19797 | 0.19516 | 0.20270 | Mean benefit |
| BUN | 3.02925 | 3.05742 | 3.06585 | 3.09671 | Mean benefit |
| Urine output | 111.91949 | 104.69012 | 103.74997 | 102.94314 | Cost-sensitive |
| Potassium | 0.42890 | 0.38493 | 0.38381 | 0.38448 | Cost-sensitive |
| Bicarbonate | 1.91184 | 1.88293 | 1.87777 | 1.88599 | Mean benefit |

Four primary mean errors improve vs the preceding experiment. That does not imply
four promotions or four complete cross-database gains.

- Creatinine passes MIMIC comparisons vs the previous selector, always-expert and
  shuffled evidence, including seven evaluation resamples, unit-balanced and late-stay
  checks. But it remains significantly worse than persistence: delta +0.00960,
  95% patient-bootstrap CI [0.00727, 0.01182]. Source hospital superiority is also
  incomplete. The repair helps selection without solving the task.
- BUN improves eICU mean MAE but not MIMIC. Its MIMIC delta vs persistence is
  +0.03660, CI [0.01622, 0.05773]. Additional inputs are not uniformly helpful.
- Urine improves vs persistence and the previous mean error, but is significantly
  worse than always-expert: delta +0.80684, CI [0.47416, 1.12227]. It does not beat
  the matched cost-sensitive blind or shuffled controls.
- Potassium has lower point MAE than always-expert, but the difference is not
  significant: delta -0.000666, CI [-0.001376, 0.000045]. Source and external
  control comparisons are not uniformly stable. Primary benefit AUC rises from
  the old 0.478 to 0.564, but ranking improvement is not sufficient for promotion.
- Bicarbonate beats always-expert in the pooled MIMIC comparison (delta -0.00822,
  CI [-0.01194, -0.00420]), but fails unit-balanced/late-stay checks. It also fails
  the MIMIC CRPS comparison. A pooled improvement is not a complete win.

The unselected potassium mean-benefit variant has MAE 0.38105 in MIMIC, but it was
not the primary chosen on eICU. We do not retrospectively switch to it after seeing
external outcomes. A future experiment needs source-only selection and new validation.

## What the remaining failure actually looks like

### Stable-state harm outweighs dynamic-state benefit in kidney targets

"Stable/dynamic" below is a retrospective outcome partition, not a healthy/diseased
diagnosis and not an inference-time input. Thresholds are unchanged: absolute delta
0.3 for creatinine, 5 for BUN, 50 for urine, 0.3 for potassium, and 2 for bicarbonate.

| Target | eICU patient-test dynamic fraction | MIMIC dynamic fraction | MIMIC stable MAE: persistence / repaired | MIMIC dynamic MAE: persistence / repaired |
|---|---:|---:|---:|---:|
| Creatinine | 30.92% | 15.69% | 0.09113 / 0.10946 | 0.69295 / 0.65561 |
| BUN | 35.66% | 18.76% | 1.64242 / 1.77069 | 9.03554 / 8.67513 |
| Urine output | 57.62% | 59.70% | 16.62736 / 24.55035 | 176.24567 / 157.21304 |
| Potassium | 49.75% | 49.40% | 0.14741 / 0.19989 | 0.71722 / 0.57221 |
| Bicarbonate | 51.79% | 49.28% | 0.64219 / 0.79518 | 3.21886 / 2.99221 |

All five candidates reduce point MAE within the retrospectively dynamic group but
increase it within the stable group. The net result depends on both proportions and
the size of those harms/gains. These conditional means have not independently passed
all subgroup gates and do not authorize future-label-based routing.

### Population proportions alone do not explain the failure

As a post-hoc diagnostic only, keep the eICU stable/dynamic error differences fixed
and reweight them to MIMIC's observed dynamic proportion:

`reweighted_delta = (1 - mimic_dynamic_fraction) * eicu_stable_delta + mimic_dynamic_fraction * eicu_dynamic_delta`

| Target | eICU observed delta vs persistence | eICU delta at MIMIC mixture | Actual MIMIC delta |
|---|---:|---:|---:|
| Creatinine | -0.01988 | -0.00884 | +0.00960 |
| BUN | -0.25117 | -0.13164 | +0.03660 |

Both would still beat persistence if only the two-group proportions changed. The
sign reversal therefore also involves changed errors *within* these broad groups.
It is not justified to say "just add more stable patients" will solve transfer.
This descriptive decomposition is not causal and does not identify which clinical,
measurement, selection or representation difference produced the shift.

### Wrong direction and excessive amplitude both remain

For the fixed creatinine expert, wrong-direction proposals occur in 37.0% of eICU
patient-test anchors and 33.6% of MIMIC anchors. Same-sign/zero-change overshoot
(`not wrong direction` and `abs(proposal) > 2*abs(actual_delta)`) increases from
8.7% to 29.5%. This includes proposing movement when the measured delta is zero;
it is not proof that biology was perfectly constant.

Thus the MIMIC creatinine failure is not simply a higher wrong-direction rate.
Excessive movement on small/zero observed changes is a concrete failure mode. The
new gate reduces some errors but still chooses the expert on about 55.3% of MIMIC
creatinine anchors and 72.9% of BUN anchors. Its binary choice cannot correct a
useful-direction proposal's magnitude except by rejecting the entire proposal.

## Intervals

Primary patient-equal coverage in MIMIC: creatinine 96.98%, BUN 95.44%, urine 94.93%,
potassium 93.38%, bicarbonate 95.77%. All exceed the unchanged 93% upper gate.
No external recalibration or selective activation was performed. Overcoverage is
not an undercoverage safety failure, but does fail the requested tight calibration
contract. CRPS uses the unscaled empirical sample distributions; coverage uses the
separately scaled conformal interval.

## Boundaries and next unresolved work

The input/decision-interface omission is fixed in the candidate implementation.
The complete cross-database prediction problem is not fixed. Remaining work should
target conditional amplitude/error calibration and patient-state comparability,
not another unrestricted increase in latent size. Any amplitude correction must
be fit on eICU-only held-out expert predictions and compared against both fixed
experts, the new selector and a matched placebo before activation.

Raw-data existence does not establish usable or identical semantics. The MIMIC
transition file lacks per-label quality/actual-delta metadata, while the eICU file
has it. The current MIMIC extractor uses nearest measurements within its configured
window; eICU strict labels for these five targets have 4-8h actual deltas around the
nominal 6h. This check did not establish that timestamp policy alone explains the
failure, nor that either source provides perfectly equivalent future supervision.

Seven seeds are patient evaluation resamples, not seven refits. Late-stay is not
calendar-time-heldout. Prior frozen-encoder training exposure remains unaudited.
MIMIC is repeatedly inspected transfer validation, not a pristine lockbox. Small
care units have noisy estimates, and a unit-balanced gate is not per-unit validity.
Neither calibrated clinical-event risk nor treatment causality is established.

## Reproduction

```sh
python -m unittest discover -s tests -q
python -m scripts.audit_forecast_aware_routing --output-dir artifacts/forecast_aware_routing_reproduction
python -m scripts.validate_release
```

185 tests passed. Completed outputs are in
`artifacts/forecast_aware_routing_20260905/`: `development.json`, `selector_heads.joblib`,
and `report.json`. The heads depend on the previous experiment's frozen expert bundle
and the hash-bound release, not on a newly trained standalone body model. Reports
are aggregate-only; no patient-row data was added to the deliverables.
