# Tolerance Interval Audit and Model-Side Requests

## Scope

This report audits the current release artifacts against the interval-related
requests in `MODEL_SIDE_REQUESTS.md`. It validates stored patient, care-unit,
time and cross-database evidence plus runtime behavior. It does not refit an
interval or reuse raw external outcomes to choose a model.

## Tolerance interval test result

The final runtime authority is `unified_source_selector.json`, not the older
`runtime_registry.json` alone. Resolving the selector, release calibrator and
event exclusions produces exactly the 25 interval cells declared by the model
manifest.

| Check | Result |
|---|---:|
| Manifest interval cells | 25 |
| Resolved runtime interval cells | 25 |
| Patient/domain/time empirical coverage gate | 25/25 pass |
| Cells with stored mean width | 24/25 |
| Cells with stored CRPS vs persistence distribution | 0/25 |
| Cells with target-value applicability range | 0/25 |
| Runtime API separates nominal and empirical coverage | No |

Across the aggregate empirical coverage values used by this audit, coverage
ranges from 0.873 to 0.914 with a mean of 0.897. The nominal target is 0.90 and
the accepted range is 0.87-0.93. This means the existing intervals pass their
current calibration contract. It does not mean every care unit separately has
90% coverage; the existing contract evaluates pooled and care-unit-equal
coverage over supported units.

The narrowest stored mean widths are potassium@24h (1.73 mmol/L),
potassium@12h (2.05 mmol/L), hemoglobin@12h (4.46 g/dL), o2sat@12h
(8.28 percentage points) and sodium@24h (12 mmol/L). The widest are urine
output@1h (350.76 mL/h), urine output@3h (233.01 mL/h) and glucose@3-12h
(approximately 133-140 mg/dL). Passing coverage therefore does not imply that
an interval is narrow or operationally informative.

`bicarbonate@24h` uses a patient-conditional normalized interval, so its width
changes per prediction. The selector artifact does not store its population
mean width; this is the one missing width summary.

## Integrated response to MODEL_SIDE_REQUESTS.md

| Request | Current status | Evidence or remaining work |
|---|---|---|
| Preserve `derived_interval_rate` | Completed | It is now a canonical quality class and survives parquet round trips. |
| Convert `NaT` end times safely | Completed | Missing end times become `None`; regression tests cover the round trip. |
| Reject stale target anchors | Completed | Continuous targets beyond the freshness contract abstain instead of silently persisting. |
| Expose head abstention | Completed | Runtime reports `abstained`, `patient_persistence_anchor` and `continuous_head_abstained`. |
| Explicit label policy for release targets | Completed | All current 53 variables have explicit policies; unknown policies fail closed. |
| Reconcile registry/report metrics | Partly completed | The selector is now the final runtime authority and artifacts are checksum-bound. Historical registry and report values still use different cohorts/estimands and must not be compared without their provenance. |
| Add calibrated distributions | Partly completed | 25 runtime cells now have eligible 90% intervals and all pass the stored empirical coverage gates. The stricter distribution release is incomplete because CRPS versus persistence, applicability ranges and API-level empirical coverage are absent. |
| Dynamic hurdle/quantile modeling | Tested, not promoted | The state-conditioned kidney audit implemented `P(dynamic)`, signed delta quantiles, persistence as the stable expert and CRPS scoring. Dynamic patients often improved, but false dynamic classifications harmed stable patients and no target passed all controls. |

## Interpretation

The old statement that release 20260822 emitted no intervals is no longer true
for the current release. The current release has 25 coverage-qualified runtime
interval cells. It is still too optimistic to call this a complete
distributional model: coverage is only one property. A complete claim also
requires interval sharpness, a proper score against an empirical persistence
distribution, explicit deployment applicability and API-visible empirical
coverage.

The state-conditioned experiment supports the document's central hypothesis:
persistence is a legitimate expert for stable physiology, while a dynamic head
can help changing patients. It also identifies the unresolved failure mode:
the current shared state does not classify stable versus dynamic patients with
enough cross-database specificity. Consequently, it remains candidate-only and
does not alter the validated release.

## Required next interval work

1. Store patient-equal empirical coverage, nominal coverage and calibration
   population separately in each runtime response.
2. Add per-cell applicability ranges and fail closed outside them.
3. Evaluate the adaptive distribution with CRPS against an empirical
   persistence residual distribution on disjoint patients.
4. Store mean and median patient-conditional width, including
   bicarbonate@24h.
5. Report coverage and width separately for stable and dynamic physiology;
   never shrink intervals merely to make them look precise.

All conclusions are factual research findings. Causal and clinical promotion
remain disallowed.
