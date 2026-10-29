# Codex 凍結前審查第四輪：因果 time zero 第四版（2026-10-27，gpt-6.1-sol，唯讀、僅程式碼與彙總結果）

結論：修完列出的小問題後可以凍結。處理狀況見 `CAUSAL_TIME_ZERO_REDESIGN_20261027.md` 的「第四輪回應」。

---

**Verdict: freeze after the minor fixes listed below.** Version 4 resolves the substantive round-3 provenance, observation-support and estimand-rule findings. One reproducible failure path still prevents freezing the current implementation.

I reviewed all Markdown files, all six Python files, the spec and both aggregate JSONs. No files were modified or patient data accessed.

**1. Medium — Blocking implementation defect: an entirely unavailable laboratory item can still abort the one-shot estimate.**

[`with_provenance()`](./mimic_crystalloid_tte_extract.py#L174) creates columns only for laboratory names present in selected rows. Consequently:

- Selecting chloride but no bicarbonate produces no `day1_bicarbonate` column.
- Selecting no historical chloride produces no `nc_prior_chloride` column.

[`estimate()`](./causal_tte_analysis.py#L282) accesses these columns while constructing arguments or selecting the negative-control subset, **before entering `effect()`’s exception handler**.

Both cases were reproduced through the estimator with filesystem operations emulated in memory:

| Extractor-shaped input | Observed result |
|---|---|
| No `nc_prior_chloride` column | `KeyError: 'nc_prior_chloride'` |
| No `day1_bicarbonate` column | `KeyError: 'day1_bicarbonate'` |

In both cases, the estimate claim had been consumed and **no report was written**. The supplied all-NaN endpoint test preserves the column, so it misses this extraction-to-analysis failure.

**Required before freezing:** make laboratory extraction return the requested, fixed nullable schema even when an item has no selected rows. Add regressions covering absent primary, descriptive and negative-control items, including an entirely empty laboratory selection. An unavailable primary or control must produce a failed readiness report; unavailable descriptive bicarbonate must remain a structured secondary failure.

The remaining round-3 changes have these statuses. Severity identifies the issue assessed; resolved rows are not current blockers.

| Required change | Status / severity | Assessment |
|---|---|---|
| Approval/execution provenance | **Resolved — high** | `LEDGER` is anchored to the project, and `_ledger()` uses the protocol digest. `claim_step()` and `record_step()` use exclusive creation; completion requires a claim. `step_record()` verifies claim, step and protocol identity. `review()` verifies the estimate receipt’s exact inputs/output and the result’s own protocol digest before considering approval. See `osler_jepa/frozen_protocol.py:26,54–97` and `causal_tte_analysis.py:330–346`. |
| Observation support | **Resolved for the demonstrated failure — high** | `observation_support()` checks each arm within covariate strata: individual values for columns with ≤10 unique values, otherwise pooled quintile bins. One-hot encoding preserves category-specific checks. Strata with ≥50 arm-specific stays require ≥5% observed. The gate applies to primary and negative-control final diagnostics, excluding descriptive/sensitivity analyses. The previous three-value counterexample now returns support zero. See `causal_tte_analysis.py:77–78,208–221,250–253,309`. |
| Executable hypothetical-estimand rule | **Resolved — high** | Extraction creates strict-before-24-hour death/discharge flags; `change()` masks outcomes when either flag is present. An 18-hour sample followed by death at 20 hours becomes unobserved. Synthetic checks confirmed masking before 24 hours and retention at/after the stated boundary. See `mimic_crystalloid_tte_extract.py:309–313` and `causal_tte_analysis.py:282–285`. |
| Constant-SMD numerics | **Resolved — medium** | The scale-relative `1e-9` tolerance repairs the reproduced constant-105 failure. Identical constants under unequal weights return zero; separated constants zero versus one still return infinity. See `causal_tte_analysis.py:114–123`. |
| Failure handling/bootstrap gate | **Partially resolved — medium** | For present columns, unavailable analyses return structured failures, including zero valid bootstrap draws. Injected primary/control failures wrote failed reports; an injected descriptive failure preserved readiness. Bootstrap readiness uses only primary and control. The missing-column failure above remains. See `causal_tte_analysis.py:224–247,305–313`. |
| Laboratory provenance | **Resolved for the requested auditability — medium** | Selected laboratory rows retain sample time, `labevent_id`, `storetime`, admission ID and units. `lab_audit()` reports window counts, nonnumeric/implausible values, units and selected admission matching. Explicit spec covariate lists prevent these provenance fields entering adjustment. See `mimic_crystalloid_tte_extract.py:174–217` and `causal_tte_analysis.py:84–96`. |
| Disclosure items | **Mostly resolved — low** | The survivor output is explicitly named a subset association; rate-unit counts and restriction history are present. Some contradictory wording remains, listed below. |

**The complete-case negative-control implementation matches its revised definition.** It restricts to stays with historical chloride before calling `effect()`, so propensity is refitted in that subset and within each bootstrap sample. Because the control is fully observed there, `outcome_weights()` skips observation weighting. Its support diagnostic consequently reports complete observation **within that subset**; it does not establish historical-test availability across the original cohort. See `causal_tte_analysis.py:166–174,177–186,292–294`. This is acceptable for the stated subset falsification analysis.

**2. Low — Correct the remaining labels before freezing.**

- `causal_tte_analysis.py:34–35` still describes survivors at **30 hours** and a “while-alive quantity.” The spec and executable subset use the **24-hour** death flag. Update that text to the implemented descriptive association.
- `mimic_crystalloid_tte_extract.py:54–56` says the categories contain only saline, LR and D5LR. The supplied audit contains **32 additional item IDs**, including other fluid formulations. Describe the retained comparator items accurately and call time zero the first qualifying **recorded LR/saline** administration. This requires wording clarification, not reopening the accepted comparator rule.
- Mark the historical statements about bolus-volume eligibility and imputation outside bootstrap as superseded (`CAUSAL_TIME_ZERO_REDESIGN_20261027.md:75,87`).
- Revise the method-validation wording at redesign line 102: passing these diagnostics supports compatibility with the frozen checks; it does not validate causal identification.

**Residual limitations requiring disclosure, rather than another redesign:**

- The support gate detects the previous counterexample, but checks marginal strata rather than every joint covariate configuration. Passing cannot prove observation positivity or exchangeability.
- The hypothetical outcome still depends on the explicitly stated independence and measurement-consistency assumptions.
- The negative control concerns historical-test complete cases and cannot bound primary-outcome confounding.
- Laboratory episode matching remains imperfect. For chloride, the audit reports 28,757 selected values, 16,737 matching admissions and 12,018 missing admission IDs, leaving **two different-admission selections** (`results/mimic_crystalloid_tte_design.json:44–53`). Missing admission IDs alone do not establish emergency-department provenance. The baseline audit is computed before the chloride eligibility restriction, explaining some other items’ larger selected counts.
- The rate audit supports **44 mL/min and two µL/kg/min comparator rows**. These are supplied aggregate observations; I did not independently verify the raw records.

All six Python files parsed. **All 18 supplied tests passed with filesystem operations emulated in memory**, alongside the additional checks described above. DuckDB is unavailable, so SQL execution and real-file integration remain unverified. The freeze recommendation changes because the previous substantive safeguards now have executable implementations and successful counterexample checks; the outstanding work is the fixed-schema failure repair and precise disclosure corrections.