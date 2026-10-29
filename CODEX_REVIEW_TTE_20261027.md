# Codex 凍結前審查：因果 time zero 重新設計（2026-10-27，gpt-6.1-sol，唯讀、僅程式碼與彙總結果）

處理狀況見 `CAUSAL_TIME_ZERO_REDESIGN_20261027.md` 的「Codex 審查回應」。Claude 以 MIMIC 原始表核實：#1 成立（MIMIC-IV inputevents 每個 orderid 只有一列，連續輸注的 amount 是開始後實際輸入量）；#9 的 Ringers（220953）在這兩個醫囑類別中沒有任何紀錄，只有 225158、225828（另有 D5LR 225827，不列入）。

---

**I would not freeze this design as is.** Moving assignment to an initiation event repairs important parts of the earlier design, but the implementation still uses post-start delivered volume to determine eligibility, time zero and baseline adjustment. Those are substantive identification defects.

I read every Markdown file, inspected the relevant code, tests and aggregate JSONs, and parsed all nine Python files successfully. No files were modified or patient data accessed. The two supplied `time_zero` tests passed when executed against extracted source in memory. Additional synthetic checks reproduced the failures below. Full integration testing was unavailable: `duckdb` is absent, and importing the analysis fails on the missing `osler_jepa.recipe_availability` dependency. Literature checks used primary publications and official documentation.

1. **High — Confirmed: eligibility and time zero still depend on treatment delivered after initiation.**

   In [mimic_crystalloid_tte_extract.py:83](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/mimic_crystalloid_tte_extract.py:83), `sum(amount)` is combined with `min(starttime)` by `orderid`. Lines 99–103 then select the first aggregate reaching 500 mL.

   This is not a demonstrated pre-start prescription for ≥500 mL. MIMIC documents input volumes as derived from administration rates and durations; `orderid` identifies an administration instance, while `linkorderid` connects instances across changes. Consequently, the eventual amount depends on continuation, interruption and rate changes after its start. [Official inputevents documentation](https://mimic.mit.edu/docs/iv/modules/icu/inputevents.html)

   **Reproduced counterexample:** saline starts at hour 1 and balanced fluid at hour 2. Changing only the saline administration’s eventual amount from 400 to 600 mL changes time zero from hour 2 to hour 1 and assignment from balanced to saline.

   Thus later discontinuation—and potentially deterioration, transfer or death—can determine inclusion and assignment retrospectively. The supplied invariance test only adds separate later orders; it does not alter the future portion of an already-started administration.

   **Before freezing:** use a validated, pre-start prescribed-volume field, if one genuinely represents the decision being emulated. Otherwise remove the eventual-volume threshold and define eligibility at the first qualifying administration start. Simply regrouping by `linkorderid` would not solve this. Alignment must hold in the data construction, not only the protocol description. [Hernán et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27237061/)

2. **High — Confirmed: “prior volume” includes fluid administered after time zero. Weight provenance is also unresolved.**

   [mimic_crystalloid_tte_extract.py:104](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/mimic_crystalloid_tte_extract.py:104) selects orders whose **start** precedes time zero, then sums their **entire** amounts.

   An infusion beginning before time zero and continuing afterward contributes future volume to `prior_icu_*_ml`. For example, an eventual 400 mL infusion spanning hours 1–5 contributes all 400 mL to a time-zero-at-hour-2 covariate. Subsequent treatment decisions can therefore enter the adjustment set.

   Separately, `first(patientweight)` at line 85 has no temporal ordering. Its selected row need not correspond to the earliest start, and renaming it `admissionweight` at line 177 does not establish that it was measured at admission or before the treatment decision.

   **Before freezing:** retain administration intervals and calculate volume delivered strictly before time zero, with an explicit bolus convention. Select weight deterministically from a documented baseline source. The affected-patient counts cannot be established from this packet.

3. **Medium — Confirmed: the cohort is not genuinely new-user, and the simultaneous-treatment exclusion is incomplete.**

   Lines 99–103 select the first **large** administration, permitting earlier smaller administrations of either comparator. The supplied test explicitly includes prior saline followed by balanced assignment.

   The mixed-start check also examines only large administrations. **Reproduced:** simultaneous balanced 500 mL and saline 100 mL is retained as balanced, with `mixed_start_excluded = 0`, despite the stated exclusion of both fluids starting at time zero.

   Unrecorded pre-ICU exposure further limits the new-user interpretation, as already disclosed in the notes. These limitations matter because baseline chloride can already reflect crystalloid treatment.

   **Before freezing:** either implement an observable initiation/washout definition or rename the study around the actual treatment decision being compared. Check concurrent comparator administration across all relevant starts and ongoing infusions, using a prespecified timestamp rule. The active-comparator new-user framework requires a defensible common initiation point. [Lund, Richardson and Stürmer, 2015](https://pmc.ncbi.nlm.nih.gov/articles/4778958/)

4. **High — Confirmed omission: missing-outcome weighting does not resolve death, structural missingness or the outcome estimand.**

   [causal_tte_analysis.py:120](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/causal_tte_analysis.py:120) models
   \(P(\text{finite outcome}\mid X,A)\), then divides treatment weights by that probability.

   This can recover a full-cohort mean under appropriate observation-exchangeability, positivity and model assumptions. It does **not** automatically identify that mean:

   - Death before the endpoint is not ordinary missing laboratory measurement; the subsequent physiological outcome may be undefined.
   - Discharge and clinically driven testing can depend on evolving physiology beyond baseline \(X,A\).
   - For bicarbonate change, missing baseline bicarbonate makes the change unobservable regardless of later testing. Because baseline missingness is itself in \(X\), this creates a structural observation-positivity problem for a full-cohort change estimand.

   The readiness checks assess only treatment-weight balance, overlap and ESS. They do not inspect observation probabilities, final combined weights, endpoint-specific ESS or observation-model adequacy. Only IPTW is truncated; division by the measurement probability can reintroduce extreme weights.

   **Before freezing:** specify how death, discharge, missing baseline bicarbonate and absent follow-up measurements enter each estimand. Prespecify observation-model terms, combined-weight diagnostics, failure rules and sensitivity analyses. Adding post-treatment predictors indiscriminately would require a justified longitudinal observation model. Weighting depends on identification assumptions and diagnostics, not merely fitting a logistic model. [Cole and Hernán, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2732954/)

5. **Medium — Confirmed: the negative control overlaps its adjustment variable and can become partly self-fulfilling.**

   [mimic_crystalloid_tte_spec.json:16](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/mimic_crystalloid_tte_spec.json:16) describes the negative control as outside the adjustment set. However, its window is `[-48,-12)` hours, while baseline chloride uses `[-24,0)`.

   If the last baseline chloride occurs at hour −18, the **same laboratory result** can become both `baseline_chloride` and `nc_prior_chloride`. The propensity model then balances a variable also used as the purported falsification outcome. The synthetic negative-control test does not reproduce this: its prior value contains separate random variation.

   Temporal precedence does establish that assignment cannot cause the earlier sample. It does not establish that this control shares the relevant unmeasured-confounding structure, or that conditioning on later chloride preserves its sensitivity. Negative controls require those additional arguments. [Lipsitch, Tchetgen Tchetgen and Cohen, 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC3053408/)

   **Before freezing:** prevent shared specimens between the control and adjustment variables, retain their timestamps/identifiers, and justify the expected bias pathways. Prespecify handling of unavailable historical measurements.

   Requiring a **95% CI entirely within ±1** is substantially better than “CI contains zero” and is conservative relative to conventional 5% TOST using a 90% CI. However, ±1 has no supplied justification and exceeds the benchmark’s smallest accepted primary effect, 0.3 mmol/L. A pass cannot establish that primary-outcome confounding is below 0.3 mmol/L.

6. **Medium — Confirmed: the numerical RCT benchmark is not justified for this estimand.**

   The gate in `causal_tte_analysis.py:152–156` requires an estimate within `[-8,-0.3]` mmol/L and a negative upper confidence limit.

   The evidence supports an expected chloride direction under relevant fluid strategies, but does not establish these numerical bounds for this initiation contrast:

   | Evidence | Relevance and limitation |
   |---|---|
   | [SMART](https://pubmed.ncbi.nlm.nih.gov/29485925/) | Compared fluid-use policies, rather than only the identity of one qualifying initiation. |
   | [BaSICS](https://pmc.ncbi.nlm.nih.gov/articles/PMC8356144/) | Compared continued assigned-fluid use and found lower chloride with balanced solution. |
   | [PLUS](https://www.nejm.org/doi/full/10.1056/NEJMoa2114464) | Compared Plasma-Lyte 148 with saline for ICU fluid therapy and found higher chloride with saline. |
   | [Xue 2019](https://link.springer.com/article/10.1186/s13613-019-0506-y) | Reported a chloride **standardized** mean difference of −1.23; this is dimensionless. The biochemical synthesis involved three studies, not all 19,301 participants. |

   Neither a standardized effect nor results from sustained fluid policies directly supplies a raw-unit acceptance interval for unrestricted subsequent treatment after an initial administration.

   **Before freezing:** provide an explicit transport argument covering population, formulation, dose, subsequent exposure and endpoint timing. Otherwise label this an investigator-chosen directional benchmark, not RCT effect reproduction. A precise null could reflect the actual initiation estimand; a large negative association could reflect residual confounding. Neither is a reliable binary test of methodological validity.

7. **Medium — Confirmed calendar-variable defect; adjustment adequacy remains unestablished.**

   [mimic_crystalloid_tte_spec.json:8](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/mimic_crystalloid_tte_spec.json:8) contains useful physiological and admission variables. It lacks an explicit causal justification for the set, including how it addresses fluid indication, pre-ICU treatment, operative context and contraindications.

   One concrete problem is using `anchor_year_group` directly as admission-era adjustment. This describes the patient’s **anchor year**. Later admissions require the offset between the admission year and `anchor_year`; the older extractor calculates such an offset at `mimic_crystalloid_iv_extract.py:57–59`. [Official patients documentation](https://mimic.mit.edu/docs/iv/modules/hosp/patients.html)

   That matters particularly because the supplied historical diagnostics show substantial changes in fluid use and laboratory distributions over calendar time. Balancing anchor-year categories does not necessarily balance treatment era.

   **Before freezing:** derive an admission-specific approximate calendar variable with its uncertainty acknowledged. Specify a causal diagram and baseline operational definitions for important common causes. Examine plausible nonlinearities, interactions and laboratory recency. The maximum reported SMD of 0.063 establishes balance on the reported columns; it cannot establish exchangeability.

8. **Medium — The outcome-blind unit restriction is defensible, but its RCT justification contains a factual error.**

   [CAUSAL_TIME_ZERO_REDESIGN_20261027.md:23](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/CAUSAL_TIME_ZERO_REDESIGN_20261027.md:23) clearly discloses that the exclusions followed design diagnostics. Assuming the stated absence of outcome access is accurate, this is legitimate design development. It changes the target population and must remain disclosed.

   However, the statement that **PLUS and BaSICS both excluded brain injury is incorrect**. PLUS excluded traumatic brain injury or risk of cerebral edema. BaSICS enrolled patients with traumatic brain injury and reported a treatment interaction in that subgroup. [PLUS eligibility](https://www.nejm.org/doi/abs/10.1056/NEJMoa2114464), [BaSICS results](https://pmc.ncbi.nlm.nih.gov/articles/PMC8356144/)

   Also, excluding neuro care units does not exclude all brain injury, and “balanced solutions are hypotonic” is too broad across formulations.

   **Before freezing:** correct those statements, distinguish unit restrictions from diagnosis-based contraindications, and freeze the exact included population. Document the alternative restrictions examined. There is no demonstrated need for a multiplicity penalty merely because outcome-blind balance diagnostics informed eligibility.

9. **Medium — Fluid classification and administration units need a version-specific audit before freezing.**

   The configured balanced set at `mimic_crystalloid_tte_extract.py:47–48` includes item 220953, described as **Ringers**. Ordinary unbuffered Ringer’s contains approximately 155–156 mmol/L chloride, so that label does not justify classifying it as a low-chloride buffered comparator. [Ringer’s prescribing information](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=01721071-af03-4894-95c1-10b8bc64b888)

   **Unverified:** whether this item occurs in the relevant MIMIC-IV 3.1 records, and whether its local meaning matches ordinary Ringer’s. No dictionary or item/category frequency table is supplied, so numerical contamination is not established.

   Other implementation risks are explicit:

   - `min(itemid)` silently chooses an arm if a grouped order contains multiple comparator types.
   - `lower(amountuom) = 'ml'` drops other units rather than converting or auditing them.
   - Category and main-component filters are sensible, but their completeness for the intended treatment definition has not been demonstrated.

   **Before freezing:** preserve a version-specific dictionary and aggregate audit of included/excluded items, categories, units and mixed groups. Reject ambiguous fluid groups. For reference, official MIMIC extraction code labels 225827 **D5LR**, so it should not simply be added as ordinary LR. [Official crystalloid extraction code](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts_postgres/fluid_balance/crystalloid_bolus.sql)

10. **Medium — The laboratory windows are mostly correct, but availability, ties and episode attribution are underspecified.**

    [mimic_crystalloid_tte_extract.py:114](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/mimic_crystalloid_tte_extract.py:114) implements baseline windows with a strict upper bound: `[t0−24h,t0)` for labs and `[t0−6h,t0)` for vitals. The outcome window is correctly inclusive at 18 and 30 hours, selecting the value nearest 24 hours.

    Matching labs by subject and time is reasonable: MIMIC explicitly warns that `hadm_id` can be missing for relevant laboratories. However, the current helpers discard specimen identifiers, selected timestamps and admission provenance, preventing an audit of overlapping windows or measurements belonging to another nearby episode. [Official labevents documentation](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)

    `arg_min` and `arg_max` also lack secondary tie-breaking rules. Equidistant outcomes or conflicting values at one timestamp therefore have no declared deterministic selection policy. Laboratory units and plausible-value rules are absent.

    Finally, `charttime < t0` establishes a pre-treatment sample, not necessarily a result available before treatment: MIMIC distinguishes sample timing from availability at `storetime`. This is not automatically post-treatment physiology, but the protocol must distinguish **pre-treatment sampling** from **decision-time availability**, especially for eligibility.

    The running-treatment predicate `starttime < t0 AND endtime >= t0` reasonably represents an already-running support state; it does not itself require future treatment duration.

11. **Medium — Confirmed readiness bug: complete separation can be reported as perfect balance.**

    [causal_tte_analysis.py:77](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/causal_tte_analysis.py:77) returns SMD zero whenever pooled within-arm variance is zero.

    **Reproduced:** a binary covariate equal to zero in every control and one in every treated patient returns `0.0`. That is complete imbalance. Only equal constants should return zero; unequal constants require an explicit failure.

    Additional readiness issues are:

    - SMDs are rounded before comparison with 0.10.
    - Effects and confidence limits are rounded before benchmark/equivalence decisions.
    - The overlap rule permits a small unsupported tail, while clipping treatment weights does not establish positivity there.
    - Final IPTW×observation-weight diagnostics are absent.

    **Before freezing:** gate on full-precision, finite quantities; fail on degenerate unequal distributions; and report support by arm and important strata. These are implementation safeguards. I have not established that the supplied 0.063 maximum is numerically wrong. Distributional diagnostics should also extend beyond means. [Austin and Stuart, 2015](https://pmc.ncbi.nlm.nih.gov/articles/pmid/26238958/)

12. **High — Confirmed: the proposed freeze and one-shot restrictions are not enforced by the current execution path.**

    [external_value_conditional_interval_audit.py:328](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte/external_value_conditional_interval_audit.py:328) checks whichever hashes a protocol happens to list. **Reproduced:** an empty `frozen_code_sha256` mapping is accepted.

    `causal_tte_analysis.py:143–165` does not require that the manifest binds the actual spec, input tables, estimator dependencies or invocation. Outcome extraction at `mimic_crystalloid_tte_extract.py:195–209` requires no protocol and can overwrite its output. Repeated analysis is likewise permitted.

    Any truthy `external_review` value can enable `causal_claim_allowed`; there is no structured approval decision tied to the reviewed result. A nonempty record describing rejection would still be truthy.

    The absence of a completed TTE freeze is expected at this stage. The defect is that the existing mechanism cannot enforce the planned freeze.

    **Before freezing:** require the relevant manifest entries, bind the actual invocation and spec, verify before outcome extraction, and record extraction/analysis execution with overwrite protection. External approval must identify its decision and reviewed artifact. Hashes demonstrate integrity, not one-time execution.

13. **Low — Bootstrap resampling is substantially correct; preprocessing uncertainty, denominators and failure handling need clarification.**

    `causal_tte_analysis.py:123–136` calls both probability fits inside the resampled statistic. `osler_jepa/instrumental_variable.py:89–98` resamples whole subjects and preserves repeated draws.

    **Confirmed dynamically:** five bootstrap draws plus the original estimate produced six treatment-model fits and six observation-model fits. This addresses the brief’s main bootstrap concern.

    Remaining issues:

    - Median imputation is performed before resampling, so its estimation uncertainty is held fixed.
    - Nonfinite draws are silently discarded; the valid count is not included in the TTE report. Fitting exceptions instead abort execution.
    - `estimate()` merges tables without asserting unique, nonmissing keys or `validate="one_to_one"`. Duplicate outcome keys can multiply stays.
    - Subject clustering changes uncertainty, not the estimand’s weighting: the estimator averages **eligible stays**, not equally weighted unique patients.

    The untruncated treatment weights correspond to an ATE approach, not ATT. Observation weighting can target the eligible cohort under its assumptions; it does not inherently restrict the estimand to measured patients. Truncation introduces a bias–variance tradeoff and should have prespecified sensitivity analyses.

    One primary chloride analysis with descriptive bicarbonate reporting does not automatically require Bonferroni correction. Freeze that inferential hierarchy and any sensitivity families before outcome access.

The supplied aggregate counts, 74 adjustment columns and reported maximum weighted SMD are internally consistent. They cannot validate the underlying extraction. Before extracting outcomes, repair the administration-based timing defects, settle the endpoint and observation assumptions, revise the negative control and benchmark, and bind the execution to a complete protocol. The new evidence supporting a refusal to freeze is the demonstrated dependence of assignment and baseline adjustment on post-start administration—not simply residual uncertainty about observational causal inference.