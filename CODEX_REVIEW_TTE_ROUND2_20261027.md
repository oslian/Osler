# Codex 凍結前審查第二輪：因果 time zero 第二版（2026-10-27，gpt-6.1-sol，唯讀、僅程式碼與彙總結果）

處理狀況見 `CAUSAL_TIME_ZERO_REDESIGN_20261027.md` 的「第二輪回應」。Claude 以 MIMIC 原始表核實 #1（bolus）：157,101 筆 bolus 全部是 1 分鐘、FinishedRunning，amount 與 originalamount、totalamount 幾乎都相同，登打時間第 90 百分位落在開始後 92 分鐘，所以無法證明量是事先決定的；已改為不論量多少，任何 bolus 都算。速率：只有 46 筆不是 mL/hour，已全部換算，沒有空值。

---

**Verdict: do not freeze.** Version 2 fixes several first-round defects, but the bolus eligibility rule remains unsupported as start-time information, the death-related estimand remains undefined, and the new execution safeguards still permit mismatched inputs and repeated runs.

I read all five Markdown files and inspected every supplied Python file, the spec and both aggregate JSONs. No files were modified or patient data accessed. Raw-data facts in the brief are treated as supplied evidence, not independently verified observations.

The first-round findings have the following status. Severity describes the issue being assessed; resolved entries are not current blockers.

| # | Status / severity | Assessment |
|---|---|---|
| **1** | **Partially resolved — high** | The infusion branch correctly uses the row’s starting rate, given the supplied rate semantics. The bolus branch still uses `amount >= 250`, without a separately validated prescribed-volume field or evidence establishing that this value was fixed before administration. See [mimic_crystalloid_tte_extract.py:109](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_extract.py:109), lines 109–118. Describing bolus volume as “ordered” does not establish that provenance. |
| **2** | **Resolved for the original defects — medium** | Entire delivered volumes are no longer summed into baseline covariates. Weight now comes deterministically from the selected initiation row rather than an unordered aggregate. See [mimic_crystalloid_tte_extract.py:144](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_extract.py:144), lines 144–154. This establishes the weight’s record source, although its actual measurement/availability time remains unverified. The new low-rate indicators have a separate classification defect described below. |
| **3** | **Partially resolved — medium** | The simultaneous-start exclusion now checks **all retained comparator starts**, including nonqualifying low-rate starts, rather than only qualifying therapy. That repair is correct. However, “new user” means first qualifying **recorded ICU therapy**, with earlier nonqualifying administrations permitted; it is not crystalloid-naïve initiation. The bolus uncertainty in finding 1 also affects this definition. See [mimic_crystalloid_tte_extract.py:143](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_extract.py:143), lines 143–156. |
| **4** | **Partially resolved — high** | Per-outcome baseline restriction, combined-weight truncation, endpoint-specific ESS, measurement-probability reporting and death/discharge summaries are implemented. See [causal_tte_analysis.py:137](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/causal_tte_analysis.py:137), lines 137–170 and 193–212. But “had everyone been measured” does not define chloride after death, and ESS alone does not diagnose observation positivity or final-weight balance. |
| **5** | **Partially resolved — medium** | The negative-control window is now `[-48,-24)`, disjoint from baseline `[-24,0)`, removing the demonstrated same-time-result overlap. The code correctly requires the control CI to lie within ±0.5. See [mimic_crystalloid_tte_extract.py:245](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_extract.py:245) and analysis lines 206–212. The new assertion that this certifies a ±0.5 bound on **primary-outcome bias** is unsupported. |
| **6** | **Resolved for the original benchmark defect — medium** | The unsupported RCT numerical range has been removed. The spec explicitly calls the benchmark investigator-chosen and distinguishes initiation from sustained-fluid policies. See [mimic_crystalloid_tte_spec.json:19](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_spec.json:19). With margin 0.5 and negative direction, the implementation correctly requires `CI upper < -0.5`. Passing remains a directional compatibility check, not validation of identification. |
| **7** | **Partially resolved — medium** | Admission-era adjustment now incorporates the admission-specific offset rather than using anchor-year group directly. See [mimic_crystalloid_tte_extract.py:96](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_extract.py:96), lines 96–99. The notes also supply a useful verbal adjustment rationale. Important unmeasured indications and pre-ICU exposure remain acknowledged assumptions. A missing diagram alone would not justify refusing freeze; the unresolved identification questions below do. |
| **8** | **Resolved — low** | The PLUS/BaSICS wording has been corrected in the supplied notes. Both rounds of unit exclusion and rejected alternatives are disclosed. See [CAUSAL_TIME_ZERO_REDESIGN_20261027.md:25](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/CAUSAL_TIME_ZERO_REDESIGN_20261027.md:25), lines 25–34. The CCU restriction is defensible as outcome-blind design development, conditional on the stated exposure history. |
| **9** | **Partially resolved — low** | Ringers is removed from the comparator mapping, and the supplied category audit supports its absence. Selecting a grouped order’s minimum item ID has also disappeared; the supplied one-row-per-order fact removes that particular aggregation concern. However, exceptional/missing rate units are silently classified as nontherapy, and the audit does not enumerate unit handling. See [mimic_crystalloid_tte_extract.py:104](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_extract.py:104), lines 104–133. |
| **10** | **Partially resolved — medium** | Baseline labs require both sampling before t0 and `storetime < t0`; the filter occurs before ranking. `QUALIFY` deterministically selects the latest eligible baseline result, while outcomes use nearest time, then earlier sample, then smaller ID. Plausibility filters are applied before selection. See [mimic_crystalloid_tte_extract.py:164](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_extract.py:164), lines 164–205. Selected timestamps, specimen/admission provenance and units are still discarded, limiting auditability. |
| **11** | **Resolved — medium** | Unequal constants now produce infinite SMD; equal constants produce zero. Design and effect gates operate before reporting-rounding. See [causal_tte_analysis.py:90](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/causal_tte_analysis.py:90), lines 90–132 and 205–225. I reproduced the supplied separation assertions. |
| **12** | **Partially resolved — high** | The canonical required-file list, digest verification, outcome sidecar check and explicit approval decision are genuine improvements. They do not yet bind the actual invocation, enforce single execution, or tie approval to the reviewed result. Details below. |
| **13** | **Partially resolved — low** | Both probability models remain inside the subject-cluster bootstrap; duplicate-key detection and one-to-one merging are added; valid replicate share is reported and gated. Imputation remains outside resampling. Fit exceptions still abort rather than become counted failed replicates, and unique keys can still contain one null. See [causal_tte_analysis.py:155](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/causal_tte_analysis.py:155), lines 155–170 and 188–190; `osler_jepa/instrumental_variable.py:86–102`. |

The following defects and unresolved requirements must be addressed before freezing.

1. **High — Establish start-time validity for the bolus branch.**

   The provided evidence supports using an infusion’s starting rate. It does not establish that the bolus `amount` field is a pre-start prescription rather than completed administration volume.

   Evaluating the extractor’s actual classification expression in an in-memory SQL harness, changing only an earlier saline bolus’s amount from **249 to 251 mL** changed assignment from later LR to earlier saline and moved t0 backward. This demonstrates threshold dependence; whether that dependence uses future information depends on the unresolved field semantics.

   **Required:** validate a genuinely pre-start bolus-volume source, or remove delivered amount from bolus eligibility. Add a regression check that changes post-start delivered quantities while preserving the initiation decision. The existing tests supply `fluid_therapy` as an already-computed Boolean, so they cannot detect this extraction-level problem.

2. **Medium — Correct what the “low-rate line” indicators actually represent.**

   The extraction Boolean is false for small boluses and unrecognized/missing infusion rates as well as known infusions below 50 mL/h. `time_zero()` then flags **any earlier retained comparator administration**, without checking rate, category or whether it remains running.

   Consequently, a **200 mL bolus is recorded as a prior low-rate saline line**. An infusion with an unsupported unit can also become “nontherapy” and fail to exclude an earlier substantive exposure. These are confirmed classification paths; their numerical impact in MIMIC is unknown.

   **Required:** distinguish known low-rate infusions, small boluses and unknown-rate administrations. Prespecify unknown-unit handling. Either implement an ongoing-line definition or accurately name these as prior-administration indicators. Preserve the rates/categories needed to audit the classification.

3. **High — Define an outcome estimand that handles death.**

   [causal_tte_analysis.py:14](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/causal_tte_analysis.py:14), lines 14–22, retains a full eligible-stay ATE under hypothetical measurement while explicitly leaving death unmodelled.

   Measuring everyone cannot produce a physiological result after death. Reporting death proportions improves transparency but does not resolve this definition. Likewise, a measurement model using baseline variables and arm does not automatically account for discharge or testing driven by evolving post-treatment physiology.

   **Required:** specify the endpoint and treatment-effect target when death precedes assessment, and state the corresponding identification assumptions. A composite or explicitly defined survivor quantity would answer a different, prespecified question; merely dropping deaths would not repair the current full-cohort ATE.

4. **High — Complete the protocol, execution and approval binding.**

   Three independently consequential gaps remain:

   **Actual spec and inputs are unbound.** Extraction reads an arbitrary `--spec`, verifies whatever filenames that object lists, then independently reads `--design`. Analysis similarly trusts the supplied spec object. See [mimic_crystalloid_tte_extract.py:271](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_extract.py:271), lines 271–278, and analysis lines 181–190, 231–235.

   An alternate spec can name the unchanged canonical files in `frozen_files` while changing the actual design path, margin or analysis configuration. Verification succeeds because the verifier never establishes that the object being used came from the frozen spec. Extraction’s sidecar contains no actual design/spec digest, so `estimate()` cannot detect outcomes extracted against different anchors with matching stay keys. I reproduced the verifier’s acceptance of an alternate unbound spec using mocked file reads/digests.

   **Write-once is only a non-atomic filename check.** [osler_jepa/frozen_protocol.py:38](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/osler_jepa/frozen_protocol.py:38), lines 38–41, neither reserves the path nor records consumption of a protocol step. Another output filename permits another extraction or estimate. Concurrent callers can both pass the existence check. Analysis writes no execution record.

   **Approval is neither result-bound nor compatible with post-result review.** `review_approved()` accepts an approved decision and nonempty reviewer without checking the reviewed artifact. Conversely, adding approval to the protocol after reviewing results changes the protocol hash and invalidates the extraction sidecar checked by `estimate()`. See [osler_jepa/frozen_protocol.py:44](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/osler_jepa/frozen_protocol.py:44) and analysis lines 183–185, 214–222.

   **Required:** bind the actual loaded spec and all actual inputs; record their digests in extraction provenance; atomically claim canonical execution steps independently of output filename; and keep post-result approval separate from the immutable protocol, referencing the exact protocol and result digests.

5. **Medium — Remove the negative-control “bias bound” claim.**

   [mimic_crystalloid_tte_spec.json:18](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/mimic_crystalloid_tte_spec.json:18) says a passing control certifies a bias bound. The analysis docstring and redesign notes repeat this interpretation.

   A small adjusted association with earlier chloride does not bound confounding of later chloride change. Equal measurement units do not establish equal confounder effects.

   **Synthetic counterexample:** with true treatment effect zero, an unmeasured variable affecting treatment choice and future chloride change—but not historical chloride—produced:

   | Quantity | Estimate | 95% CI |
   |---|---:|---:|
   | Primary change | −2.728 | [−2.842, −2.618] |
   | Prior-chloride control | 0.037 | [−0.011, 0.078] |

   Running the actual estimator with only filesystem/protocol I/O mocked passed **every readiness check** and returned `ready_for_external_review`; causal permission correctly remained false without approval. This does not show that MIMIC has this confounding structure. It disproves the claimed general bias guarantee.

   **Required:** retain ±0.5 as an explicitly chosen falsification/equivalence margin and the primary threshold as a directional benchmark, without calling either a certified primary-bias bound or methodological validation.

6. **Medium — Diagnose the final weighted analysis, not only IPTW design balance and ESS.**

   Combined weights are correctly multiplied before truncation, including within bootstrap replicates. However, [causal_tte_analysis.py:207](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte2/causal_tte_analysis.py:207), lines 207–212, reuses full-design IPTW balance and adds only combined-weight ESS.

   In an in-memory example with an entirely unobserved control stratum, treatment-weighted design SMD was approximately zero and final ESS remained **1,507 / 2,988**, yet final measured-sample weighted SMD was **1.406**. The current ESS criterion cannot detect that failure.

   **Required:** prespecify endpoint-specific observation-support and final-weight diagnostics, including balance against the intended target population and an explicit failure policy. Report the effect of truncation. Per-outcome baseline restriction is correctly implemented; diagnostics should follow each resulting analysis population.

Several smaller points should also be settled in the freeze:

- **Low:** make the documented tie policy match the SQL. The general extractor comment says latest/largest identifiers, while nearest-outcome ties use earlier/smaller identifiers. Both are deterministic; the inconsistency is documentation.
- **Medium:** retain selected laboratory timestamps, identifiers, units and admission provenance in an auditable extraction artifact. Plausibility bounds are implemented, but plausible numeric values cannot establish correct units or episode attribution. Report exclusions and missingness caused by these filters.
- **Low:** reject null keys explicitly and define bootstrap exception handling. Either refit imputation inside resampling or clearly describe the resulting preprocessing-conditional uncertainty.
- **Low:** preserve the complete outcome-blind restriction history, including CCU, trimming and calendar alternatives. The second exclusion itself does **not** require rejection or an automatic multiplicity penalty; it changes the target population and must remain frozen after outcome access.

The supplied aggregate arithmetic is consistent: the cohort flow yields 34,802 initiators before baseline-chloride restriction, and the design report contains 82 SMD columns consistent with the reported rounded maximum. Those checks do not validate extraction semantics.

All six Python files parsed successfully. The two supplied `time_zero` tests and separation assertions passed; additional checks used memory-only synthetic data. DuckDB is unavailable, so I did not validate DuckDB SQL execution or run the file-writing integration suite.

**The refusal to freeze now rests on specific remaining defects: unestablished bolus start-time validity, an undefined death-related estimand, incomplete execution binding, and overstated negative-control guarantees. These require substantive changes rather than minor cleanup.**