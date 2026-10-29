# Codex 凍結前審查第三輪：因果 time zero 第三版（2026-10-27，gpt-6.1-sol，唯讀、僅程式碼與彙總結果）

處理狀況見 `CAUSAL_TIME_ZERO_REDESIGN_20261027.md` 的「第三輪回應」。

---

**Verdict: do not freeze.** Version 3 resolves the delivered-volume eligibility defect and several other round-2 findings. Remaining blockers concern execution provenance, observation support, the operational definition of the hypothetical estimand, and failure handling.

I read all five Markdown files, all supplied Python files, the spec and both aggregate JSONs. No files were modified or patient data accessed. Raw-data statements in the brief remain supplied evidence, not independently verified observations.

The round-2 changes have the following status. Severity describes the assessed issue; resolved entries are not current blockers.

| Required change | Status / severity | Assessment |
|---|---|---|
| Bolus start-time validity | **Resolved conditionally — high** | `crystalloid_starts()` no longer reads delivered amount; `time_zero()` qualifies every retained bolus. Given the supplied category/start-time semantics, later changes to delivered volume cannot change assignment. `mimic_crystalloid_tte_extract.py:115–120,149–151`. |
| Low-rate indicators | **Resolved — medium** | Only earlier, nonbolus infusions below 50 mL/h contribute. Unknown-rate infusions at or before t0 exclude the stay. Names and documentation accurately describe previously started infusions, without claiming they remain running. `mimic_crystalloid_tte_extract.py:152–164`. |
| Death estimand | **Partially resolved — high** | A hypothetical strategy and a strong independence assumption are now stated. The code still lacks a corresponding, unambiguous intercurrent-event and assessment rule. See finding 3. |
| Protocol / execution / approval binding | **Partially resolved — high** | Loading the actual spec from the protocol and comparing extraction input/output digests repair important gaps. Approval does not verify estimate provenance; execution identity remains filename-dependent. See finding 1. |
| Negative-control wording | **Resolved — medium** | Spec and analysis correctly describe a falsification test that cannot bound primary-outcome confounding. The counterexample is retained as a test. `mimic_crystalloid_tte_spec.json:19–23`; `test_causal_tte_analysis.py:89–93`. |
| Final-weight diagnostics | **Partially resolved — medium** | The implementation correctly compares each weighted measured arm with the **whole endpoint analysis population**, and checks between-arm balance and ESS. Mean balance does not provide the missing observation-support check; numerical constant handling also fails. See findings 2 and 4. |
| Smaller points | **Mixed — low / medium** | Tie documentation, null stay-key rejection, per-replicate imputation and restriction history are repaired. Laboratory provenance remains incomplete; bootstrap failure handling and the descriptive-endpoint gate need correction. Audit timestamps do not currently enter adjustment. |

The following findings require changes before freezing.

1. **High — Approval can authorize an unrecorded result from a different protocol.**

   [`review()`](</private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte3/causal_tte_analysis.py:281>) loads the frozen spec, reads its result path and checks the separate review record’s hashes. It never reads the **estimate execution record**, verifies completion of that step, or checks the result’s own `protocol_sha256`.

   **Reproduced with in-memory filesystem operations:** a result containing `protocol_sha256: "different_protocol"` and `readiness_status: "ready_for_external_review"` received `causal_claim_allowed: true`, despite having no estimate claim or completion record. The review record correctly matched the current protocol and result file hashes. Those hashes therefore bind an approved pairing, without establishing that the result came from that protocol’s execution.

   Other remaining gaps are in [`frozen_protocol.py:47–73`](</private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte3/osler_jepa/frozen_protocol.py:47>):

   - `claim_step()` is atomic **within one protocol pathname**. A byte-identical protocol copied to another filename gets a fresh step directory. I reproduced a second extraction claim with the same protocol digest.
   - `record_step()` neither requires a claim nor writes exclusively. I reproduced overwriting an existing extraction record.
   - `step_record()` merely parses JSON. The extraction checks at `causal_tte_analysis.py:225–229` correctly compare protocol, input and output digests, but do not validate claim/completion state or the record’s step identity.

   **Required:** define one canonical execution identity and ledger independent of protocol filename; require claimed steps and exclusive completion records; validate the extraction receipt’s identity; and make `review()` verify the completed estimate receipt, its exact inputs/output, and the result’s protocol identity before considering approval. This need not become a tamper-proof security system, but the documented execution guarantees must hold through the supported functions.

2. **High — Structural observation nonpositivity can pass every readiness gate.**

   [`final_weight_diagnostics()`](</private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte3/causal_tte_analysis.py:177>) checks covariate means. `min_measurement_probability` is reported at line 204 but never gated. Neither check establishes observation support.

   **Reproduced through the actual estimator, with filesystem/protocol I/O mocked and all statistical calculations intact:**

   - Treatment was balanced independently of numeric covariate \(x\in\{-1,0,1\}\).
   - Both treatments had the same full-data change, \(-3x^2\): the true treatment effect was zero.
   - All treated outcomes were measured; controls were measured only at \(x=0\).
   - Baseline covariates varied, and the negative control had no treatment association.

   | Output | Result |
   |---|---:|
   | Estimated primary effect | −2.000 |
   | 95% CI | [−2.0419, −1.9561] |
   | Final mean-balance diagnostics | Passed |
   | All readiness checks | Passed |
   | Status | `ready_for_external_review` |

   Symmetry makes the weighted means match while two control strata have **zero** observation support. The linear observation model reports positive fitted probabilities despite this failure.

   **Required:** prespecify endpoint-specific observation-support diagnostics and failure rules across relevant covariate strata, including suitable nonlinear/distributional checks. State observation positivity explicitly in the identification assumptions. No diagnostic can prove positivity, but the frozen policy should detect this demonstrated failure.

   This counterexample establishes a readiness-policy limitation; it does not establish that MIMIC contains this pattern.

3. **High — The hypothetical estimand is named, but its executable death/discharge rule remains unresolved.**

   The statement in [`causal_tte_analysis.py:20–31`](</private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte3/causal_tte_analysis.py:20>) is a substantive improvement. A hypothetical strategy is not inherently invalid.

   However, the protocol says events “before the outcome window” would not occur, selects measurements from 18–30 h, and reports events through 30 h. These define different possible censoring boundaries. The estimator uses only `np.isfinite(outcome)` as its observation indicator (`causal_tte_analysis.py:197–198`); death and discharge flags do not affect the primary analysis.

   For example, a sample at 18 h followed by death at 20 h remains an observed “~24 h” outcome. The code does not specify whether that measurement represents the outcome under the proposed no-death scenario, or whether the scenario instead requires survival through assessment or 30 h.

   **Required:** freeze the assessment-time rule, the exact intercurrent-event boundary, and the consistency assumptions connecting observed measurements to the hypothetical outcome. Make extraction and observation weighting implement that rule. A single observation model may be defensible under strong assumptions, but its observed-case indicator must correspond to the stated estimand.

4. **Medium — Identical constant covariates can produce large, false SMDs.**

   [`weighted_smd()`](</private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte3/causal_tte_analysis.py:101>) handles mathematically exact zero variance, but floating-point weighted means can create tiny artificial variances and differences.

   **Reproduced:** a covariate equal to **105 for every stay**, using the estimator’s combined weights, produced between-arm SMD **2.0** and arm-versus-population SMD **1.414**. Thus a perfectly identical covariate can fail readiness.

   **Required:** handle constant inputs and numerical degeneracy robustly while preserving failure for genuinely unequal constants. Extend the regression test beyond uniform weights and constants equal to one.

5. **Medium — Optional analyses can abort the one-shot run, and bicarbonate incorrectly participates in the bootstrap gate.**

   Per-replicate imputation is correctly inside [`weighted_effect()`](</private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte3/causal_tte_analysis.py:163>), and the listed fitting exceptions become invalid draws. I observed different imputation medians across resamples.

   Remaining failures:

   - With no valid bootstrap draws, `cluster_bootstrap()` calls `np.quantile([])` and raises `IndexError` (`osler_jepa/instrumental_variable.py:94–102`).
   - An empty survivor subset or wholly missing endpoint fails in `effect()` before the protected bootstrap statistic. Both failures were reproduced.
   - Secondary and sensitivity analyses execute before the report is written, after the estimate step has been consumed.
   - `bootstrap_valid` uses **every** entry in `results`, including descriptive bicarbonate (`causal_tte_analysis.py:262`). A 90%-valid bicarbonate bootstrap vetoes readiness even when primary and negative-control bootstraps are fully valid, contradicting the stated inferential hierarchy.

   **Required:** report structured failure for unavailable secondary/sensitivity analyses; produce a failed primary report when inference is unavailable; handle zero valid draws explicitly; and restrict bootstrap readiness to the prespecified gated analyses.

6. **Medium — Laboratory auditability remains only partially repaired.**

   [`last_before()` and `nearest()`](</private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_tte3/mimic_crystalloid_tte_extract.py:180>) now retain selected sample times. They still discard the selected laboratory identifiers, result availability time, units and admission/specimen provenance. Subject-based matching makes the episode provenance particularly useful. Counts removed by plausibility filters are also absent.

   **Required before freezing the extraction contract:** retain sufficient selected-row provenance and define the unit/episode checks and filter audit counts. Plausible numeric ranges alone do not validate units.

The following items need accurate disclosure or labeling, rather than a new identification claim.

- **Medium — Survivor sensitivity:** `causal_tte_analysis.py:251–254` conditions on observed post-treatment survival and refits propensity within that subset. It is not automatically a causal while-alive effect or an effect among patients who would survive under either treatment. Moreover, `fu_died_before_30h == 0` establishes absence of a recorded qualifying death, not independently confirmed survival after discharge. Label this as an observed-survivor subset association unless a separate causal estimand is defined. It can remain a descriptive sensitivity analysis.
- **Low — Audit timestamps:** no adjustment leak was found. `adjustment_frame()` uses explicit spec lists, which contain no `*_time` fields. Baseline audit times pass the design-table check; `day1_*_time`, `nc_*_time` and `fu_*` columns are rejected by the outcome prefixes. I reproduced these behaviors.
- **Low — Any bolus and unknown rates:** these implement the revised definition correctly. Unknown-rate exclusion is limited to infusions at or before t0; later unknown-rate rows do not change inclusion. The broad bolus definition and recorded-ICU initiation population should remain disclosed. Raw rate-unit/conversion counts would make the supplied conversion claim auditable.
- **Low — Documentation:** the redesign notes still say imputation occurs outside bootstrap, list 12 rather than 13 tests, and retain an earlier bolus-volume description (`CAUSAL_TIME_ZERO_REDESIGN_20261027.md:70,82,96,105`). Correct these inconsistencies.

The aggregate cohort arithmetic is consistent: \(38{,}788-272-3{,}488=35{,}028\), followed by 28,757 eligible stays with baseline chloride. The reported 82 adjustment columns and rounded maximum SMD are internally consistent. These checks do not validate extraction or identification.

All six Python files parsed. Eleven analysis tests passed with filesystem operations emulated in memory; both `time_zero` tests passed against the extracted function. DuckDB is unavailable, so this was **not** a DuckDB or real-file integration test.

The bolus evidence changes my judgment on that previous blocker: its delivered-volume dependence is repaired. The current refusal to freeze rests on the specific remaining failures above, especially the reproduced approval-provenance gap and structural-observation-support counterexample.