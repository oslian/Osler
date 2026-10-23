# Codex 二次審核（2026-10-22，gpt-6.1-sol，唯讀、僅程式碼與彙總結果）

Claude 核對狀態（2026-10-22）：
- 已在程式碼確認：#1（錨點作為輸入特徵進入系統 token，且依未來標籤有無而不同）、#2（normalize_cohort 先依未來標籤過濾，再算 lag／slope）、#3（標籤取 t+h 前後 min(2,h/2) 小時內最近量測，但線上更新在 t+h 就使用；換住院即解出所有待定結果）、#16（teacher_anchored_joint_jepa_audit.py 已偏離 v8 雜湊）。
- 已確認為設計限制或過度宣稱：#4、#5、#7、#8、#9、#10、#11、#12、#13、#14、#15。
- #6 已動態確認：organ 階段 `organ_to_system`、`system_feedback_gates` 梯度為 0，body 階段又被凍結，永遠停在初始值。

修正進度（2026-10-22，第 1 步「預測時點不變性」）：
- 資料契約：所有萃取器保留每個錨點（不論有無未來標籤），並記錄標籤量測時間差 `*_offset_hr`；全部資料重建中（`rebuild_full_timeline_data_20261022.sh`）。
- 歷史特徵：`add_history_features` 在完整串流上、任何依標籤過濾之前計算 lag／slope／onset（#2）。
- 部署時鐘：`osler_jepa/deployment_clock.py`，各院住院以 Poisson 到達、彼此重疊；結果在 anchor + h + |量測偏移| 才可用於線上更新；換住院不再解出待定結果（#3）。
- 暖機：前 min(200, 25%) 個到達的住院只提供第一天權重，且只用第一個計分住院到達前收到的輸入；暖機住院不計分（#4）。
- 共用 `Replay` / `network_replay`（`site_adaptive_interval_dev.py`）取代 `stream_order` / `network_order`；外部稽核、點校正／DtACI／θ0／網路共享的選擇腳本、診斷腳本全部改用，並只計分非暖機列。
- JEPA 錨點：每列都有專家預測，與有無未來標籤無關；錨點 recipe 的 MIMIC 訓練樣本排除 JEPA 隊列的 MIMIC 病人（#1、#7 的錨點重疊部分）。
- JEPA 部署比較：同一部署時鐘、標籤可得時間、暖機排除。
- #6：organ 層對目標的訊息改為先經 `organ_to_system` 進入系統 token，organ 殘差讀取「organ 層對系統 token 的改變」（與 body 層對稱）；新增各階段介面都收到點預測梯度的測試。
- 測試：`test_label_timing_contract`、`test_forecast_time_invariance`、`test_deployment_clock`、`test_deployment_replay_integration`、`test_expert_anchor_invariance`、`test_jepa_deployment_form`、`test_adjacent_hierarchy` 等全數通過。
- 影響：既有點校正、DtACI、θ0、網路共享的選擇結果，以及所有 JEPA 結果，都是在有洩漏的重播下得到的；資料重建完成後須重跑選擇與評估（第 2 步）。

---

The results support continued development, but **the supplied evidence does not establish the full JEPA takeover gate, deployment-causal online evaluation, or valid treatment-effect identification**. Several problems are implementation defects, beyond the selection bias already acknowledged in the notes.

I read every Markdown file, inspected the relevant code and aggregate JSONs, parsed all 35 Python files successfully, and ran small synthetic checks against extracted source functions. No files were modified or patient data accessed. Full training and integration tests were unavailable: important dependencies, prediction exports, split manifests and tests are absent; a PyTorch check also failed during import under the read-only environment.

1. **High — Confirmed: JEPA expert anchors depend on future-label availability.**

   `build_expert_anchor_columns.py:42–48` initializes every anchor to the current value, then computes expert predictions **only for rows with a future label**. `:63–64` writes those predictions back. Thus identical deployment-time inputs receive different anchors depending on whether a future measurement exists.

   A synthetic check, with the fitting function stubbed to return `99`, produced anchor `99` when a future label was present and `1` when it was absent, with current value `1` in both cases.

   This also affects other targets: expert anchors enter system tokens and are pooled into organs in `osler_jepa/adjacent_hierarchy.py:461–488`. An unlabeled target can therefore expose its future observation availability to a different, evaluated target through the hierarchy.

   **Fix:** compute all expert anchors from forecast-time inputs, independently of future columns. Apply label masks only to losses and scoring. Verify that deleting or changing every `future_*` column leaves anchors and predictions unchanged.

2. **High — Confirmed: history features are constructed after selecting rows by future outcomes.**

   `mimic_to_eicu_49_cell_external_audit.py:114–120` filters by future-label quality and finite future values. `_feature_frame` subsequently constructs lag and slope features from the surviving rows (`target_specific_domain_conformal_audit.py:114–138`). The anchor builder repeats future-label filtering before calling that pipeline (`build_expert_anchor_columns.py:48–62`).

   Synthetic example: current values at hours `0, 6, 12` were `0, 10, 20`. Removing the hour-6 row because its future label was missing changed the hour-12 lag from `10` to `0`.

   These are retrospective histories unavailable to a deployment that cannot know which forecasts will eventually acquire labels. This is distinct from merely scoring observed outcomes.

   **Fix:** construct histories on the complete causal anchor stream, then select labeled rows for fitting or evaluation. Preserve the true admission/onset origin rather than deriving it from label-selected rows.

3. **High — Confirmed: online updates can consume labels before their measurements occur.**

   `site_adaptive_external_audit.py:174–179` supplies `resolve_hours = anchor_hours + horizon`; `jepa_deployment_form_dev.py:64,77` does the same. However, labels may come from measurements **after** that time: the protocol explicitly permits a tolerance of `min(2, h/2)` hours, implemented, for example, in `emory_challenge2019_transition_extract.py:129–140`.

   Consequently, a 6-hour forecast labeled by an hour-8 measurement can update the next hour-6 prediction. A synthetic check confirmed that changing this unavailable label changed the next corrected prediction.

   Additionally, both trackers resolve every outstanding label on a stay change, regardless of its timestamp (`osler_jepa/site_adaptive_point.py:72–78`; `osler_jepa/site_adaptive_conformal.py:231–237`). That is coherent only under an artificial deployment where each stay finishes before the next begins. It does not establish correctness for overlapping hospital or network streams.

   **Fix:** retain actual label-selection and result-availability timestamps, including any delay needed to finalize a nearest-measurement label. Replay forecasts and results on one event clock; a stay change must not resolve pending outcomes.

4. **High — Confirmed: “day-one” weighting looks ahead within the evaluated warm-up stream.**

   `site_adaptive_external_audit.py:169–173` fits weights using **all rows** from the first selected stays, then applies those weights from the beginning of those same stays. The network path repeats this at `:207–210`.

   Later measurements from the first admission—and inputs from later admissions—therefore influence its first interval. These are unlabeled covariates, but they are still future information for the claimed online deployment.

   **Fix:** either collect a separate, completed warm-up cohort before evaluation or update density-ratio estimates using only inputs already received. Score cold-start performance separately.

5. **High — Confirmed: the JEPA gate does not implement the mandatory coupling or takeover requirements.**

   `jepa_7seed_gate.py:47–58` checks a selected path against `system_only`. It does not require a full path to beat both no-organ-coupling and no-body-feedback alternatives. A full model can therefore pass because one component helps while another enabled component hurts.

   The ablations are inference-time switches on one trained adapter (`teacher_anchored_joint_jepa_audit.py:3862–3920`), rather than independently trained, matched ablations. In the four-database evaluation, the same pooled ablation result is reused for every database; this is explicit in `jepa_7seed_gate_rule_four_db_20261021.json:20`.

   The gate also lacks direct checks against persistence, the prescribed five training samples, and the required care-unit/time validation. These requirements appear in `AGENTS.md:82–88`. Missing validation may justify withholding a conclusion; it does not constitute a pass.

   **Fix:** retain the 13 cells as exploratory candidates. Require separately trained ablations, evaluated per database and per cell under the same deployment corrections, and explicit evidence for every enabled path. The claim of passing the “complete AGENTS.md gate” in `JEPA_EXPERT_ANCHORED_DEPLOYMENT_FORM_FINDINGS_20261021.md:53` is unsupported.

6. **Medium — Confirmed by static dependency tracing: part of the organ-stage interface receives no point-loss training.**

   `set_training_stage("organ")` enables `organ_to_system` and `system_feedback_gates` (`osler_jepa/adjacent_hierarchy.py:325–344`). But the organ-stage point prediction uses `target_organ_context` directly (`:653–670`), not the computed `organ_system` (`:573–593`). Body feedback is disabled in this stage, and distributional point-stage loss is disabled.

   These interface parameters are then frozen during body training (`:345–355`), although the body pathway uses them (`:613–630`). They consequently remain at their initial values under the supplied training sequence.

   **Fix:** ensure each intended interface has a loss-connected training stage. Add a focused gradient/update check for those named modules. This finding was not dynamically verified because PyTorch import failed.

7. **Medium — Confirmed calibration reuse; additional anchor-training overlap remains unverified.**

   Calibration labels perform several model-selection jobs before being reused for interval calibration:

   - `_fit_hurdle` selects among 30 candidate policies using calibration errors, then returns those same selected calibration predictions (`target_specific_domain_conformal_audit.py:1034–1095`).
   - JEPA coupling paths are selected on calibration labels (`teacher_anchored_joint_jepa_audit.py:1310–1367`), and predictions from the selected paths on those same patients become downstream calibration data (`:3985–4002`).
   - Residual scales are fitted on the same calibration residuals subsequently converted into conformal scores (`osler_jepa/interval_scale.py:67–75`; `jepa_deployment_form_dev.py:51–55`).

   These procedures do not retain ordinary split-conformal validity simply because the rows were excluded from gradient training.

   Separately, expert anchors are fitted on independently sampled MIMIC patients before JEPA splitting (`build_expert_anchor_columns.py:46–62`). No supplied manifest establishes that expert-fitting patients exclude JEPA calibration patients. **Actual overlap cannot be determined from this packet.**

   **Fix:** separate fitting, policy/path selection and final calibration at patient level, or use a justified cross-fitting procedure. Audit patient intersections across the entire expert-plus-JEPA pipeline.

8. **Medium — Confirmed: the adaptive quantile targets rows rather than the patient-equal evaluation estimand.**

   `weighted_quantile` receives density-ratio weights without inverse patient-frequency weighting (`osler_jepa/site_adaptive_conformal.py:46–57`; `site_adaptive_external_audit.py:154–173`). JEPA explicitly supplies uniform row weights (`jepa_deployment_form_dev.py:74–75`).

   Synthetic counterexample: one patient contributes 100 zero residuals and another contributes one residual of 10. The implemented 90% quantile is zero, giving **50% patient-equal coverage**. The project’s patient-weighted helper returns 10.

   Moreover, the test-point weight is replaced by the mean calibration weight, and an unattainable quantile returns the largest finite score. This differs from the test-covariate-specific weight and infinity mass used in weighted conformal prediction. [Tibshirani et al., 2019](https://arxiv.org/abs/1904.06019)

   **Fix:** align calibration and adaptation with the declared patient-level estimand. Describe the current tracker as an empirical variant until its weighted, delayed, hierarchical guarantees are established.

9. **High — Confirmed: the crystalloid target trial has misaligned eligibility, treatment assignment and outcome timing.**

   ICU admission is declared time zero, but eligibility requires receiving at least 1,000 mL and satisfying fluid-composition criteria over the following 24 hours (`eicu_crystalloid_iv_extract.py:5–17,108–119`). “Baseline” labs may occur one hour after treatment begins (`:43`). Outcomes can occur at hour 18 while treatment classification still includes fluids through hour 24 (`:44,84`).

   This conditions admission-time inclusion on later treatment and potentially survival, permits post-treatment adjustment, and can classify treatment using fluids given after the measured outcome. The MIMIC design has analogous timing (`mimic_crystalloid_iv_extract.py:41–45,63–80`).

   These are identification problems, independent of precision. Target-trial methods require eligibility, assignment and follow-up to be aligned. [Hernán and Robins, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4832051/)

   **Fix:** redesign the estimand around an eligible treatment decision time. A first-fluid initiation comparison or a properly specified longitudinal strategy would require different eligibility and analysis—not another estimator on the current table.

10. **High — Confirmed: the IV diagnostics cannot establish validity or attribute failure solely to low precision.**

   The reported first-stage F is a homoskedastic individual-level statistic (`osler_jepa/instrumental_variable.py:51–64`), although the instrument is hospital preference. Hospital clustering of the outcome bootstrap does not make that first-stage diagnostic cluster robust.

   The placebo and negative-control gates pass whenever a confidence interval includes zero (`crystalloid_iv_estimate.py:95–96`). The supplied placebo chloride interval is `[-2.777, 15.826]`, and the platelet interval is `[-83.626, 8.744]` (`results/crystalloid_iv_estimate_20261020.json:28–48`). These are inconclusive tests, not evidence of negligible effects. Platelet change also needs justification as an outcome unaffected by the fluid strategy.

   Hospital preference can represent other hospital practices affecting outcomes. Balance on measured numeric variables does not establish exclusion or exchangeability. [Brookhart and Schneeweiss](https://pmc.ncbi.nlm.nih.gov/articles/PMC2719903/)

   **Fix:** use cluster-robust weak-instrument diagnostics and weak-IV-robust inference; justify exclusion, monotonicity and the negative control explicitly. Replace “validity passed; only precision failed” (`CAUSAL_CRYSTALLOID_IV_FINDINGS_20261020.md:31`) with **“diagnostics were insufficient to establish identification.”** Suitable strength diagnostics include [Montiel Olea and Pflueger, 2013](https://www.tandfonline.com/doi/abs/10.1080/00401706.2013.806694).

11. **Medium — Confirmed: the proposed FACTT route answers a different treatment question.**

   `CAUSAL_CRYSTALLOID_IV_FINDINGS_20261020.md:38` proposes FACTT as the randomized-data route. FACTT randomized **conservative versus liberal fluid management**, not balanced crystalloid versus saline. Its randomized assignment cannot directly identify the current composition effect. [Official BioLINCC FACTT catalog](https://biolincc.nhlbi.nih.gov/catalog/factt/)

   **Fix:** explicitly change the estimand if using FACTT, or obtain randomized evidence matching fluid composition. Randomization also does not eliminate precision limitations for arbitrary secondary endpoints.

12. **Medium — Confirmed: the reported leave-out comparisons are not outer validation of the complete selected system.**

   The θ₀ LODO analysis loads recipes already selected using all three development databases (`site_adaptive_dtaci_prior_dev.py:63,81`), then leaves a database out only when selecting θ₀ (`:139–155`). Therefore `93 versus 85` evaluates that component conditional on a previously selected pipeline.

   Network widths are generated using all hospitals before splitting their scored results into halves (`site_adaptive_network_dev.py:128–143`). Both halves influence shared adaptive state; recipes were also selected using the same development database.

   Sharing earlier outcomes is legitimate for the intended network product. It nevertheless means these halves are not independent deployments, and the comparison cannot establish untouched-network transfer.

   **Fix:** distinguish component-development comparisons from complete-pipeline validation. For an outer evaluation, repeat upstream selection within each training fold and evaluate separately initialized held-out networks.

13. **Medium — Confirmed: Emory is previously exposed external evidence, with an incompatible glucose denominator.**

   Prior researcher exposure to Emory forecasting results is disclosed (`EMORY_LOCKBOX_FINDINGS_20261020.md:7`; protocol `:10`). A researcher decision cannot make that historical exposure disappear. Conversely, this packet does **not** establish direct Emory-label fitting of the current expert recipes.

   The glucose issue is stronger: v8 declares mixed-source glucose unevaluable (`adaptive_model_lockbox_protocol_v8_20261020.json:28`), but the extractor still maps `Glucose` to chemistry `glucose` (`emory_challenge2019_transition_extract.py:70`). Counts alone cannot establish that measurements are predominantly POC.

   Removing the six glucose cells from the supplied original result changes the headline from **40/41 to 34/35**, still including the persistence-fallback cell. The later bicarbonate extension is additional evidence from an already inspected cohort, with a changed measurement definition; its result artifact is absent here.

   **Fix:** describe Emory as a previously exposed external evaluation; separate incompatible targets and post-unblinding extensions. Preserve untouched-site evaluation for the final frozen system.

14. **Medium — Confirmed: coverage gates certify averages, not individual hospitals, units or patients.**

   `external_value_conditional_interval_audit.py:158–184` gates only patient-equal and hospital-/unit-equal **mean** coverage. Two groups with coverage 0.80 and 1.00 have mean 0.90 and can pass. The gate has no interval-utility threshold and no uncertainty requirement around estimated coverage.

   Requiring observed coverage below 0.93 also makes conservative intervals fail a safety gate, especially in small samples. That may be an efficiency criterion, but it should be distinguished from undercoverage.

   **Fix:** report group-specific uncertainty, worst-group behavior and clinically interpretable widths; use a proper interval score for efficiency. Do not translate mean coverage into a claim that intervals work for every hospital or patient.

15. **Medium — Confirmed: seed/sample stability does not resolve multiplicity or adaptive-stream dependence.**

   Point gates use unadjusted per-comparison bootstrap intervals (`target_specific_domain_conformal_audit.py:192–227`; `mimic_to_eicu_49_cell_external_audit.py:256–293`) across many cells, recipes and development iterations.

   Seven seeds share evaluation patients, and the five training samples need not be disjoint. Consequently, 7/7 or 4/5 passes measure particular forms of stability; they are not independent replications or multiplicity correction.

   Online corrections also couple later patients’ predictions to earlier patients’ outcomes. Bootstrapping fixed realized patient losses does not reproduce uncertainty in those adaptive histories.

   **Fix:** predeclare the confirmatory family, adjust cell-level inference appropriately, and retain effect sizes and absolute units. Evaluate state uncertainty through independent streams or suitable block/replay resampling. The phrases “clinically meaningful” in the JEPA note remain unsupported by a clinical utility endpoint.

16. **Medium — Confirmed: protocol verification checks a manifest, not the actual invocation.**

   `verify_protocol` hashes named files (`external_value_conditional_interval_audit.py:328–336`), but callers do not bind CLI arguments to that manifest. Different recipe/configuration paths, training templates, sample counts and cell subsets remain accepted (`site_adaptive_external_audit.py:228–244,339–358`). Extracted-data addenda and the one-shot execution rule are not enforced there.

   Among files supplied under their frozen names, `teacher_anchored_joint_jepa_audit.py` also differs from its v8 digest (`adaptive_model_lockbox_protocol_v8_20261020.json:119`). That would correctly block verification. Missing snapshot files and the older Emory v5 protocol prevent verification of historical execution; they do not prove that execution was invalid.

   **Fix:** bind actual arguments, loaded artifact digests, extraction manifests and execution records to an immutable run specification. Restore the exact snapshot or document a new freeze before accessing untouched results.

The best-established methods I would compare next are:

| Area | Method and source | Concrete application |
|---|---|---|
| Expert forecasts and intervals | [Conformalized quantile regression — Romano et al., 2019](https://arxiv.org/abs/1905.03222) | Fit asymmetric conditional quantiles on training patients, then calibrate on separate patients; compare width and interval score with quintile residual scaling. |
| Online adaptation | [Gibbs and Candès, 2022](https://arxiv.org/abs/2208.08401) | Benchmark the implemented log-scale mixture against the published procedure under identical delayed-feedback streams. Do not inherit guarantees without checking assumptions. |
| Treatment effects | [Target-trial emulation — Hernán and Robins, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4832051/), plus cluster-robust IV methods above | Repair time zero and treatment strategies first. A different estimator cannot repair an invalid instrument or post-baseline eligibility. |
| JEPA residual value | [GRU-D — Che et al., 2018](https://www.nature.com/articles/s41598-018-24271-9), alongside existing ridge/boosting tools | Use missingness/time-aware and simple residual predictors as bounded research comparators. Within the required hierarchy, compare trained versus random backbone features and independently trained, capacity-matched disabled-path models. |

I would prioritize work in this order:

1. **Repair forecast-time invariance:** anchors and histories must survive removal of future labels unchanged; online updates must respect actual availability; warm-up inputs must precede evaluation.
2. **Re-evaluate existing development data**, preserving previous results as historical artifacts. Measure how many passes survive these corrections before consuming another lockbox.
3. **Rebuild the JEPA gate** around per-database, independently trained ablations, disjoint selection/calibration patients and all mandatory validation requirements.
4. **Redesign the causal study before extracting further outcomes.** Keep the current failed result; neither another database nor narrower intervals would by itself establish identification.
5. **Freeze one executable protocol** with target definitions, arguments, hashes, multiplicity handling and endpoint denominators, then perform untouched-site evaluation.

The aggregate arithmetic checks out: I reproduced the supplied Emory 40/41 count and the stated 13-cell JEPA intersection. Those checks establish reporting consistency. The leakage paths and gate omissions above prevent treating those counts as validated capability.