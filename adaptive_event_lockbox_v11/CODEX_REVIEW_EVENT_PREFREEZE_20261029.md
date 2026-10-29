# Codex 凍結前審查：事件格線上校準與 lockbox 路徑（v11 凍結前，2026-10-29，gpt-6.1-sol，唯讀；送審資料夾不含任何病人層級資料）

## Claude 處理

**Codex 結論：** 修正後可以凍結。

**Codex 確認沒有問題的部分：**
- 核心的時間因果成立：
  - 事件評估的入口會先認領一次性步驟，才載入 parquet；
  - 擾動尚未揭曉的結果，不會改變在它之前的預測。
- warm-up 透過同一個時鐘更新、不計分，這個做法沒問題。
- 經 OPS 校準的盛行率，是公平的同條件比較基準。
- ONS 更新與 A-norm 廣義投影的代數正確。

### 逐條處理

1. **（高）其他評估器可以繞過一次性認領。** 已修。
   - **問題：**
     - `multi_sample_external_audit.py` 只要 protocol 範圍包含二元格，就會直接讀 lockbox 資料；
     - `external_value_conditional_interval_audit.py` 對 lockbox 根本沒有任何保護。
   - **修法：** 改在所有評估器共用的資料載入點 `external_cohort` 統一把關。
     - SICdb／HiRID 只有在入口程式已經認領一次性步驟（`args.claimed_step`，並且確認 ledger 裡真的有認領紀錄）之後才能讀取。
     - 共用這個載入點的評估器：multi-sample、value-conditional、activity／module-conditional、metrics report、calibration check、事件評估、v10 連續格評估。
   - **刪除：** `multi_sample_external_audit.py` 自己的 lockbox 分支與 `--protocol` 參數（不保留相容層）。
   - **新增：** `is_claimed` 加在 `osler_jepa/frozen_protocol.py`。
   - **對 v10 的影響：**
     - 這三個檔案（另含 `frozen_protocol.py`）都在 v10 的雜湊清單裡，修改後 v10 的 `verify_protocol` 就不再通過。
     - v10 的兩個 lockbox 都已經評分完畢，ledger 也不允許重跑，所以不影響已經產生的結果。
     - v10 凍結時的原始檔案保留在 oslian/Model 的 `adaptive_lockbox_v10/`，雜湊值記錄在 v10 protocol 裡。
   - **沒有修改：** `site_adaptive_external_audit.py`（v10 的連續格評分器）。
2. **（高）雜湊驗證的檔案，可能和實際執行的程式不是同一份。** 已修。
   - lockbox 評分必須從專案資料夾執行（`cwd == PROJECT`）。這樣 protocol 的相對路徑、載入的模組和設定檔，都會解析到同一個專案。
   - `loaded_project_files()` 新增兩道防線：
     - 和專案模組同名、卻從專案外載入的模組，直接報錯；
     - 專案內不是從 `.py` 原始檔載入的模組，例如 `.pyc`，也直接報錯。
3. **（中）沒有檢查訓練資料是否都有雜湊。** 已修。
   - 事件格各 horizon 的 MIMIC 訓練檔，都必須列在 `frozen_training_data_sha256`。
   - 計分的格子必須和 protocol 的 `scored_cells` 完全一致。
   - 測試涵蓋三種情況：缺少雜湊、訓練檔被改過、格子不一致。
4. **（高）獨立性的說法要收窄，來源要能驗證。** 已修，並在 v11 文件揭露。
   - **程式：**
     - v11 新增 `previous_protocol` 欄位（指向 v10），而且 v10 protocol 檔也必須列在 v11 的雜湊清單裡；
     - 評分前檢查：v11 addendum 記錄的 1h／3h 抽取檔，必須和 v10 addendum 記錄的雜湊逐位元相同。
   - **文件：**
     - 不再說這兩個資料庫「未被碰過」；
     - 改稱：在先前用過的外部資料庫上，新凍結的二元任務評估；
     - 揭露 v10 已經有的結果、所有事後決定，以及標籤存取的邊界。
5. **（中）模擬的揭曉時間，不等於前瞻情境下真的拿得到結果。** 在 v11 揭露。
   - 這是回溯性、依觀測時間重建的 replay。其中：
     - HiRID 用最終值，可能包含之後才輸入的更正；
     - SICdb 的事件定義是對「每小時彙總值」設門檻，和瞬間的低血氧不完全相同；
     - HiRID 一個病人代碼就是一次入院；
     - 承接 v10 已揭露的 replay 性質。
   - 輸出的 JSON 加上 `claim_scope` 欄位。
6. **（中）合併計算的校準，只能支持合併層級的宣稱。** 已修，並揭露。
   - 校準門檻（點估計）明確定位為研究容許範圍，不是統計等價檢定。
   - 每個樣本、每個方法新增不納入 gate 的診斷：
     - 以病人為 cluster 的 bootstrap 95% CI；
     - 依醫院、病房、資料流階段（各院時鐘順序三等分）分組的校準，附支援數；樣本數不夠的組只列計數。
   - SICdb／HiRID 的合併校準就是院層級的校準。eICU 的合併結果不代表每家醫院都校準良好。
7. **（中）合法的邊界輸入會讓一次性評估當掉。** 已修。
   - `calibration()` 遇到以下情況，回傳 `identifiable: False`，不再拋出例外，gate 判為不通過：
     - 空資料；
     - 只有一種結果；
     - 機率不合法（NaN 或超出 [0, 1]）；
     - 斜率擬合沒有收斂。
   - CITL 的搜尋範圍改用保證包含解的 ±(max|logit p| + |logit 事件率| + 1)。
   - OPS 遇到不合法的機率，該列輸出 NaN，而且永遠不拿它來更新。
   - `score_stream` 只要有任何不合法的預測，該資料流就判為不通過，不會當掉。
   - 測試涵蓋：單一類別、機率等於 1e-6、NaN、超出範圍、分組支援數不足。

### 驗證

- **新增或修改的測試：**
  - 事件 lockbox：17 個；
  - OPS：6 個；
  - 校準：7 個；
  - multi-sample：5 個。
- **相關套件共 90 個測試，全部通過**，包括 v10 的連續格 lockbox 測試。
- **突變測試：** 7 道新防線逐一拿掉，每一次都有測試失敗：
  - 執行目錄；
  - 計分格子；
  - 訓練雜湊；
  - 前一版 addendum；
  - 模組來源；
  - 雜湊清單；
  - `external_cohort` 的把關。
- 測試用暫存的 ledger，真正的 `causal_executions/` 沒有被動到。
- **開發庫參考值重跑：** 用修正後的程式重跑三個開發庫，輸出 `site_adaptive_event_{db}_20261029.json`。
- **Codex 環境的限制：** 缺 `osler_jepa.recipe_availability`，所以只跑了 OPS 的測試，其餘只做靜態審查。這次修正後，完整測試已在本機 `.venv` 跑過。

---

freeze after listed fixes

1. **High — Other evaluators bypass the one-shot claim.**  
   **Location:** `multi_sample_external_audit.py:73–92`; `external_value_conditional_interval_audit.py:388–413`.  
   **Failure scenario:** Adding binary-event scope to v11 enables the older multi-sample evaluator to open external data without checking the addendum, enforcing deployment arguments, or claiming `binary_<db>`. In-memory control-flow probes confirmed both entry points reach external loading without a claim.  
   **Fix:** Restrict lockbox scoring to the authorized event entry point; make other evaluators reject these databases before loading data. Add regression tests for these alternate routes. Preserve the frozen v10 continuous implementation.

2. **High — Source hashes can cover a different checkout from the executing code.**  
   **Location:** `site_adaptive_event_audit.py:62–69,85–88`; `external_value_conditional_interval_audit.py:328–335`.  
   **Failure scenario:** Launch checkout B’s script while working in checkout A. Module discovery produces relative filenames from B, while verification hashes those filenames in A. Matching names can therefore conceal changed executing code. The discovery function also skips relative paths, bytecode files, and project modules imported from elsewhere.  
   **Fix:** Resolve manifest paths against one explicit project root, verify the actual loaded source paths and digests, and reject unexpected project-module origins. Test execution from another working directory and checkout.

3. **Medium — Training-data hash completeness is unchecked.**  
   **Location:** `external_value_conditional_interval_audit.py:332`; `site_adaptive_event_audit.py:85–98,149`; `test_site_adaptive_event_audit.py:35–40`.  
   **Failure scenario:** A protocol omitting `frozen_training_data_sha256` passes verification. Matching the training-template argument does not establish that its 1h/3h files were hashed. The test protocol itself omits training hashes.  
   **Fix:** Before claiming the step, require both resolved training files in the training manifest. Assert the exact two expected event contracts, and test missing and changed training inputs.

4. **High — Independence requires a narrower claim and documented provenance.**  
   **Location:** `REVIEW_BRIEF.md:53,57–62`; `BINARY_EVENT_REEVALUATION_AFTER_LEAK_FIX_20261028.md:60–64`.  
   **Failure scenario:** A new protocol digest creates fresh ledger steps even though these sites have already influenced project evaluation. Unused future-column names do not establish that related SpO₂ observations, previous results, or overlapping measurements never influenced development. This does **not** prove contamination; independence remains unverified.  
   **Fix:** Describe this as a newly frozen binary-task evaluation on previously used external sites. Disclose v10 results already available, all post-hoc choices, and the claimed label-access boundary. Bind v11 to the previous protocol/addendum receipts and verify the promised extraction-byte identity. Avoid calling the databases untouched.

5. **Medium — Simulated resolve times do not establish prospective availability.**  
   **Location:** `site_adaptive_interval_dev.py:126–132`; `results/hirid_observation_transitions_1h_v1.json:4,159`; `results/sicdb_observation_transitions_1h_v1.json:160–164`.  
   **Failure scenario:** HiRID uses final-value reconstruction: a later-entered correction could be supplied to OPS at an observation-based deadline preceding its actual availability. SICdb’s endpoint thresholds an hourly aggregate, which can differ from instantaneous hypoxemia. HiRID’s “patient” identifier represents an admission without readmission linkage.  
   **Fix:** Freeze these as retrospective observation-time replay results. Disclose final-value availability uncertainty, aggregation, matching tolerances, admission-level clustering, and the inherited replay properties. A prospective claim would require availability-aware extraction.

6. **Medium — Pooled calibration supports a pooled claim only.**  
   **Location:** `site_adaptive_event_dev.py:60–72`; `site_adaptive_event_audit.py:163–166`.  
   **Failure scenario:** Opposing hospital or subgroup errors can cancel in pooled CITL and slope. Also, 45,000–59,000 correlated rows do not establish calibration precision by themselves.  
   **Fix:** Keep the point bounds as explicitly chosen research tolerances, and freeze diagnostic uncertainty estimates plus hospital, care-unit, and temporal summaries with support counts. For SICdb/HiRID, pooled calibration is site-level calibration; pooled eICU results cannot establish calibration at every hospital. Do not interpret the point gate as statistical equivalence.

7. **Medium — Valid edge cases can crash the consumed evaluation.**  
   **Location:** `osler_jepa/event_calibration.py:23–27`; `osler_jepa/online_platt.py:44–56`.  
   **Failure scenario:** I reproduced calibration failures for one-class outcomes and for finite, two-class outcomes with probabilities of `1e-6`, where the CITL root lies outside `[-10,10]`. A NaN probability also crashes OPS; out-of-range probabilities are silently clipped.  
   **Fix:** Validate probability, outcome, and clock contracts. Define frozen handling for empty or non-identifiable calibration, inspect fitting convergence, and use a reliable CITL bracket. Invalid predictions must prevent a full pass. Add these cases to the tests.

The core timing checks held: the event entry point claims before parquet loading, and unresolved-outcome perturbations left preceding predictions unchanged. Warm-up updates are sound when processed through that same clock and excluded from scoring. The OPS prevalence comparator receives the same information and is a reasonable adaptive baseline. The ONS update and generalized A-norm projection are algebraically correct; independent numerical projection checks agreed. NaN outcomes are skipped without poisoning state. All five OPS tests passed.

Could not verify: literal agreement with the cited paper, omitted transitive modules, historical label access, v10 ledger receipts, extraction identity, patient-level results, or the unwritten v11 protocol. The other two test suites could not import because `osler_jepa.recipe_availability` is absent. All supplied Python files parsed, and all nine supplied v10-hashed files matched. No files were modified.