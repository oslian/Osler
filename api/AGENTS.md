# 專案工程規則

## 強制執行方式

- 開始任何分析、規劃、修改、測試或命令執行前，必須先完整閱讀本檔。
- 若工作目錄內還有更具體的 `AGENTS.md`，也必須一併閱讀並遵守。
- 除非與系統或開發者層級指令衝突，以下規則均為強制要求。

## 核心規則

1. 不保留向後相容。過時的實作、介面或格式直接刪除；不新增相容層、不寫 migration、不保留舊版 fallback。本條針對舊版相容，不得誤刪目前需求明確要求的安全防線，例如 fail-closed 或經驗證的 persistence fallback。
2. 選擇能滿足目前需求的最簡單實作。不要建立預防性抽象、沒有必要的配置層或未被目前需求使用的擴充點。
3. 系統可以有清楚且長期可維護的分層，但實作順序必須先跑通最小端到端版本，再逐層增加能力。不得為了尚未完成的複雜度拆掉已經能正確運作的流程。
4. 元件保持模組化並維持關注點分離。每個模組應有單一清楚責任，跨層依賴必須明確。
5. 優先使用成熟、持續維護且已有實務驗證的函式庫。沒有明確理由，不自行重寫既有成熟能力。
6. 新增套件或自行實作前，先檢查專案目前的依賴、標準函式庫與既有工具是否已能完成需求。不得預設現有依賴做不到。
7. 架構決策以長期維護為準。不接受「先暫時這樣，以後再全部換掉」的方案；最小實作也必須位於正確的長期架構方向上。
8. 設計新功能前，先研究成熟產品、主流函式庫或已驗證系統如何解決相同問題。優先採用已證實有效的模式，不從零發明。

## 任務開始檢查

開始動手前，確認已完成以下事項：

- 已閱讀本檔與適用範圍內的其他 `AGENTS.md`。
- 已閱讀相關程式碼、測試、依賴與現有架構。
- 已確認哪些舊實作應直接刪除，而不是包裝成相容層。
- 已找出可交付的最小端到端路徑。
- 已確認現有依賴是否足以完成需求。
- 已參考成熟實作或已驗證模式；若未採用，已能說明具體原因。
- 已確認方案同時符合目前需求與長期架構。

## 全身預測強制架構（expert + 自身 residual + 跨格相依層）

本節適用於所有全身 factual observation/forecast 的訓練、外部驗證、權重與 runtime candidate。開始任何模型任務前，必須先完整閱讀本節。

依據（2026-10-31，開發資料，5 個 MIMIC 抽樣，eICU、NWICU、Zigong）：
- 交換各格點預測沒有增益：只用快照時 +0.04%，加上軌跡時 −0.10%（`STACKED_EXPERT_DEV_RESULT_20261031.md`、`STACKED_EXPERT_TRAJECTORY_DEV_RESULT_20261031.md`）。
- 自身目標的軌跡有增益：+1.1%；只看自身（含軌跡）的 residual 比 expert 好 2.9%，屬事後觀察。
- 跨格 Gaussian copula 相依層在多變數分數上 robust 地較好：跨時間窗 −19% 到 −54%，跨目標 −0.1% 到 −0.5%（`JOINT_FORECAST_DEV_RESULT_20261031.md`）。
- t-copula 沒有在事先宣告的規則下勝過 Gaussian（`T_COPULA_JOINT_DEV_RESULT_20261031.md`）。
- 依現值分組的 Mondrian 邊際勝過單一誤差分布：單一事件在 46/83 組合上 robust 地更接近 1，energy score 6/6 robust 較好（`TAIL_CALIBRATION_DEV_RESULT_20261031.md`）。
- 每家醫院的線上 Mondrian 邊際（線上點校正 + Mondrian MultiQT）勝過靜態 Mondrian：單一事件在 56/83 組合上 robust 地更接近 1，energy score 6/6 robust 較好，校準範圍內組合 39 → 57（`ONLINE_TAIL_DEV_RESULT_20261031.md`）。
- 上述 NWICU 數值受開發 cohort 的 stay_id 撞號影響（NWICU 與 Zigong 都從 1 編號，2,798 個撞號）；eICU、Zigong 不受影響，各開發文件已加勘誤。`stacked_expert_dev.load_cohort` 已改用「資料庫:編號」。同一形式完全修正後的 NWICU 演練：自身 residual 點預測 7 格判定翻轉（robust 格 17 → 22，MAE 最大差 2.1%），單一事件與相依層沒有判定翻轉。

外部證據（一次性 lockbox，INSPIRE，protocol v13，2026-11-02；`INSPIRE_LOCKBOX_RESULT_V13_20261102.md`，registry `adaptive_replication_registry_20261102.json`）：
- 只有一個外部站點：首爾大學醫院的術後 ICU，照護單位以手術科別代理，數值經量化並截在第 2.5/97.5 百分位。
- 第二層：自身 residual 在 24/36 格 robust（map、potassium 全部；bicarbonate 5/6；glucose 3/6；heart_rate 3/6；creatinine 只有 48h）。creatinine 1–24h 沒有勝過 expert 與 persistence。
- 邊際（線上 Mondrian M）的單一事件：17/24 可評估組合 robust；HCO₃ < 18 六個時間窗全部預測過多（O/E 0.30–0.66）；HR > 120、K ≥ 6 因外部截尾無法評估。
- 第三層：兩個區塊都 robust；跨時間窗 variogram −19% 到 −23%，跨目標只有約 −0.03%。
- 這些都只是單一 lockbox 的候選。第二層取代 expert、第三層正式啟用，都需要第二個未使用的外部資料庫。

JEPA 只保留為研究比較（見下方「JEPA 研究線」）。

### 唯一正式架構

```text
validated target-horizon expert cells (frozen)              每格的點預測錨點
                ↓
own-target bounded residual, one per target × horizon      每格只看自己目標的資訊（含自身軌跡）
                ↓  per-cell marginal predictive distribution
cross-cell dependence layer (Gaussian copula)             輸出端：跨目標、跨時間窗的聯合相依
```

**第一層：凍結 expert**
- 已驗證的 target × horizon expert 格子保持凍結，是每格的預測錨點，不重新發明或捨棄。
- expert 預測對後續各層的所有 fit、selection、conformal 病人都必須是 out-of-sample：expert recipe 擬合在不含這些病人的 MIMIC 病人上。

**第二層：自身 residual（每格的點預測與邊際分布）**
- 每格輸出 = 凍結 expert 預測 + 有界 residual。residual 為 0 時必須精確等於 expert。
- residual 只能看自己目標在 anchor 時刻已知的資訊：
  - 現值、距上次量測時間；
  - 各時間窗的 expert 預測變化量；
  - 自身軌跡：現值減去同一住院中不晚於 t − L 的最晚 anchor 的值，L ∈ {1, 3, 6, 12, 24} h。軌跡只能讀較早的 anchor，anchor 集合不得依標籤而定。
- 缺值保持缺值，由模型原生處理；不得填物理 0。
- 模型使用成熟函式庫（目前是 `sklearn.ensemble.HistGradientBoostingRegressor`）。超參數固定，並在看結果前寫入 dev rule。
- 跨格交換點預測（其他目標的輸入）只能作為 ablation。只有在事先宣告的規則下穩定勝過自身 residual，才可為該格啟用。
- 相依層使用的每格邊際分布是凍結部署形式下、每家醫院的線上 Mondrian 預測分布（`jepa_deployment_form_dev.deploy_distribution`，member 由 `online_joint_dev.at_member_levels` 取得）：
  - 點 = 共享部署時鐘上的線上點校正（凍結的 `site_adaptive_point_amendment_20261024.json`）；
  - 分布 = 每家醫院各自的 Mondrian MultiQT，使用凍結的 v12 part C 設定（`threshold_event_candidates_20261029.json`）；
  - 起始分數 = MIMIC conformal 病人的帶號誤差，除以只用 MIMIC selection 病人擬合的 v2 quintile 尺度；分組 = 現值的 tail8 分位數分組；
  - 只有暖機後的列可評分與輸出；暖機中的列不得輸出邊際或聯合事件機率。
- 尺度不得掉到 0：任何以中位數絕對誤差為尺度的形式，都必須設記錄精度下限（`threshold_event_form.resolution`）。
- 已知限制（必須在任何使用此邊際的結果中揭露）：
  - 跨時間窗的 variogram score 比靜態 Mondrian robust 地較差（eICU +4.7%、NWICU +5.9%、Zigong +29%）。每個時間窗各自追蹤分位數，同一病人跨時間窗的成員間距不一致；跨時間窗的聯合情境（例如軌跡形狀）不得聲稱優於靜態 Mondrian。
  - 稀有事件 K ≥ 6 在 eICU、NWICU 變成預測過多（O/E 約 0.26–0.94，校準範圍內的時間窗為 0）。每家醫院高值區的線上資料太少；K ≥ 6 的事件機率不得視為已校準。
  - HCO₃ < 18 在外部 lockbox（INSPIRE）六個時間窗全部預測過多（O/E 0.30–0.66）；HCO₃ < 18 的事件機率不得視為已校準。
- 靜態 Mondrian（`tail_calibration_dev.calibrate`、`osler_jepa.joint_copula.conditional_members`）只保留為比較對照，不作為邊際輸出。
- 正式的點預測區間仍是凍結部署形式下的 DtACI；此邊際只用於相依層的聯合情境與事件機率。

**第三層：跨格相依層（輸出端的聯合調整）**
- 不改變任何一格的點預測或邊際分布，只決定各格之間的聯合關係。
- 形式：Gaussian copula（`osler_jepa.joint_copula`）。
  - 相關係數由各格帶號誤差的成對 Spearman 換算（2 sin(πρ/6)），只用 MIMIC selection 病人估計；
  - 支持不足的配對設為 0；
  - 修正為有效的相關矩陣。
- 輸出的聯合情境是 member ensemble：每個 member 是一組跨目標、跨時間窗一致的未來值。聯合事件機率 = 同時發生的 member 比例。
- 以多變數分數驗證（variogram score 為主，energy score 為輔），對照組是「同邊際、各格獨立」。聯合事件另報 O/E 與 Brier。
- 聯合事件的品質受限於邊際尾端的校準。邊際尾端未通過校準的事件，不得輸出聯合事件機率。
- 誤差相依只能稱為預測關聯，不得稱為生理機制或因果。
- 跨目標區塊的增益在開發（−0.1% 到 −0.5%）與外部（INSPIRE 約 −0.03%）都很小。即使通過判定，也不得宣稱跨目標聯合情境有實質改善；實質增益只來自跨時間窗區塊。

### 強制驗證與輸出邊界

- 每個 target × horizon 的點預測單獨驗證，不要求整個模型一次接管全部格子。
- 第二層每格必須在凍結的部署形式下（線上點校正、DtACI、共享部署時鐘），於該資料庫的 point axes 上同時勝過既有 expert anchor 與 persistence。
- 每格的 conformal 區間必須通過 patient、hospital、care-unit gate。不得在 runtime 手寫固定 interval half-width；區間未通過時輸出 `null` 與明確的 calibration status。
- 第三層必須在每個資料庫與區塊（跨目標、跨時間窗）上，以病人為單位的 cluster bootstrap 勝過各格獨立。未通過的區塊輸出各格獨立的邊際，並註明 dependence status。
- fit、selection、conformal 病人互斥（`osler_jepa.patient_split`）。
- 多抽樣規則：每格與每個相依區塊都在 5 個不同的 MIMIC 病人訓練抽樣上重新擬合並評分，至少 80% 抽樣通過才算穩定通過。單一抽樣的通過不得作為晉升或分級依據。
- 開發資料庫（eICU、NWICU、Zigong）的結果只是開發證據。獨立重現只能來自凍結 protocol 下的一次性外部 lockbox。
- 未通過的格子保持既有 validated expert 或 persistence；此 fallback 是安全政策，不是舊版相容層。
- 所有輸出維持 factual research-only，固定 `causal_claim_allowed: false`；沒有 RCT 或可靠因果識別時，不得輸出治療反事實或臨床決策宣稱。

### JEPA 研究線

- `osler_jepa.adjacent_hierarchy.AdjacentThreeLevelResidualAdapter`、`osler_jepa.patient_state.PatientStateAdapter`、`teacher_anchored_joint_jepa_audit.py` 與 `jepa_gate.py` 只作研究比較。不得寫入 runtime registry，不得產生可晉升的 checkpoint。
- `whole_body_joint_jepa.load_joint_event_examples` 從 2026-11-03 起把 stay_id 改成「資料庫:編號」。修正前產生、含 NWICU 與 Zigong 的 JEPA 研究結果受撞號影響，不得解讀為這兩個資料庫的證據。
- 研究執行仍遵守這些程式內建的結構限制（解剖器官節點、`canonical_target_organ_membership`、`canonical_organ_adjacency`、無 system↔body 直接邊、staged training、matched ablation），以維持結果可比較，並執行 `test_adjacent_hierarchy.py`。

### 任務開始追加檢查

- 已確認點預測是凍結 expert 加上有界的自身 residual，residual 為 0 時精確等於 expert。
- 已確認自身 residual 只讀自己目標的資訊；跨格輸入只出現在明確的 ablation 中。
- 已確認軌跡特徵只讀較早的 anchor，且不依標籤而定。
- 已確認相依層不改變任何邊際，相關係數只用 MIMIC selection 病人估計。
- 已確認邊際是每家醫院的線上 Mondrian 分布（`deploy_distribution`），只用暖機後的列，尺度有記錄精度下限，並揭露跨時間窗 variogram 與 K ≥ 6 兩個限制。
- 已確認 expert 預測對後續各層的所有病人都是 out-of-sample。
- 已確認 dev rule（超參數、對照組、通過條件）在看結果前寫好。
- 已確認 JEPA 只出現在研究比較，而非正式候選。
- 已執行 `test_stacked_expert.py`、`test_joint_copula.py`、`test_tail_calibration_dev.py`、`test_online_tail_dev.py`。
