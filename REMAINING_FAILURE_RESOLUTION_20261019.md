# 剩餘失敗的診斷與處理（2026-10-19）

全部在開發資料庫（eICU、NWICU、Zigong；皆已使用過）上進行，非獨立驗證。gate 依研究者決定維持現有標準（醫院／病房平均納入所有群組）。

## 診斷

| 問題 | 診斷 | 證據 | 處理 |
|---|---|---|---|
| Zigong 腎功能點預測（creatinine@12h/24h、bun@24h） | 資料限制：感染族群 24h 內 creatinine 幾乎不變，且只有檢驗、無尿量與治療 | creatinine@24h：510 位病人，模型比 persistence 好 0.004 mg/dL，95% CI −0.019 至 +0.011；縮減特徵重擬合無效（MAE/persistence 0.975 → 0.972）`reduced_feature_diagnostic_20261019.json` | 不宣稱點預測改善；不再追 |
| eICU hematocrit@3h 等點預測、NWICU 病房平均區間 | 醫院／病房平均讓 1–12 列的群組同等權重 | hematocrit@3h 病人軸 −0.29（CI −0.36 至 −0.22）通過，醫院軸 CI 上界 +0.07；NWICU 大病房覆蓋 0.894–0.902、小病房 0.69–0.80 `interval_site_size_diagnostic_20261019.json` | 研究者決定維持 gate；誠實記錄為未通過 |
| 新醫院上線初期區間偏窄 | MIMIC 校準在新醫院起步偏窄，DtACI 需約 200 筆結果才收斂 | eICU 每家醫院前 200 筆覆蓋 0.83–0.86，之後 0.87–0.90 | DtACI 起始倍數 θ₀（見下） |

## 起始倍數 θ₀（`site_adaptive_dtaci_prior_dev.py`）

依 robust conformal（Cauchois et al., JASA 2024），新醫院從 exp(θ₀) × 校準分位數起步，DtACI 再往下或往上調。留一資料庫選擇（規則評分前寫定）：

| 留出庫 | 另兩庫選出的 θ₀ | 留出庫區間穩定通過（θ₀ 選定 / θ₀=0） |
|---|---|---|
| eICU | 0.20 | 38 / 31 |
| NWICU | 0.15 | 37 / 36 |
| Zigong | 0.15 | 18 / 18 |
| 合計 | | **93 / 85 → 採用** |

凍結 θ₀ = 0.15（起始寬度 ×1.16）；寬度中位數增加 eICU 5%、NWICU 0.2%、Zigong 3%。已接入 `site_adaptive_external_audit.py`（`--dtaci-prior`）。

## 目前系統在開發庫的成績（`site_adaptive_final_external_{eICU,NWICU,Zigong}_20261019.json`）

設定：外部開發庫選出的 recipe + persistence 退回（platelets@1h）+ 點預測線上校正 + DtACI（θ₀ = 0.15）；5 個 MIMIC 抽樣，穩定 = 完整 gate ≥ 80%。

| 資料庫 | 先前系統 | 目前系統 |
|---|---|---|
| eICU | 27 / 47 | **36 / 47** |
| NWICU | 37 / 44 | **37 / 44** |
| Zigong | 10 / 21（lockbox） | **16 / 21** |
| 合計 | 74 / 112 | **89 / 112** |

**這是 in-sample 數字**：recipe 與 θ₀ 都是在這三個庫上選的。θ₀ 部分有留一資料庫估計（區間 85 → 93）；recipe 部分沒有。真正的效果須以凍結後的 SICdb / HiRID 一次性驗證為準。

## 剩下的 23 個失敗

- eICU（11）：glucose@1h/6h/12h 醫院平均覆蓋 0.81–0.85（glucose 特有）；potassium@3h/12h 醫院平均過度覆蓋 0.93–0.95（θ₀ 的副作用）；creatinine@6h/12h、bicarbonate@12h 覆蓋略低；hematocrit@3h、hemoglobin@3h、sodium@24h 醫院軸點預測（小醫院）。
- NWICU（7）：除 sodium@24h 外全部是病房平均覆蓋（小病房）。
- Zigong（5）：腎功能點預測（資料限制）、creatinine@6h 區間、platelets@3h 不穩定。

independent_validation: false；causal_claim_allowed: false。

## 2026-10-20：eICU glucose 區間（`interval_scale_dev.py`，未採用）

診斷（`glucose_interval_volatility_diagnostic_20261020.json`）：外部 glucose 覆蓋不足集中在當前值 > 250（0.75–0.83）與距上次 anchor 變化 > 100（0.81–0.84）。v2 的五分位值分組最高組為 ≥ 186，組內誤差中位數 44（180–250）到 203（> 350）；eICU 實驗室血糖族群 > 350 佔 14%，MIMIC 2%。

候選（`osler_jepa/interval_scale.py`）：`quintile`（現行）、`tail8`（2/10/25/50/75/90/98 分位）、`tail8_change`（再乘上近期變化的分離倍數）。規則評分前寫定：三庫區間穩定通過合計最多者，須嚴格多於 `quintile`。

| 方案 | eICU | NWICU | Zigong | 合計 |
|---|---|---|---|---|
| quintile | 39 | 37 | 18 | **94** |
| tail8 | 36 | 36 | 19 | 91 |
| tail8_change | 39 | 35 | 19 | 93 |

**不採用。** `tail8_change` 修好 eICU glucose@1h、@6h（醫院平均 0.84 → 0.87），但讓 eICU potassium@6h（醫院平均 0.93，過寬）、chloride@12h，以及 NWICU creatinine@24h、glucose@24h 失去通過。留一資料庫估計：全域選擇 92 vs 94；依 target 分別選擇也是 92 vs 94。glucose 的改善證據只來自 eICU（Zigong 無 glucose），留出 eICU 時另兩庫選的是 `quintile`，即這個修正無法在其他資料庫上被證實可轉移。

部署維持 `quintile`（`site_adaptive_external_audit.SCALE_SCHEME`）；`osler_jepa/interval_scale.py` 保留作為現行尺度的實作與本次開發紀錄。

## 2026-10-20：小醫院／小病房（新穎性加寬，未採用）

診斷（`interval_small_group_diagnostic_20261020.json`，目前凍結形式、抽樣 0，置換檢定保持群組大小打散覆蓋）：

- 系統性偏低（p < 0.05）：eICU glucose@6h/12h、creatinine@12h；NWICU glucose@1h/12h、platelets@1h、chloride@12h。小群組覆蓋 0.62–0.79，大群組 0.87–0.91。
- 雜訊範圍內（p 0.10–0.21）：eICU glucose@1h、creatinine@6h、bicarbonate@12h；NWICU hemoglobin@12h。純雜訊下群組平均的 2.5 百分位可低至 0.80–0.81，即使逐列校準完美也常無法通過現行 gate。
- NWICU creatinine@6h 是小病房過寬（0.938）。

方法：依有限樣本 conformal 校正的精神，新醫院／病房的輸出區間加上 θₙ·k/(n+k)（n = 該群組已回來的結果數），不影響 DtACI 學習。候選 θₙ ∈ {0.1, 0.2, 0.3} × k ∈ {10, 30}；留一資料庫規則評分前寫定。

| | eICU | NWICU | Zigong | 留一資料庫合計 |
|---|---|---|---|---|
| 現行 | 39 | 37 | 18 | 94 |
| 最佳候選 θₙ=0.1, k=10 | 36 | 37 | 20 | 91 |

**不採用**（`site_adaptive_novelty_dev_20261020.json`）。它修好 eICU creatinine@6h/12h、bicarbonate@12h，但讓 6 個原本合格的格醫院平均覆蓋超過 0.93（例如 potassium@6h 0.92 → 0.95）：小醫院並非每個指標都偏窄，統一加寬無法區分。程式已移除，`osler_jepa/site_adaptive_conformal.py` 恢復為 v5 雜湊。

## 2026-10-20：小醫院 — 網路共享 DtACI（採用，v6）

方法（`site_adaptive_network_dev.py`）：多院部署時，各院整段住院隨機交錯成一條網路資料流；DtACI 的 site 層由所有醫院共同學習，unit 層改以醫院為單位（累積 20 個結果後啟用）。小醫院在自身結果不足時借用網路的調整。單一醫院部署完全不變（NWICU 逐位元相同）。

預先宣告規則：eICU 醫院依雜湊分成兩半，一半選、另一半評，雙向加總須嚴格多於現行。結果（`site_adaptive_network_dev_20261020.json`）：留出合計 77 vs 73 → **採用**，醫院層步長比例 1.0。已接入 `site_adaptive_external_audit.py`（`--network-dev`），凍結為 `adaptive_model_lockbox_protocol_v6_20261020.json`。

eICU 完整系統重跑（`site_adaptive_final_external_eICU_network_20261020.json`，非獨立：選擇用過 eICU）：

| | 完整 gate 穩定 | 區間穩定 |
|---|---|---|
| v5（各院獨立） | 36 / 47 | 39 |
| v6（網路共享） | **39 / 47** | **42** |

- 新通過 6 格：potassium@3h、@12h（醫院平均由過寬 0.94 降至 0.91–0.93），creatinine@6h、@12h，glucose@6h，bicarbonate@12h。
- 失去 3 格：urine_output@1h、@3h（醫院平均 0.88 → 0.86–0.87），platelets@1h（persistence 退回格，醫院平均 0.91–0.94 偏寬）。
- 仍未通過：glucose@1h、@12h（區間），hematocrit@3h、hemoglobin@3h、sodium@24h（醫院軸點預測，小醫院），以及上述 3 格。

Emory（單一醫院）結果在 v6 下不變且仍有效；SICdb / HiRID 為單一醫院，無法驗證網路共享。

## 2026-10-20：疾病族群 — 醫院匹配訓練樣本（診斷，未進入開發評估）

方法（`site_matched_training_diagnostic.py`）：共變數偏移重要性加權（Shimodaira 2000）。以該院第一天的輸入（不看結果、只用該院有記錄的共變數）訓練 MIMIC-vs-該院分類器，依權重從完整 MIMIC 抽出同樣大小（15,000 列）的訓練樣本，部署形式不變。抽樣 0。

| 格 | MAE / persistence（均勻 → 匹配） | 病人軸點預測 |
|---|---|---|
| Zigong creatinine@24h | 0.970 → 0.968 | 未過 → 未過 |
| Zigong creatinine@12h | 0.986 → 0.989 | 過 → 未過 |
| Zigong bun@24h | 0.981 → 0.975 | 未過 → 未過 |
| Zigong creatinine@6h | 0.981 → 0.981 | 未過 → 未過 |
| Zigong platelets@3h | 0.944 → 0.908 | 未過 → 過 |
| Zigong potassium@6h（對照） | 0.875 → 0.871 | 過 → 過 |
| NWICU creatinine@24h（對照） | 0.891 → 0.914 | 過 → 過 |
| eICU creatinine@24h（對照） | 0.905 → 0.917 | 過 → 過 |
| eICU hematocrit@3h（對照） | 0.908 → 0.905 | 過 → 過 |

結論：腎功能格幾乎不變（±0.003），NWICU/eICU 的 creatinine 反而變差 1–2%；只有 platelets@3h 改善。沒有進入 5 抽樣開發評估的理由。

**疾病族群問題的判斷**：Zigong 腎功能的限制是資訊不足，不是訓練族群不匹配——訓練樣本已調整成像 Zigong 的病人，預測仍與 persistence 打平。該族群 24h creatinine 幾乎不變（平均變化 0.006 mg/dL），而能預測腎功能變化的輸入（尿量、CRRT、腎毒性藥物、液體）在 Zigong 都沒有。這只能靠資料解決，不是方法能解決的。

## 2026-10-20：缺少的變數 — 推導變數（採用，v7）

盤點：
- 根本沒有量測（無法解決，正確行為是標為不可評估）：尿量（NWICU、Zigong、Emory）、鈉（Emory）。
- 時間精度不足（無法解決）：Zigong 生命徵象（護理紀錄只到日）。
- 可由其他量測推導：Emory 碳酸氫根（HCO3 欄 0.8%，但 pH 與 PaCO2 有 13%）。

方法：`osler_jepa/derived_measurements.py` — 計算碳酸氫根（Henderson–Hasselbalch，0.0307 × PaCO2 × 10^(pH−6.1)，即 MIMIC itemid 50803 的定義）與 MAP（(SBP + 2·DBP)/3）；只在該時間沒有實測值時補上。NWICU 的 MAP 推導改用此共用函式（公式相同）；Emory extractor 加入碳酸氫根推導。凍結為 `adaptive_model_lockbox_protocol_v7_20261020.json`；之後的 SICdb / HiRID extractor 套用相同規則。

Emory 延伸評分（一次性；碳酸氫根結果從未被讀過；新萃取檔，原 Emory 萃取檔與雜湊未動；`emory_lockbox_bicarbonate_addendum_20261020.json`、`site_adaptive_lockbox_Emory_bicarbonate_20261020.json`）：

| 格 | 資料對（原 → 推導後） | 結果 | MAE / persistence | 靜態 v2 |
|---|---|---|---|---|
| bicarbonate@12h | 28 → 2,019 | **穩定通過** | 0.921 | 未過 |
| bicarbonate@24h | 7 → 1,112 | **穩定通過** | 0.905 | 未過 |

Emory 連續格合計：**42 / 43 可評估格穩定通過**（原 40/41 + 2）。仍不可評估：sodium@24h、urine_output@1h/3h/6h（無來源資料）。

## 2026-10-20：血糖定義 — 新增床邊血糖格（採用，v8）

盤點（只看資料對數量）：ICU 主要的血糖量測是床邊血糖（MIMIC 1h：化學 13,355 對 vs 床邊 71,772；eICU：25,002 vs 219,289）。NWICU 沒有床邊血糖項目；Zigong 幾乎只有血氣血糖；Emory 的「serum glucose」密度（每列 21%）高於 MIMIC 床邊血糖（11%），幾乎確定以床邊血糖為主（Emory 結果已加註更正）。

研究者核准（2026-10-20）依 LOINC / ISO 15197 的概念分離：保留化學檢驗 `glucose`，新增 `glucose_poc@1/3/6/12/24/48h`（`glucose_poc_catalog_extension_20261020.json`）。設定：eICU 模組 endocrine_stress、治療特徵同 glucose、hurdle 門檻 10 mg/dL、合理值 10–2000 mg/dL。

Recipe 選擇（同一預先宣告規則；只有 eICU 有床邊血糖；`glucose_poc_recipe_selection_dev_20261020.json`）：6 格皆 MPR，5 抽樣皆穩定勝過 persistence。

eICU 完整系統（含網路共享；非獨立；`site_adaptive_final_external_eICU_glucose_poc_20261020.json`）：

| 格 | MAE / persistence | 病人平均 | 醫院平均 | 結果 | 靜態 v2 |
|---|---|---|---|---|---|
| glucose_poc@1h | 0.913 | 0.877–0.887 | 0.904–0.927 | 穩定 | 穩定 |
| glucose_poc@3h | 0.899 | 0.887–0.893 | 0.888–0.903 | 穩定 | 穩定 |
| glucose_poc@6h | 0.874 | 0.894–0.898 | 0.873–0.899 | 穩定 | 未過 |
| glucose_poc@12h | 0.847 | 0.890–0.896 | 0.860–0.893 | 穩定 | 未過 |
| glucose_poc@24h | 0.872 | 0.884–0.891 | 0.879–0.886 | 穩定 | 穩定 |
| glucose_poc@48h | 0.843 | 0.888–0.891 | 0.869–0.877 | 穩定 | 穩定 |

6 / 6 穩定通過（對照：化學檢驗 glucose@1h、@12h 在 eICU 仍未通過）。部署 recipe 合併為 `deployment_recipes_20261020.json`（53 格）；凍結為 `adaptive_model_lockbox_protocol_v8_20261020.json`。獨立驗證需 SICdb / HiRID 能區分床邊血糖。
