# SICdb 一次性 lockbox 結果（protocol v10，2026-10-27）

- **Protocol**：`adaptive_model_lockbox_protocol_v10_20261027.json`（sha `4b43aa2c…`）。
- **Addendum**：`causal_executions/4b43aa2c…/addendum_SICdb.json`，記錄 15 個檔案的雜湊：extractor、測試、資料字典、6 組 parquet 與報告。
- **評分收據**：`causal_executions/4b43aa2c…/continuous_SICdb.json`。
- **結果檔**：`site_adaptive_lockbox_SICdb_v10.json`（sha `8b84bedb…`）。
- **執行**：約 3 小時 50 分，一次跑完，沒有當掉，也沒有任何修正。

## 主要終點

| | 數值 |
|---|---|
| 可評估的連續格 | 47 |
| 不可評估 | 6 個 glucose_poc 格：SICdb 沒有床邊血糖項目，事前已宣告 |
| **在 `point_and_dtaci_adaptive` 下穩定** | **44 / 47** |
| 靜態 v2（沒有任何線上更新） | 4 / 47 |

各方法的穩定格數：

| 方法 | 完整 gate 穩定 | 點預測穩定 |
|---|---|---|
| static_v2 | 4 | 27 |
| weight_site_unit_0.02 | 27 | 27 |
| point_and_interval_adaptive | 44 | 44 |
| **point_and_dtaci_adaptive（凍結形式）** | **44** | **44** |

- **靜態模型**：
  - 41 格區間不過，主要是過度覆蓋（病人層級覆蓋率中位數 0.94）。
  - 19 格點預測輸給 persistence，大多是 3h／6h 的生命徵象、血球與電解質。
- **線上校正**：affine_0.005 點校正加上 DtACI 後，這兩類問題大多消失。

## 沒通過的 3 格

| 格 | 完整 gate 通過比例 | 原因 |
|---|---|---|
| glucose@1h | 2/5 | 區間過度覆蓋（病房層級最高 0.937）；只有 155 列 |
| chloride@12h | 3/5 | 點預測只有 3/5 抽樣勝過 persistence（MAE 比 0.997）；eICU 的 chloride@12h 也沒過，Emory 的 v8 結果也是 |
| creatinine@12h | 3/5 | 點預測只有 3/5 抽樣顯著（MAE 比 0.991） |

## 開發庫參考（in-sample，不是預測值）

eICU 40、NWICU 36、Zigong 15、Emory 41。SICdb 是 44/47，而且是第一個完全沒看過標籤的資料庫。

## 判讀與限制

- **判讀規則**（protocol）：SICdb 與 HiRID 兩庫都穩定，才算獨立重現。目前只有 SICdb，所以這 44 格都還只是**候選**。
- **結果檔的 `independent_validation: false`**：這是程式寫死的欄位，不代表這次是開發資料。
- **glucose 格**：SICdb 的 glucose 是中央實驗室血糖，量測稀疏，只有 155–1,764 列。這些格雖然穩定，但樣本小。
- **揭露事項**：
  - 入 ICU 病房：2,606 個 case 在入院那一秒記了兩個病房，取 id 較後的那一筆；
  - replay 性質 #3–#5，見 v10 protocol；
  - 標籤以採檢時間計；
  - 每小時聚合的標籤代表一段時間，不是單點量測；
  - cohort 包含中度照護病房。
- `causal_claim_allowed: false`；`clinical_promotion_allowed: false`。任何 registry 變更都要使用者同意。
