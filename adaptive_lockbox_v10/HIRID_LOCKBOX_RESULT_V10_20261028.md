# HiRID 一次性 lockbox 結果（protocol v10，2026-10-28）

- **Protocol**：`adaptive_model_lockbox_protocol_v10_20261027.json`（sha `4b43aa2c…`）。
- **Addendum**：`causal_executions/4b43aa2c…/addendum_HiRID.json`，記錄 18 個檔案的雜湊：
  - extractor、測試、`hirid_variable_reference.csv`；
  - 來源揭露、250 份 parquet 的雜湊清單、Codex 第二輪審查；
  - 6 份 parquet 與 6 份報告。
- **評分收據**：`causal_executions/4b43aa2c…/continuous_HiRID.json`。
- **結果檔**：`site_adaptive_lockbox_HiRID_v10.json`（sha `ff0455ec…`）。
- **執行**：約 4 小時 14 分（14:26–18:40），一次跑完，沒有當掉，也沒有任何修正。
- **資料來源限制**：用的是非官方轉售副本，無法以官方 checksum 驗證，詳見 `HIRID_LOCKBOX_PROVENANCE_20261028.md`。這是使用者在 2026-10-28 的決定。

## 主要終點

| | 數值 |
|---|---|
| 可評估的連續格 | 38 |
| 不可評估 | 15 格，原因都是 `insufficient_external_support`，事前已宣告 |
| **在 `point_and_dtaci_adaptive` 下穩定** | **36 / 38** |
| 靜態 v2（沒有任何線上更新） | 9 / 38 |

15 個不可評估的格：
- glucose 與 glucose_poc 的 12 格：字典沒有區分中央實驗室、床邊與血氣血糖。
- hematocrit@3h、6h、12h：HiRID 沒有 hematocrit 項目。

| 方法 | 完整 gate 穩定 | 點預測穩定 |
|---|---|---|
| static_v2 | 9 | 27 |
| weight_site_unit_0.02 | 28 | 27 |
| point_and_interval_adaptive | 36 | 35 |
| **point_and_dtaci_adaptive（凍結形式）** | **36** | **35** |

`platelets@1h` 是 persistence fallback recipe（SICdb 也是），完整 gate 穩定。

## 沒通過的 2 格

| 格 | 列數 | 完整 gate | 原因 |
|---|---|---|---|
| creatinine@6h | 1,919 | 0/5 | 點預測 0/5 勝過 persistence（MAE 比約 0.993–1.000）；有 1 個抽樣區間覆蓋不足（病人層級 0.842） |
| creatinine@12h | 1,712 | 0/5 | 點預測 0/5 勝過 persistence（MAE 比約 0.990–0.999）；區間 5/5 通過 |

- HiRID 的 creatinine 量測稀疏，這兩格只有不到 2,000 列。
- 模型幾乎就是 persistence，沒有可顯著證明的增益。

## 與 SICdb 的重現情況（protocol 判讀規則）

判讀規則：兩庫都穩定才算**獨立重現**；只有一庫穩定則是**候選**。

| 類別 | 格數 | 格 |
|---|---|---|
| **兩庫都穩定：獨立重現** | **35** | bicarbonate@12h/24h；bun@24h；creatinine@24h/48h；heart_rate@3h/6h/12h/24h/48h；hemoglobin@3h/6h/12h；map@1h/3h/6h/12h/24h/48h；o2sat@12h；platelets@1h/3h；potassium@3h/6h/12h/24h/48h；respiratory_rate@3h/6h/12h/24h；sodium@24h；urine_output@1h/3h/6h |
| 只有 SICdb 穩定，HiRID 沒過 | 1 | creatinine@6h |
| 只有 SICdb 穩定，HiRID 不可評估 | 8 | glucose@3h/6h/12h/24h/48h；hematocrit@3h/6h/12h |
| 只有 HiRID 穩定，SICdb 沒過 | 1 | chloride@12h |
| 兩庫都沒過 | 1 | creatinine@12h |
| SICdb 沒過，HiRID 不可評估 | 1 | glucose@1h |
| 兩庫都不可評估 | 6 | glucose_poc 全部 |

兩庫都可評估的 38 格中，35 格在兩庫都穩定。

## 開發庫參考（in-sample，不是預測值）

eICU 40、NWICU 36、Zigong 15、Emory 41。SICdb 44/47、HiRID 36/38，這兩庫的標籤在評分前都沒有被看過。

## 判讀與限制

- **資料來源**：HiRID 用的是無法驗證的轉售副本（見 provenance 檔）。SICdb 原始檔的來源也還沒確認。所以「獨立重現」的證據強度受限於這兩個資料來源。
- **HiRID 只有一個科別**：病房層級的覆蓋率等於整體覆蓋率，無法說明跨病房的穩定性。
- **病人等於住院**：HiRID 沒有再入院的連結，所以 bootstrap 以入院為單位。
- **住院結束時間**：用最後一筆有效心率代替，HiRID 沒有出院時間。
- **標籤**：用最終值、依觀測時間重建，不代表前瞻情境下當時就拿得到。
- **重現格的範圍**：以生命徵象、電解質和血球為主，glucose 沒有被重現。
- **其他揭露**：replay 性質 #3–#5，見 v10 protocol。
- `independent_validation: false` 是程式寫死的欄位。
- `causal_claim_allowed: false`；`clinical_promotion_allowed: false`。
- **registry 沒有變更**：任何 registry 變更都要使用者同意。

## Registry（2026-10-28，使用者同意）

- 使用者同意將 35 格標為**已獨立重現**。
- 產出檔案：`adaptive_replication_registry_20261028.json`（schema `adaptive_replication_registry.v1`）。
- 產生方式：`build_external_expert_promotion_registry.py`，已改寫為讀取兩個 v10 lockbox 結果；測試 4 個全部通過。
- 狀態分布：`independently_replicated` 35、`candidate` 10、`not_promoted` 8。
- 兩份結果檔的 SHA-256 都必須與 ledger 中 `continuous_<db>` 收據記錄的輸出一致，builder 才會接受。
- registry 一併記錄：
  - 凍結的部署形式參數；
  - 資料來源限制；
  - 適用範圍限制：只有一個科別；glucose 沒有被重現。
- `causal_claim_allowed: false`；`clinical_promotion_allowed: false`。
