# 自適應模型一次性驗證 protocol v9（2026-10-27 凍結）

機器可驗證版本：`adaptive_model_lockbox_protocol_v9_20261027.json`（sha256 `ee9fe49e…`）。

**已被 v10 取代**（`ADAPTIVE_MODEL_LOCKBOX_PROTOCOL_V10_20261027.md`；v9 下從未認領任何步驟）。

- 雜湊值涵蓋 62 個程式／設定檔，以及 6 個 MIMIC-IV v5 訓練資料檔。
- 取代 v8：v8 從未在 SICdb／HiRID 上評分，而且 v8 的部署形式與結果都是在 2026-10-22 發現有洩漏的 replay 下產生的。

## 要回答的問題

在 leak-free replay 下重新選出的自適應部署形式，到完全沒用過的 ICU（SICdb、HiRID），不做任何在地校準研究，能否穩定通過？

## 凍結的部署形式（v9）

| 部分 | 內容 |
|---|---|
| 訓練 | 只用 MIMIC-IV v5 |
| 點預測 recipe | `deployment_recipes_20261024.json`（47 個 catalog 格 + 6 個 glucose_poc 格） |
| 點預測線上校正 | affine_0.005（2026-10-24 事後修正，已揭露） |
| 區間 | 五分位數值分組 + DtACI（meta rate 5，醫院與病房層級），θ₀ = 0 |
| 網路共享 | 只用於多醫院；SICdb、HiRID 都是單一醫院，不啟用 |
| replay | leak-free 部署時鐘：結果在實際量測時間才解出 |

這組設定就是 `site_adaptive_external_audit.py` 現在的預設值。凍結前已重跑 Zigong，與 `site_adaptive_amended_Zigong_20261024.json` 逐格完全相同（53 格，15 格穩定）。

## 範圍

- **評分**：連續值專家格。
- **不評**：
  - 二元事件格：洩漏修正後沒有在開發庫上重評；`multi_sample_external_audit.py` 會拒絕在 v9 下對 lockbox 評分。
  - 4 個 JEPA 格：仍是開發候選，要評需要約 45 小時的訓練。

## 讀取任何結果前必須完成

1. 每個資料庫的 extractor，符合 MIMIC-IV v5 合約：保留所有 anchor、記錄標籤偏移、只依資料字典對應變數、統一的 glucose 定義。
2. 用 `lockbox_addendum.py` 寫入一次性的 addendum 收據，放在 `causal_executions/<protocol sha256>/`，記錄 extractor、對應表與所有萃取輸出的雜湊值。
3. addendum 之前只能看支援數量與時間戳，不得看任何目標值。

## 評分（每個資料庫一次）

`site_adaptive_external_audit.py --database <db> --external-template <addendum 中的模板> --protocol adaptive_model_lockbox_protocol_v9_20261027.json --output site_adaptive_lockbox_<db>_v9.json`，其他選項一律用預設。

評分程式會依序檢查：
- 呼叫參數必須等於 protocol 記錄的部署形式；
- addendum 收據必須存在，而且裡面的檔案都沒改過；
- 所有雜湊值相符；

全部通過後才認領一次性步驟 `continuous_<db>`。

- 5 個 MIMIC 訓練抽樣，完整 gate ≥ 80% 抽樣通過才算穩定。gate 不變。
- **主要終點**：每個資料庫在 `point_and_dtaci_adaptive` 下穩定的連續格數。
- **判讀**：SICdb 與 HiRID 都穩定才算獨立重現；只有一個算候選。任何 registry 變更都需要使用者同意。

## 開發庫參考（in-sample，不是預測值）

eICU 40、NWICU 36、Zigong 15、Emory 41（`site_adaptive_amended_*_20261024.json`）。

## 凍結前的程式修改

- `site_adaptive_external_audit.py`：
  - 預設值改為 v9 部署形式；
  - lockbox 評分要求呼叫參數與 protocol 相同、要有 addendum 收據，並以一次性步驟執行。
- `multi_sample_external_audit.py`：lockbox 清單移除 Emory；不在 protocol 範圍內的二元格拒絕評分。
- 新增 `lockbox_addendum.py`。
- 測試：
  - 更新兩個早已過時的測試：Emory 已不是 lockbox；`run_variant` 的函式簽名已改。
  - 新增「呼叫參數不同或沒有 addendum 就拒絕評分」的測試。
  - 全套 624 個測試只有 `test_bun_forecast_demo` 無法匯入（venv 沒有 fastapi，是 8 月的 demo，與本步驟無關）。

`causal_claim_allowed: false`；`clinical_promotion_allowed: false`。
