# 自適應模型一次性驗證 protocol v4（2026-10-20 凍結）

機器可驗證版本：`adaptive_model_lockbox_protocol_v4_20261020.json`（50 個程式／設定檔 + 6 個 MIMIC v4 訓練資料檔的 SHA-256）。`site_adaptive_external_audit.py` 與 `multi_sample_external_audit.py` 在 SICdb / HiRID 上必須帶 `--protocol`，任何雜湊不符即拒絕評分。取代 v3（v3 的 SICdb/HiRID 部分從未評分）。

## 要回答的問題

在完全沒用過的 ICU（SICdb、HiRID），不做任何在地校準研究，凍結的自適應部署形式能否穩定通過？

## 凍結的部署形式

| 部分 | 內容 | 來源 |
|---|---|---|
| 訓練 | 只用 MIMIC-IV v4 | |
| 點預測 recipe | 每格固定（MPR 40 格、hurdle 7 格），在 eICU/NWICU/Zigong 依預先規則選出 | `external_recipe_selection_dev_20261019.json` |
| persistence 退回 | platelets@1h：輸出目前值，不宣稱點預測，只過區間 gate | 同上 |
| 資料可得性規則 | TCR 在無治療紀錄的醫院退回 MPR | `osler_jepa/recipe_availability.py` |
| 點預測線上校正 | affine_0.005 | `site_adaptive_point_dev_20261016.json` |
| 第一天權重 | 前 min(200, 25%) 個住院的輸入（不看結果） | |
| 區間 | v2 五分位數值分組 + DtACI（meta rate 20，醫院與病房層級），新醫院起始倍數 θ₀ = 0.15 | `site_adaptive_dtaci_dev_20261017.json`、`site_adaptive_dtaci_prior_dev_20261019.json` |
| 二元事件格 | 維持 catalog recipe | `multi_sample_external_audit.py` |

## 讀取任何結果前必須完成

1. PhysioNet 權限與資料使用協議。
2. 每個資料庫的 extractor，符合 MIMIC-IV 合約（6h anchor、6h 回看、標籤 = anchor + h 前後 min(2, h/2) 小時內最近的量測、72h 上限、相同的合理值清理）。
3. 變數對應只依資料字典，沿用統一定義（glucose = 化學檢驗；POC → `glucose_poc`；血氣 → `glucose_bg`；尿量換算為 mL/h；MAP = 實測平均壓）。
4. 寫一份 addendum，記錄 extractor、對應表與萃取結果的雜湊，**之後**才打開標籤。addendum 之前只能看支援數量與時間戳，不得看任何目標值。

## 評分

- 每個資料庫只評一次；只能修正當機，不得改方法、gate 或族群，所有修正都要報告。
- 5 個 MIMIC 訓練抽樣；穩定 = 完整 gate ≥ 80% 抽樣通過。
- gate 不變：點預測 = 病人層級配對 bootstrap 上界 < 0（對 persistence 與該 recipe 的比較對象）；區間 = 病人平均與病房平均覆蓋率在 [0.87, 0.93]，每個病房同等權重。
- 主要終點：每個資料庫在 `point_and_dtaci_adaptive` 下穩定通過的連續格數。
- 判讀：兩個資料庫都穩定 = 獨立重現；只有一個 = 候選。任何 registry 變更都需要研究者同意。

## 開發庫參考（in-sample，不是預測值）

eICU 36/47、NWICU 37/44、Zigong 16/21。recipe 與 θ₀ 都在這些資料庫上選出；θ₀ 的留一資料庫估計為區間 85 → 93。

`causal_claim_allowed: false`；`clinical_promotion_allowed: false`。
