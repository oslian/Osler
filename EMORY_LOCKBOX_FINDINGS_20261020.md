# Emory 一次性驗證結果（2026-10-20）

protocol：`adaptive_model_lockbox_protocol_v5_20261020.json`（評分時雜湊驗證通過）；萃取 addendum：`emory_lockbox_addendum_20261020.json`。
資料：PhysioNet/CinC Challenge 2019 training set B = Emory University Hospital ICU（Challenge 論文指明），19,530 個 ICU 住院。
結果：`site_adaptive_lockbox_Emory_20261020.json`（連續格）、`multi_sample_lockbox_Emory_binary_20261020.json`（hypoxemia）。每格評分一次，未修改任何東西。

**資料身分**：目前的部署形式（recipe、θ₀、區間、gate）從未使用這份資料；但專案較早的另一條模型線（2026-06 的 ICU JEPA 預訓練、DKABody 校準）用過 A+B 兩組並看過其 persistence 比較。依研究者 2026-10-20 決定，計為獨立驗證。

## 主要終點

| 設定 | 穩定通過（連續格） |
|---|---|
| 凍結的自適應形式（點校正 + DtACI，θ₀ = 0.15） | **40 / 41** |
| 只有區間自適應 | 34 / 41 |
| 靜態 v2（無自適應） | 22 / 41 |

點預測穩定通過 39 / 40（platelets@1h 為 persistence 退回格，不宣稱點預測）。hypoxemia_event@1h、@3h：2 / 2 穩定通過。

- 唯一失敗：chloride@12h（467 列；點預測 MAE 為 persistence 的 1.032，區間通過）。
- 不可評估（無來源欄位或支援不足）：sodium@24h、urine_output@1h/3h/6h、bicarbonate@12h/24h。
- 點預測 MAE / persistence：0.75（potassium@48h）至 0.98（platelets@3h、creatinine@6h）。
- 區間覆蓋：病人平均 0.887–0.915，病房平均 0.886–0.910，全部在 [0.87, 0.93] 內。

## 解讀時必須一併說明

1. 這個站點比開發庫「容易」：只有 3 個病房群組（MICU、SICU、unknown），沒有讓 NWICU 失敗的極小病房；多數格有數千到 6 萬列，DtACI 有足夠的結果可以調整。
2. glucose 對數很高（1h 有 21,699 對），Challenge 的「serum glucose」可能混入床邊血糖，與 MIMIC 的化學檢驗定義不完全相同。
3. 腎功能在這裡通過（creatinine 6/12/24/48h、bun@24h），與 Zigong 的失敗不同；差別可能在族群（一般 ICU vs 感染族群）與資料密度。
4. 依 protocol 決策規則：單一資料庫穩定 = 候選；需 SICdb / HiRID 至少再一個也穩定，才算獨立重現。任何 registry 變更需研究者同意。

`causal_claim_allowed: false`；`clinical_promotion_allowed: false`。

## 更正（2026-10-20）：Emory glucose 的定義

依支援數量（未讀任何數值），Emory 的 Challenge「serum glucose」幾乎可以確定以床邊血糖為主：每列有 glucose 對的比例為 21%，高於 MIMIC 化學檢驗（2%）甚至 MIMIC 床邊血糖（11%）。因此 Emory 6 個 glucose 格的通過，是「以化學檢驗血糖訓練的模型，在以床邊血糖為主的混合標籤上」通過，定義與 MIMIC 的 glucose 目標不同。這 6 格不能作為化學檢驗血糖的獨立重現證據，應另列為「定義不同」。
