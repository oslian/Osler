# 外部開發庫 recipe 選擇結果（2026-10-19）

工具：`external_recipe_selection_dev.py`；結果：`external_recipe_selection_dev_20261019.json`（`recipes` 可直接給 `site_adaptive_external_audit.py --recipes`）。
訓練：只用 MIMIC-IV v4，5 個抽樣 × 15,000 列（第 0 號為原始抽樣）。開發測試庫：eICU、NWICU、Zigong。
部署形式：`affine_0.005` 點預測校正 + day-one weighting + DtACI。

## 預先宣告規則

1. 某庫點預測穩定 = 點 gate（含該 recipe 的 comparator）≥ 80% 抽樣通過。
2. 選在最多可評估庫上點預測穩定的 recipe；平手選較簡單者（MPR → TCR → plain ridge → hurdle）。
3. 若選中 recipe 在可評估庫中未過半數穩定勝過 persistence，該格退回 `persistence`（輸出目前值，不做點校正，不宣稱點預測，只過區間 gate）。

## 結果

- 選中：MPR 40 格、hurdle 7 格（urine_output@1h、potassium@3h、creatinine@6h/12h/24h/48h、bun@24h）；退回 persistence 1 格（platelets@1h）。
- 改變的格：20 格，主要是 hurdle → MPR、TCR → MPR；potassium@3h 是 MPR → hurdle。
- plain ridge 在任何格、任何庫都未能穩定勝過 MPR，從未被選中。

| 測試庫 | 點預測穩定（原 catalog → 新選擇） | 完整 gate 穩定（原 → 新） |
|---|---|---|
| eICU（47 格） | 32 → 43 | 27 → 29 |
| NWICU（44 格） | 43 → 43 | 37 → 36 |
| Zigong（21 格） | 11 → 17 | 10 → 16 |
| 合計 | 86 → 103 | 74 → 81 |

**注意：選擇與評分用的是同一批資料庫，上表是樂觀估計（in-sample），不是驗證。** 真正的效果須在凍結後的 SICdb / HiRID lockbox 確認。

## 剩下的問題

- 點預測：Zigong 腎功能（creatinine@12h/24h、bun@24h）所有 recipe 都無法穩定勝過 persistence（hurdle 的 MAE 仍低於 persistence，0.975–0.991，但不顯著）；感染族群的腎功能軌跡和 ICU 訓練資料不同，換 recipe 解決不了。eICU hematocrit@3h（MPR 的 MAE 為 persistence 的 0.912 但醫院軸不顯著）、hemoglobin@3h、sodium@24h（60% 抽樣）。
- 完整 gate 的其餘失敗大多是區間：eICU glucose@1h/3h/6h/12h、o2sat@12h、map@48h、bicarbonate@12h、chloride@12h、creatinine@6h 等。
- 抽樣平均（bagging）：不穩定（40–60%）的格只有 3 個，依 AGENTS.md 第 2 條暫不加入。

## 第 4 步（未開始）

需要 SICdb 或 HiRID 的原始資料與 extractor，凍結 protocol（程式碼與 MIMIC 資料雜湊、選中的 recipes、gate、多抽樣規則）後才能一次性評分。目前電腦中尚無這兩個資料庫。

independent_validation: false；causal_claim_allowed: false。
