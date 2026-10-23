# LODO 多來源訓練結果（2026-10-19）

工具：`lodo_multisource_audit.py`（未採用，已刪除）；結果：`lodo_multisource_audit_20261019.json`。
開發資料庫：MIMIC-IV v4、eICU harmonized、NWICU v4、Zigong（僅檢驗）。皆為已使用過的資料，非獨立驗證。
部署形式：點預測線上校正 `affine_0.005` + day-one weighting + DtACI；5 個訓練抽樣，穩定通過 = 完整 gate ≥ 80%。

## 預先宣告規則與結果

規則：可比較的測試庫（eICU、NWICU、Zigong）穩定通過數加總，多來源訓練必須嚴格多於只用 MIMIC 才採用。

| 測試庫 | 多來源（其他 3 庫合併） | 只用 MIMIC | 可評估格數 |
|---|---|---|---|
| eICU | 24 | 25 | 47 |
| NWICU | 34 | 38 | 44 |
| Zigong | 8 | 11 | 21 |
| **合計** | **66** | **74** | 112 |
| MIMIC-IV（只有多來源） | 31 | — | 47 |

**結論：多來源訓練不採用。**

## 解讀

- 多來源訓練使平均 MAE 略降（相對 persistence 的 MAE 中位數：eICU 0.869 vs 0.875、NWICU 0.887 vs 0.889、Zigong 0.897 vs 0.903；多來源較佳的格數 30/47、32/44、14/21），但幅度很小。
- gate 通過率反而下降，主因是勝過 persistence 的穩定度變差：Zigong 0.83 vs 0.91、eICU 0.94 vs 0.96。區間通過率也略降。
- 依 recipe：MPR 48 vs 51；hurdle 18 vs 23；TCR 0 vs 0。hurdle 是主要失敗來源，與訓練來源無關。
- 不用 MIMIC 訓練、在 MIMIC 上測試仍有 31/47 穩定通過，代表資料庫間差異不是點預測的主要瓶頸；剩下的失敗集中在特定格與 recipe。
- 與 Rockenschaub et al. (arXiv 2303.15354) 一致：合併訓練能縮小平均的外部損失，但不保證個別格的可靠度。

## 重複出現的失敗（只用 MIMIC，跨測試庫）

- glucose@1h/3h/12h、chloride@12h、creatinine@6h：在 2–3 個測試庫都失敗。
- eICU 系統性失敗：platelets/hemoglobin/hematocrit/potassium@3h、o2sat@12h、TCR 格（map@1h/48h、hematocrit@6h、glucose@3h/6h/12h）。
- Zigong hurdle：creatinine@6h/12h/24h、sodium@24h、chloride@12h、hematocrit@12h、bicarbonate@12h/24h、bun@24h。

independent_validation: false；causal_claim_allowed: false。
