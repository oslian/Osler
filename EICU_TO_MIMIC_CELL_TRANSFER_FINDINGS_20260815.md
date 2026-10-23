# eICU 到 MIMIC-IV 格子外部驗證

日期：2026-08-15

## 結論

目前 runtime catalog 在 eICU 有 9 個 6h 格。在完整 MIMIC-IV 上重建同一 factual recipe 後，6 個點預測通過，其中 2 個連 90% 區間也通過。

> 這是方法在另一資料庫重新訓練與驗證的 replication；不是 frozen eICU 權重直接搬到 MIMIC。現有 registry 沒有保存所有來源模型的可推論權重。

## 6h Runtime 格

| 狀態 | 格子 |
|---|---|
| 點預測與區間皆通過 | potassium@6h, respiratory_rate@6h |
| 點預測通過、區間未通過 | heart_rate@6h, hemoglobin@6h, map@6h, urine_output@6h |
| 點預測未通過 | creatinine@6h, glucose@6h, hematocrit@6h |

## 未通過原因

- `creatinine@6h`：patient_discovery_selection_instability, patient_heldout_baseline_gate, patient_heldout_placebo_gate, careunit_holdout_value_gate, time_holdout_value_gate
- `glucose@6h`：careunit_holdout_value_gate
- `hematocrit@6h`：careunit_holdout_value_gate
- `heart_rate@6h`：點預測通過；區間失敗於 careunit_holdout_interval_gate
- `hemoglobin@6h`：點預測通過；區間失敗於 careunit_holdout_interval_gate
- `map@6h`：點預測通過；區間失敗於 careunit_holdout_interval_gate
- `urine_output@6h`：點預測通過；區間失敗於 careunit_holdout_interval_gate

## 判讀

- `creatinine@6h` 是明確的模型/時間尺度失敗：MIMIC 中 persistence 更好。
- `glucose@6h` 有平均增益，但 care-unit heldout 信賴區間跨 0，屬於跨病房不穩定。
- `hematocrit@6h` 勝過 persistence，卻未顯著勝過 capacity-matched placebo，表示增益未能歸因於有效生理特徵。
- heart rate、hemoglobin、MAP、urine output 的點預測可轉移，但區間 coverage 在 care-unit heldout 偏離 0.87-0.93，所以不可顯示正式窄區間。
- potassium 與 respiratory rate 是目前最完整的跨庫 6h 格：點預測與 90% 區間都成立。

所有結果仍是 factual research-only；`causal_claim_allowed: false`。
