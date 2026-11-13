# eICU 到 MIMIC-IV 格子修復結果

日期：2026-08-15

## 結論

- eICU 已驗證的 6h 格：`9`
- MIMIC 點預測通過：`8`
- MIMIC 點預測與 90% 區間皆通過：`6`
- 僅點預測通過：`2`
- 保持 fallback：`1`

| 狀態 | 格子 |
|---|---|
| 點預測與區間皆通過 | glucose@6h, heart_rate@6h, hematocrit@6h, potassium@6h, respiratory_rate@6h, urine_output@6h |
| 僅點預測通過 | hemoglobin@6h, map@6h |
| fallback | creatinine@6h |

## 已修復

- `glucose@6h`：只使用 anchor 前、unit-safe 的 insulin/dextrose/nutrition/steroid context，point、care-unit、time、placebo 與 interval 全通過。
- `hematocrit@6h`：移除無關治療欄位，只保留 fluid/albumin/diuretic/transfusion 後，完整通過。
- `heart_rate@6h` 與 `urine_output@6h`：點模型不變；使用 discovery-only domain-adaptive patient conformal 後，care-unit coverage 回到允許範圍。
- no-treatment audit 已硬性排除所有 `hist_*` 與 `act_*`；future treatment 不再可能洩漏。

## 尚未通過

- `creatinine`：6h、24h、48h 都未通過 patient-equalized gate。48h 雖改善逐列 MAE，但改善集中於重複量測較多的病人。
- `hemoglobin@6h`：care-unit coverage 仍為 overcoverage；需 target-specific heteroskedastic scale head，不可用 test labels 調 nominal。
- `MAP@6h`：care-unit coverage 仍為 undercoverage；同樣需 scale head 或更完整 anchor-time context。

> 這是 MIMIC 上重新訓練同 recipe 與 MIMIC-specific rescue candidate 的外部 replication，不是 frozen eICU 權重直接轉移。

所有結果維持 factual research-only；`causal_claim_allowed: false`。
