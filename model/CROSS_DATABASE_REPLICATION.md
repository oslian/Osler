# eICU 訓練到外部資料庫重現契約

## 目的

這份契約固定一個不可混淆的問題：

> 模型只在 eICU 訓練或選模，凍結所有權重後，能否在 MIMIC-IV 或新的資料庫重現？

目前答案是：41 個發行格中有 27 格在 MIMIC-IV 通過 patient、care-unit 與
forward-time 三道點預測閘門；14 格仍是 challenge candidate。MIMIC 標籤沒有
參與這 27 格的模型選擇，也沒有在 MIMIC 重新訓練權重。

舊報告中「在 MIMIC 重新訓練相同 recipe 後成功」不算 frozen-weight transfer，
因此不會寫進這份 registry 的通過數。

## 已重現的 27 格

| Horizon | 格子 |
|---:|---|
| 1h | `glucose`、`hypoxemia_event`、`map`、`urine_output` |
| 3h | `glucose`、`hypoxemia_event`、`map`、`urine_output` |
| 6h | `glucose`、`heart_rate`、`map`、`respiratory_rate` |
| 12h | `chloride`、`glucose`、`heart_rate`、`hematocrit`、`hemoglobin`、`map`、`o2sat`、`potassium`、`respiratory_rate` |
| 24h | `heart_rate`、`map`、`potassium`、`respiratory_rate`、`sodium` |
| 48h | `heart_rate` |

完整逐格 gate、方向準確率、delta correlation 與來源模型在
[`models/cross_database_replication_registry.json`](models/cross_database_replication_registry.json)。

## 尚未重現的 14 格

這些格仍隨模型一起保存，方便直接在新資料挑戰，但正式 runtime 只能 persistence：

`platelets@1h`、`heart_rate@3h`、`hematocrit@3h`、`hemoglobin@3h`、
`platelets@3h`、`respiratory_rate@3h`、`creatinine@6h`、
`hematocrit@6h`、`hemoglobin@6h`、`bicarbonate@12h`、
`creatinine@12h`、`bicarbonate@24h`、`bun@24h`、`glucose@48h`。

它們不能因為在新資料的單一 MAE 較好就啟用；仍須完整通過三道 grouped gate。

## 新資料怎麼接

### 1. 先轉成 canonical 生理資料

使用 `extractors/canonical_event_ledger_extract.py` 與
`extractors/canonical_event_examples.py`，保留實際 measurement time、patient、
site/care-unit、anchor 前治療與 future 實測值。單位必須符合
`models/FEATURE_CONTRACT.json` 和 README 的 53 變數表。

### 2. 使用凍結模型產生預測

推論時必須使用目前 release 的：

- `whole_body_joint_jepa.pt`
- `unified_shared_jepa_head_bundle.joblib`
- `event_risk_heads.joblib`
- `runtime_registry.json`

不得在新 lockbox 調權重、shrinkage、abstention threshold 或 subtype。將每筆
評估輸出整理成 CSV 或 Parquet：

| 欄位 | 意義 |
|---|---|
| `patient_id` | 病人 ID；只用於 grouped evaluation，不寫入公開 artifact |
| `cell` | 例如 `glucose@6h` |
| `truth` | future 實測值或事件標籤 |
| `candidate` | 凍結 eICU 模型輸出 |
| `persistence` | anchor 時該 target 的實測值；事件格填 persistence probability |
| `domain_id` | 新資料的 hospital、care unit 或 site |
| `time_block` | 依時間先後固定切出的 block |
| `current` | 選填；用來計算 direction accuracy 與 delta correlation |
| `lower` / `upper` | 選填；相容 calibrator 的區間 |
| `task_type` | 選填；事件格填 `binary_event_probability` |

### 3. 跑 frozen lockbox gate

```bash
whole-body-lockbox \
  --input outputs/new_database_frozen_predictions.parquet \
  --output outputs/new_database_lockbox_report.json
```

或未安裝 entry point 時：

```bash
python -m scripts.validate_external_lockbox \
  --input outputs/new_database_frozen_predictions.parquet \
  --output outputs/new_database_lockbox_report.json
```

validator 對連續值使用 MAE，對事件機率使用 Brier score，並分別以 patient、
domain、forward-time group 做 paired bootstrap。任何一個 grouped gate 缺失或
95% CI 未完全低於 0，該格就不算重現。

## 找到新資料後如何「調整」

新資料第一次只能作 frozen validation，不能一邊看結果一邊調模型再稱為外部重現。
若確實需要適配：

1. 將新資料明確改列為 development database。
2. 只在 development split 調 target head、校準器或 domain normalization。
3. 凍結新版本。
4. 使用另一個從未看過的資料庫或時間段作 lockbox。

這樣能直接迭代，又不會把「對新資料過擬合」誤認為人體生理的泛化能力。

所有結果只屬 factual observation/forecast；`causal_claim_allowed: false`。
