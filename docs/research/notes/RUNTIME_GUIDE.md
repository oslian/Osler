# Whole-Body JEPA 全身生理預測模型

這是一個**研究用途、非因果、非臨床決策**的 ICU 生理預測發行包。它包含完整模型權重、共享 target-head 權重、病人 CSV 填值與推論程式、MIMIC-IV/eICU 資料抽取器、外部驗證 registry，以及可直接執行的合成範例。

> 重要：此 private repository 含由 credentialed MIMIC-IV/eICU 資料衍生的模型權重。不得公開 repository 或把權重轉交未獲授權的人。程式碼可以另行公開；衍生權重若要分享，應依 PhysioNet 規則在相同資料使用協議下發布。

## 1. 這個版本是什麼

統一來源版本：`2026-09-02`

目前使用同一個 `predict_patient` 入口與同一個 JEPA checkpoint，逐格選擇
`frozen_whole_body_jepa`、`unified_shared_jepa_target_head` 或
`multisource_shared_jepa_target_head`。不是混合三套權重，也不是把通過格數相加。

`models/unified_source_selector.json` 記錄每格選中的來源、是否取代舊來源、
新驗證結果及配對區間。選源只使用 MIMIC 開發病人，校準及最終測試病人分開；
再以 NWICU 檢查選定來源。新來源未完整勝過現役來源時保留原來源及原證據範圍，
不會根據 NWICU 成績改挑第二名。MIMIC/NWICU 已被研究使用，不能宣稱新的未接觸 lockbox。
完整逐格比較見 [UNIFIED_SOURCE_SELECTION_FINDINGS_20260902.md](../../../UNIFIED_SOURCE_SELECTION_FINDINGS_20260902.md)。

統一後：11 格直接 JEPA、11 格原 shared head、4 格 multi-source head、
2 格事件風險，共 28 格可移動；13 格仍 persistence。
此次替換 `heart_rate@6h`、`heart_rate@48h`、`respiratory_rate@6h`；
NWICU 病人等權 MAE 比原來源分別改善約 1.1%、3.6%、3.3%。
`o2sat@12h` 保留原來源並通過新校準。後續以固定條件式區間救回
`bicarbonate@24h`：點預測不變，只用開發病人擬合誤差尺度，再以獨立病人校準。
這一格從 persistence 改用 multi-source head，MIMIC/NWICU 覆蓋率約 89.3%/89.4%。
見 [PERSISTENCE_INTERVAL_RESCUE_FINDINGS_20260902.md](../../../PERSISTENCE_INTERVAL_RESCUE_FINDINGS_20260902.md)。
合計 5 格完整通過這兩輪新對照，
不能把其餘保留來源解讀成已通過本輪 NWICU 檢查。

輸出新增 `source_changed`、`unified_source_audit_pass` 與 `source_validation_scope`。
新來源只使用其專屬校準區間；保留來源使用原校準。缺少 selector 或權重雜湊不符時
直接拒絕載入。下面的數字是 **2026-08-25 原始基線**，不是新來源通過數：

- 一個 frozen whole-body JEPA checkpoint。
- 53 個生理變數、11 個功能模組、64 維 shared context latent、96 維 patient belief、32 維 organ token。
- 41 個 `target@horizon` 預測格。
- 27 格通過 eICU 訓練/選擇後的 MIMIC-IV 外部點預測閘門。
- 14 格未過外部閘門，因此保留病人自己的 persistence anchor。
- 27 個可移動格中，16 格直接使用 frozen JEPA，11 格使用同一個 311 維病人 representation 上的 shared target head。
- 25 個連續可移動格重新以 MIMIC-IV conformal 校準；24 格同時通過病人、care-unit 與 forward-time 的 90% coverage 閘門並輸出正式區間。
- 其餘 1 個連續格不輸出假區間；2 個 hypoxemia 格改由獨立校準事件風險 head 輸出 0–1 機率，不再把連續 O2Sat 預測硬切成 0/1。
- `causal_claim_allowed: false`，不能回答「換治療會怎樣」。

它不是「每一格都硬猜」的模型。某格未通過驗證，或該病人根本沒有該 target 的歷史值時，runtime 會回傳 persistence 或 missing，不會捏造看似精確的數字。

### 跨資料庫重現層

這 27 格不是在 MIMIC 重新訓練後得到的數字，而是 eICU 選模/訓練、權重凍結後
直接通過 MIMIC patient、care-unit、forward-time gate。完整可攜 registry、
14 格 challenge candidate 與新資料驗證方式見
[CROSS_DATABASE_REPLICATION.md](../../../CROSS_DATABASE_REPLICATION.md)。找到新資料後可直接
輸出 frozen scored table 並執行 `whole-body-lockbox`；validator 不會在 lockbox
上重新 fit 或調 threshold。

### 生理亞型少樣本適配

Repository 另包含 candidate-only 的軟生理亞型層。它只用 anchor 前的
`shared context + patient belief + measurement-pure state`，把病人表示成多個
亞型的機率，再以階層式收縮讓少量個人資料借用相似族群的 residual prior。
個人已解出的預測誤差增加後，權重會逐步轉回該病人自己的軌跡。

2026-08-24 的正式 eICU→MIMIC-IV 7-seed 外部測試為 0/39 格完整通過，
因此這個模組不會改動正式 41 格。`MAP@12h` 與
`respiratory_rate@6h` 出現可信但未達完整門檻的 near-miss；它們仍保持
candidate-only。完整契約、逐格結果與 API 見
[PHYSIOLOGY_SUBTYPE_LAYER.md](PHYSIOLOGY_SUBTYPE_LAYER.md)。

後續只用 eICU 做了病人軌跡、三模型共識、target-specific 時窗與
hospital-balanced residual prior 實驗；三格 near-miss 仍為 0 格完整通過。
這些結果進一步確認訊號會隨病人群、醫院與時間改變，不是單純增加歷史平滑
或重新加權即可解決。MIMIC-IV lockbox 沒有被拿來反覆調方法，正式 runtime
維持不變。

另外也完成了 stable patient background 消融：年齡、性別、BMI、anchor 前
共病、歷史 level/slope/變異度，以及分離式 multi-view embedding 和
target-specific ridge 都沒有通過完整 eICU patient/hospital/time gate。
`MAP@6h` 有局部背景訊號，但 unseen-hospital 仍為 `0/3`，因此沒有接進
runtime。

最後的 eICU-only 消融再加入 anchor 前 `hist_*` 治療歷史、最壞醫院約束
stacking 與三模型方向共識閘門。`MAP@6h` 仍只過 5/7 patient、0/3
unseen-hospital、1/1 forward-time；最壞醫院選擇器在所有格都選擇零修正。
這表示治療歷史含局部訊號，但目前的 pre-anchor covariates 仍無法形成跨醫院
穩定的亞型修正。`act_*` 未來治療欄位會被驗證器直接拒絕，runtime 不變。

後續找到並修正一個真實的 cohort-selection 問題：舊 loader 依疾病名稱選來源，
使 MAP 只使用 12 家醫院；現在改成按嚴格 future-label 後的醫院深度與病人數
選 canonical source，並在相同 row budget 下優先覆蓋更多病人。修正後
`MAP@12h` 提升到 patient 5/7、unseen-hospital 2/3，respiratory rate 達
hospital 3/3，但都未通過完整 patient/hospital/time gate。跨醫院 OOF
selective gate、ensemble consensus、robust stacking 與因果線上個人 residual
update 均未補齊全部門檻，因此仍不進 runtime。

進一步稽核發現歷史摘要原本在抽樣後才計算，且曾以 patient ID 合併不同 ICU
stay 的相對時間。現已改成先用完整 stay 建立 anchor 前歷史，再以 patient 做
資料切分。`MAP@12h` 的 forward-time 因此通過 1/1，但 patient 只過 2/7、
hospital 1/3；完整 event-state 與 direct target ridge 也沒有補齊外部閘門。
資料管線錯誤已修，未觀測的後續急性事件仍不可被偽造成可識別 residual。

## 2. Repository 結構

```text
models/
  whole_body_joint_jepa.pt                 frozen JEPA checkpoint
  unified_shared_jepa_head_bundle.joblib   shared target heads
  multisource_shared_jepa_head_bundle.joblib multi-source target heads
  unified_source_selector.json             現行逐格選源與配對區間
  runtime_registry.json                    原始 41 格驗證證據，供保留來源使用
  external_validation_report.json          外部驗證細節
  cross_database_replication_registry.json  eICU -> MIMIC 凍結權重重現契約
  interval_calibrator.json                 24 格 release-bound 90% 區間
  FEATURE_CONTRACT.json                    311 維 head 輸入契約
  MODEL_MANIFEST.json                      SHA256 與檔案大小
whole_body_jepa/
  runtime.py                               CSV 解析、填值、完整推論
  contracts.py                             欄位別名與單位
osler_jepa/
  shared_release_features.py               311 維向量唯一合法組裝器
  event_ledger.py                          時間戳 event ledger
  event_dataset.py                         target-specific future labels
  event_adapters.py                        MIMIC-IV/eICU raw adapters
  urine_output.py                          尿量 interval-aware 正規化
extractors/
  canonical_event_ledger_extract.py        raw tables -> canonical events
  canonical_event_examples.py              events -> JEPA examples
  mimiciv_observation_transition_extract.py
  eicu_body_system_transition_extract.py
scripts/
  predict_patient.py                       單病人推論
  show_contract.py                         列出所有輸入與格子
  validate_release.py                      checksum + load + E2E smoke test
  validate_external_lockbox.py             新資料 frozen 跨庫重現 gate
examples/
  patient_1.csv                            synthetic wide CSV
  patient_2.csv                            synthetic wide CSV，含合法 urine=0
  patient_3_long.csv                       synthetic long-form CSV
```

Repository 不含任何 row-level MIMIC-IV/eICU 病人資料。

## 3. 安裝

建議 Python 3.11 或 3.12。

```bash
git clone https://github.com/Chun0613-code/WholeBody-JEPA.git
cd WholeBody-JEPA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
whole-body-validate
```

成功時最後會看到：

```json
{
  "status": "ok",
  "catalog_cells": 41,
  "runtime_move_cells": 27,
  "fallback_cells": 14,
  "missing_cells": 0
}
```

## 4. 最快推論方式

```bash
whole-body-predict \
  --input examples/patient_1.csv \
  --output outputs/patient_1_forecast.json \
  --model-dir models
```

也可直接使用 Python：

```python
from pathlib import Path
from whole_body_jepa import predict_patient

result = predict_patient(
    Path("examples/patient_1.csv").read_text(),
    filename="patient_1.csv",
    model_dir="models",
)

for cell in result["cells"]:
    print(cell["cell"], cell["forecast_point"], cell["source_model"])
```

缺失的當下值與未來預測是兩個不同物件。若 bicarbonate 或 heart rate
從未測量，但至少有四個相關生理輸入，輸出會在 `current_estimate` 顯示經
eICU 訓練、MIMIC-IV 閘門驗證的交叉推估與 90% 區間。原始觀測永遠
優先，未通過的 target 保持 missing，交叉推估也不會自動反餵 future
forecast。

## 5. CSV 要怎麼填

### 5.1 Wide format，建議使用

一列代表一個時間點；欄位代表生理變數。

```csv
hours_since_onset,heart_rate,map,o2sat,glucose,creatinine,urine_output
0,108,68,93,245,1.4,42
3,102,72,94,221,,0
6,96,76,95,198,1.3,68
```

規則：

1. 多列資料必須有時間欄，最推薦 `hours_since_onset`。
2. 可以改用 `timestamp`、`charttime`、`time` 等可解析時間戳。
3. 空白或欄位完全不存在才代表 missing。
4. `0` 是有效實測值，絕不代表 missing。尿量 `0 mL/hour` 會被保留。
5. 如果同時提供 `sbp` 與 `dbp`、但沒有 `map`，runtime 會計算 `(SBP + 2*DBP)/3`。
6. 每一列不必填滿 53 個欄位。最近一次有效值最多向前帶 6 小時；超過 6 小時視為過期。
7. 模型內部只對真正缺失值使用 checkpoint training median；不會把使用者未填的值改成物理上的 0。
8. 要讓某個格真正輸出 personalized forecast，該 target 本身必須至少有一次歷史實測。沒有 target 歷史時，該格回傳 missing。

### 5.2 Long format

```csv
hours_since_onset,variable,value
0,heart_rate,108
0,map,68
3,heart_rate,102
3,map,72
```

`variable` 可使用 canonical name 或常見別名，例如 `hr`、`spo2`、`hco3`、`scr`、`hgb`、`hct`、`plt`。

## 6. 53 個輸入欄位與單位

所有數值都必須先轉成下表單位。不要把 mmol/L glucose 直接填入 mg/dL 欄位，也不要把 Fahrenheit 填入 Celsius 欄位。

| 欄位 | 應填內容 | 單位 |
|---|---|---|
| albumin | 白蛋白 | g/dL |
| alkaline_phos | 鹼性磷酸酶 | U/L |
| alt | ALT | U/L |
| amylase | 澱粉酶 | U/L |
| anion_gap | 陰離子間隙 | mEq/L |
| ast | AST | U/L |
| bicarbonate | HCO3 / bicarbonate | mEq/L |
| bilirubin | 總膽紅素 | mg/dL |
| bilirubin_direct | 直接膽紅素 | mg/dL |
| bnp | BNP | pg/mL |
| bun | 血尿素氮 | mg/dL |
| calcium | 總鈣 | mg/dL |
| chloride | 氯 | mEq/L |
| ck_mb | CK-MB | ng/mL |
| cortisol | 皮質醇 | ug/dL |
| cpk | CK/CPK | U/L |
| creatinine | 肌酸酐 | mg/dL |
| crp | CRP | mg/L |
| crp_hs | 高敏 CRP | mg/L |
| esr | 紅血球沉降率 | mm/hour |
| ferritin | Ferritin | ng/mL |
| fibrinogen | Fibrinogen | mg/dL |
| free_t4 | Free T4 | ng/dL |
| glucose | 血糖 | mg/dL |
| heart_rate | 心率 | beats/min |
| hematocrit | Hct | % |
| hemoglobin | Hgb | g/dL |
| inr | INR | ratio |
| ionized_calcium | 游離鈣 | mmol/L |
| lactate | 乳酸 | mmol/L |
| ldh | LDH | U/L |
| lipase | Lipase | U/L |
| magnesium | 鎂 | mg/dL |
| map | 平均動脈壓 | mmHg |
| myoglobin | Myoglobin | ng/mL |
| o2sat | 血氧飽和度 | % |
| ph | 血液 pH | pH |
| phosphate | 磷酸鹽 | mg/dL |
| platelets | 血小板 | K/uL |
| potassium | 鉀 | mEq/L |
| prealbumin | 前白蛋白 | mg/dL |
| ptt | PTT | seconds |
| respiratory_rate | 呼吸率 | breaths/min |
| serum_osmolality | 血清滲透壓 | mOsm/kg |
| sodium | 鈉 | mEq/L |
| temperature | 體溫 | degC |
| total_protein | 總蛋白 | g/dL |
| triglycerides | 三酸甘油脂 | mg/dL |
| troponin_i | Troponin I | ng/mL |
| tsh | TSH | uIU/mL |
| uric_acid | 尿酸 | mg/dL |
| urine_output | 前一實際收集區間的尿量率 | mL/hour |
| wbc | 白血球 | K/uL |

`urine_output` 特別重要：raw output event 通常是某段不規則時間內的體積，不是瞬間 rate。請先以 `volume_ml / collection_interval_hours` 轉成 `mL/hour`。`osler_jepa/urine_output.py` 已提供同一份驗證過的轉換。

## 7. 41 個預測格

`MODEL` 代表通過 eICU -> MIMIC-IV 外部點預測閘門；`PERSISTENCE` 代表該格目前只回傳病人最近的實測 target。

| 格子 | 狀態 | Runtime source |
|---|---|---|
| glucose@1h | MODEL | frozen whole-body JEPA |
| hypoxemia_event@1h | MODEL | calibrated event-risk head |
| map@1h | MODEL | frozen whole-body JEPA |
| urine_output@1h | MODEL | shared target head |
| platelets@1h | PERSISTENCE | patient anchor |
| glucose@3h | MODEL | shared target head |
| hypoxemia_event@3h | MODEL | calibrated event-risk head |
| map@3h | MODEL | frozen whole-body JEPA |
| urine_output@3h | MODEL | shared target head |
| heart_rate@3h | PERSISTENCE | patient anchor |
| hematocrit@3h | PERSISTENCE | patient anchor |
| hemoglobin@3h | PERSISTENCE | patient anchor |
| platelets@3h | PERSISTENCE | patient anchor |
| respiratory_rate@3h | PERSISTENCE | patient anchor |
| glucose@6h | MODEL | shared target head |
| heart_rate@6h | MODEL | frozen whole-body JEPA |
| map@6h | MODEL | frozen whole-body JEPA |
| respiratory_rate@6h | MODEL | frozen whole-body JEPA |
| creatinine@6h | PERSISTENCE | patient anchor |
| hematocrit@6h | PERSISTENCE | patient anchor |
| hemoglobin@6h | PERSISTENCE | patient anchor |
| bicarbonate@12h | PERSISTENCE | patient anchor |
| chloride@12h | MODEL | shared target head |
| creatinine@12h | PERSISTENCE | patient anchor |
| glucose@12h | MODEL | shared target head |
| heart_rate@12h | MODEL | frozen whole-body JEPA |
| hematocrit@12h | MODEL | shared target head |
| hemoglobin@12h | MODEL | shared target head |
| map@12h | MODEL | frozen whole-body JEPA |
| o2sat@12h | MODEL | frozen whole-body JEPA |
| potassium@12h | MODEL | frozen whole-body JEPA |
| respiratory_rate@12h | MODEL | frozen whole-body JEPA |
| bicarbonate@24h | PERSISTENCE | patient anchor |
| bun@24h | PERSISTENCE | patient anchor |
| heart_rate@24h | MODEL | shared target head |
| map@24h | MODEL | shared target head |
| potassium@24h | MODEL | frozen whole-body JEPA |
| respiratory_rate@24h | MODEL | frozen whole-body JEPA |
| sodium@24h | MODEL | shared target head |
| glucose@48h | PERSISTENCE | patient anchor |
| heart_rate@48h | MODEL | frozen whole-body JEPA |

實際權威來源永遠是 `models/runtime_registry.json`，不是 README 的手寫表格。可執行 `whole-body-contract` 即時列出目前 artifact 的狀態。

## 8. 311 維 shared-head 輸入是什麼

不要手動拼這個向量。請只呼叫 `osler_jepa.shared_release_features.build_target_head_features`。

| Index | 寬度 | 內容 |
|---|---:|---|
| 0:64 | 64 | shared whole-body context latent |
| 64:160 | 96 | patient belief state |
| 160:224 | 64 | target-specific future latent |
| 224:256 | 32 | target organ token |
| 256:257 | 1 | frozen JEPA 對該 target/horizon 的 normalized base forecast |
| 257:310 | 53 | 依 checkpoint 順序排列的 normalized measurement state |
| 310:311 | 1 | 該 target 的 normalized current value |

`base_point` 絕對不是 current value。把 current value 填到 index 256 是一個不會產生 shape error、但會讓輸出整體偏移的危險錯誤。

## 9. 從 MIMIC-IV/eICU 原始資料抽取

資料必須由使用者自行取得並遵守各資料庫 DUA。不要把抽出的 parquet 或 CSV commit 到 GitHub。

### 9.1 建 canonical event ledger

```bash
python -m extractors.canonical_event_ledger_extract \
  --database mimiciv \
  --data-root /path/to/mimiciv/3.1 \
  --output-dir data/mimiciv_events \
  --report outputs/mimiciv_event_report.json \
  --max-stays 1000
```

```bash
python -m extractors.canonical_event_ledger_extract \
  --database eicu \
  --data-root /path/to/eicu-crd/2.0 \
  --output-dir data/eicu_events \
  --report outputs/eicu_event_report.json \
  --max-stays 1000
```

MIMIC 根目錄必須包含標準 `hosp/` 與 `icu/` 子目錄；eICU 根目錄必須包含 `patient.csv.gz`、`lab.csv.gz`、`vitalPeriodic.csv.gz`、`intakeOutput.csv.gz` 等官方表。

### 9.2 Event ledger 轉成 target-specific examples

```bash
python -m extractors.canonical_event_examples \
  --event-data data/mimiciv_events \
  --output data/mimiciv_examples.parquet \
  --report outputs/mimiciv_examples_report.json \
  --variables heart_rate,map,o2sat,respiratory_rate,glucose,sodium,potassium,bicarbonate,creatinine,bun,urine_output \
  --horizons 1,3,6,12,24,48
```

這條流程只使用 anchor 時間以前的 measurements/treatments 當輸入；future label 必須是真正 anchor 後的測量事件，並保留每個 target 的實際 `delta_t`。

## 10. 填值規則

| 情況 | Runtime 行為 |
|---|---|
| 實測為 0 | 保留為 observed zero |
| 空白、NaN、無欄位 | 視為 missing |
| 6 小時內舊值 | causal forward-fill |
| 超過 6 小時舊值 | 視為過期，不再 forward-fill |
| SBP+DBP 有值、MAP 空白 | 以公式計算 MAP |
| 模型內部缺失 state | 使用 checkpoint median，即 normalized 0 |
| 可移動格但 target 從未測過 | missing，不輸出模型值 |
| 未通過外部閘門且 target 有值 | persistence |
| 未通過外部閘門且 target 無值 | missing |

## 11. 輸出 JSON

每格包含：

```json
{
  "cell": "glucose@6h",
  "target": "glucose",
  "unit": "mg/dL",
  "horizon_hours": 6,
  "observed_value": 176.0,
  "measurement_age_hours": 0.0,
  "observations": 4,
  "forecast_point": 154.2,
  "source_model": "unified_shared_jepa_target_head",
  "can_move": true,
  "fallback_reason": null,
  "interval_status": "externally_audited_mimic_split_conformal_90",
  "lower": 84.3,
  "upper": 224.1,
  "causal_claim_allowed": false
}
```

區間 calibrator 以實際 release 點模型的 MIMIC-IV residual 建立，並綁定 checkpoint、target-head bundle 與 runtime registry 的 SHA256。只有病人、care-unit、forward-time coverage 都在 `0.87–0.93` 的 24 格會顯示 `lower/upper`。其中 `glucose@3h`、`hematocrit@12h`、`hemoglobin@12h`、`potassium@12h` 使用 patient-grouped normalized conformal，寬度會依 anchor 前病人狀態調整；其餘使用固定半徑。區間會裁切到基本生理支撐範圍，例如尿量不低於 0、O2Sat 不超過 100。未通過的格子維持 `null`。

`hypoxemia_event@1h/3h` 是分類風險，不是連續值區間。它們輸出
`event_risk_probability`、`event_definition: future_o2sat <= 92` 與
`risk_status: calibrated_event_probability`；`lower/upper` 固定為 `null`。
兩個 head 都通過 7/7 patient split、grouped hospital、care-unit 與
forward-time gate。若 O2Sat 缺失或超過 release 允許的測量年齡，runtime 會
`abstain`，不輸出 0 或 1 假裝成風險。

通過區間閘門的格子是：`glucose@1h/3h/6h/12h`、`MAP@1h/3h/6h/12h/24h`、`urine_output@1h/3h`、`heart_rate@6h/12h/24h/48h`、`respiratory_rate@6h/12h/24h`、`O2Sat@12h`、`hematocrit@12h`、`hemoglobin@12h`、`potassium@12h/24h`、`sodium@24h`。只有 `chloride@12h` 未通過全部 coverage 閘門，因此仍不顯示區間。

MIMIC-IV 已經用於點模型外部驗證，因此這次是 **release-matched interval calibration 與內部 heldout audit**，不是一個新的獨立資料庫驗證。若要對外宣稱區間能跨資料庫泛化，仍需 HiRID、AmsterdamUMCdb 或其他未接觸 lockbox。

## 12. 驗證與完整性

```bash
whole-body-validate
python -m unittest discover -s tests -p 'test*.py'
```

`whole-body-validate` 會：

1. 驗證 manifest 內六個主要 artifact 的 SHA256。
2. strict-load checkpoint state dict。
3. 驗證所有 target head 的 feature width 都是 311。
4. 跑一個完整 synthetic patient CSV。
5. 確認輸出正好 41 格。

### VitalDB 獨立波形 lockbox

跨國家、跨照護場景的凍結評估已完成。VitalDB lockbox 包含 318 位病人、
2,505 個 anchor 與 218 個事件；raw waveform JEPA 的 AUC 為 0.620，但
patient-level paired Brier 尚未顯著勝過病人波形打亂的 placebo。因此此模型
仍是 `candidate_only`，不接入 runtime。完整方法、結果與重現命令見
[VITALDB_WAVEFORM_LOCKBOX_FINDINGS_20260827.md](../findings/VITALDB_WAVEFORM_LOCKBOX_FINDINGS_20260827.md)。

另外已固定比較機率校準、future-SpO2 多任務、高解析 morphology、低秩
PCA/SVD、修正事件標籤、5/10/15/30 分鐘事件 head 與 deep ensemble。
所有候選都未通過完整 promotion gate，因此沒有接入 runtime。完整結果見
[WAVEFORM_PROMOTION_METHODS_FINDINGS_20260827.md](../findings/WAVEFORM_PROMOTION_METHODS_FINDINGS_20260827.md)。

## 13. 已知邊界

- 僅 factual observation/forecast，不是 causal treatment model。
- 外部驗證代表在既定 eICU/MIMIC cohort 與 gate 下通過，不代表前瞻性臨床驗證。
- Hormone、troponin、INR、腫瘤標記等稀疏變數目前多數只能當輸入觀測，沒有通過固定 horizon future forecast。
- 28 格的可移動資格是逐格的，不代表整個 checkpoint 的 53 個輸出都可直接使用，也不代表遮掉當下值仍可還原。
- 醫療單位、量測方式、care unit 與資料庫差異仍可能造成 domain shift。
- 如果輸入單位錯誤、把缺失填成 0、或手動拼錯 311 維向量，程式可能仍能執行但結果沒有意義。

### 現役 checkpoint 遮蔽檢查（2026-09-03）

已盤點全部 41 格，直接測封裝權重，而非引用其他重建候選的結果。遮掉
target、既定兄弟變數及完整輸入歷史，重新計算 latent 與讀出頭；沒有
把隱藏真值留在 `base_point`、重複 target 欄或 residual anchor。

- 當下還原：MIMIC 評估 15 個變數、NWICU 評估 14 個；完整診斷閘門皆 0 通過。
- Hct、platelets 在兩庫均通過七個病人切分，但不是每個有足夠樣本的病房都顯著改善。
- 缺 target 的未來預測：MIMIC 評估 38 格、NWICU 評估 37 格；僅 direct JEPA 的 creatinine@12h 在 MIMIC 通過，未在 NWICU 重現。
- 兩格事件需另有事件標籤；NWICU 尿量缺相容資料；platelets@1h 缺未用於頭部訓練的 MIMIC 支持量。這些不是模型失敗。

每項最多 4,000 位病人、每人一個預先按 hash 選定的有效 anchor，並非全量逐列測試。
同容量對照是同一模型讀取其他病人的 context，不是另外重訓的 placebo。
這次要求每個有支持量的病房都通過，嚴於原 forecast 的彙總病房閘門，
不能直接比較兩種通過數。結果只否決「現役模型能普遍遮蔽還原」的宣稱，
不撤回有實測 anchor 情境下的 forecast 證據。不新增 promotion、不套用舊區間。

完整逐格結果與重現方式見
[RELEASE_MASKED_CELLS_FINDINGS_20260903.md](../../../RELEASE_MASKED_CELLS_FINDINGS_20260903.md)。

### 遮蔽訓練修正與七 seed 重建頭

後續已修正當下重建 loss 與未來 latent 的語意錯接，並完成 eICU 的
A/B/C0/C 對照。只重訓重建頭可改善多數遮蔽誤差，且正常 forecast 逐筆完全
不變；聯合更新 encoder 雖讓部分 nowcast 更準，卻使部分既有 forecast 退步。

再固定 encoder，使用七個訓練 seed 的平均重建頭，並加入另行訓練的等容量
亂序 placebo：creatinine 在 NWICU 通過本輪完整遮蔽診斷，但 MIMIC 部分
神經病房仍未過。這不是新增 creatinine 未來預測格，也不是全身已可還原。
沒有變數跨兩庫完整過關，現役仍為 28 格可移動、13 格 persistence。

完整方法、逐變數誤差、負遷移與權重位置見
[MASKED_SHARED_STATE_REPAIR_FINDINGS_20260903.md](../findings/MASKED_SHARED_STATE_REPAIR_FINDINGS_20260903.md)。

### 來源醫院抽樣修正

查出前述 5,000-stay eICU cohort 按 stay ID 取前段，只涵蓋 12 家醫院。
已改成同預算、不同病人優先的跨醫院抽樣：5,000 位病人、208 家醫院，
可用 anchor 涵蓋 207 家。固定 JEPA 和七-seed readout 預算重跑後，MIMIC
10/15、NWICU 11/14 個當下重建 MAE 改善，但沒有新增完整通過 target。
部分病房退步與 trained-placebo 未過仍存在；現役 28/13 不變。
這不是單獨隔離醫院抽樣的因果實驗，病人數與來源病房組成也改變了。
逐指標結果與失敗拆解見
[MASKED_SOURCE_COHORT_REPAIR_FINDINGS_20260903.md](../findings/MASKED_SOURCE_COHORT_REPAIR_FINDINGS_20260903.md)。

### 四個泛化方向的受控測試（2026-09-03）

已完成原始觀測／latent ridge 探針、32／128／512 步訓練、病房平衡／group
DRO，以及有限共享 state 更新加預測保留約束。所有選擇只使用 eICU fit/tune，
另外保留 55 家來源醫院；MIMIC、NWICU 仍是已接觸的外部驗證資料。

延長訓練改善 MIMIC 10/15、NWICU 9/14 個當下重建 MAE，但沒有新增完整
跨庫通過 target。保留約束把既有 forecast 的彙總顯著退步格數由 MIMIC
11 降至 3、NWICU 11 降至 8，仍未消除負遷移。這不是新 forecast promotion；
現役權重與 28 可移動／13 persistence 不變。詳見
[完整結果與失敗分類](../findings/MASKED_GENERALIZATION_FINDINGS_20260903.md)。

### 穩健／非線性重建讀出（2026-09-03）

再固定 shared JEPA，測中位數回歸、Huber、小型非線性樹讀出與 eICU-only
選源。樹讀出改善 MIMIC 9/15、NWICU 8/14 個當下重建平均誤差，但沒有新增
完整通過項目。Hgb/Hct 部分改善集中在中間值，高低尾端可能更差；部分
病房仍顯著輸給等容量 trained placebo。未接入 serving，28/13 維持不變。
詳見 [逐方法、逐病房與尾端誤差](../findings/MASKED_ROBUST_READOUT_FINDINGS_20260903.md)。

## 14. 安全聲明

本專案只用於研究與工程驗證。不得用於診斷、處方、治療排序、醫療警報或任何病人照護決策。使用者必須自行確認資料授權、隱私、法規、單位轉換與外部驗證適用性。
