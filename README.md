# Model / WholeBody-JEPA

2026-10-05 獨立交付至 **oslian/Model**。包含實際權重、程式、提取器、
輸入轉換器、測試、校準產物與研究文件，不只是訓練方法。
Independent private working-tree snapshot. Licensed patient datasets are excluded.

## 先選對介面 / Choose the interface

| 用途 | 權重 | 入口 | 限制 |
|---|---|---|---|
| 現役未來預測 | `models/`，clone已附實際二進位檔 | `scripts.predict_patient` | 41格；本次合成驗證27 move、14 persistence；依病人資料仍可abstain |
| Shared-body當下重建 | 根目錄4份 `coverage_fold*_real_epoch10.pt` | `scripts.predict_coverage_csv` | 研究候選；不是future forecast；沒有校準區間 |
| 歷史研究／干預權重 | 同倉庫Release的 `research-weights-*.tar.gz` | 按各自訓練與audit程式載入 | 包含負結果與對照，不可當成promoted模型 |

最新研究權重不等於最佳模型。完整共享state泛化修法仍未交付，見
`ROOT_CAUSE_LEDGER.md`。本次僅交付，沒有訓練、沒有改動現役權重或gate。
歷史文件的格數是當時快照；目前以registry、selector與validator為準。

## 安裝與forecast

```sh
git clone https://github.com/oslian/Model.git
cd Model
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m scripts.validate_release
python -m scripts.predict_patient --input examples/patient_1.csv \
  --output outputs/patient_1_forecast.json --model-dir models
python -m scripts.show_contract
```

依賴以 `pyproject.toml` / `requirements.txt` 為準，特別是
`scikit-learn==1.7.2`。joblib/pickle只能載入可信來源的檔案。
一個CSV是一個病人的歷史；不自行normalize，不把不同病人串成一條軌跡。

```csv
hours_since_onset,heart_rate,map,o2sat,glucose,creatinine,urine_output
0,108,68,93,245,1.4,42
3,102,72,94,221,,0
6,96,76,95,198,1.3,68
```

- 空白才是missing。`0`是真實零值，尿量零不可刪除。
- 尿量必須是 **mL/hour**，不是不定長收集區間的總體積。
- 不必填滿全部變數；freshness、觀測支持與已驗證資格決定是否可輸出。
- 正式區間只有相容calibrator及gate通過時輸出，null不等於零寬區間。
- 詳細CSV、單位、缺失、輸出與MIMIC/eICU提取命令見
  [RUNTIME_GUIDE.md](RUNTIME_GUIDE.md)。其歷史成果段落並非本次重新驗證。

### 311維shared-head內部介面

不要手動拼向量，使用 `osler_jepa/shared_release_features.py` 的builder。

| 索引 | 寬度 | 內容 |
|---|---:|---|
| 0:64 | 64 | shared context latent |
| 64:160 | 96 | patient belief state |
| 160:224 | 64 | target-specific future latent |
| 224:256 | 32 | target organ token |
| 256:257 | 1 | **JEPA的normalized base forecast，不是當下目標值** |
| 257:310 | 53 | checkpoint順序的normalized measurement state |
| 310:311 | 1 | normalized current target value |

## 當下重建：native-history完整輸入包

4個獨立訓練fold為0、1、3、4；fold2與fold0等價。不是ensemble，
也沒有經驗證的新病人fold選擇器。下面固定fold0只是執行範例。

```sh
python -m scripts.load_coverage_candidate coverage_fold0_real_epoch10.pt
python -m scripts.predict_coverage_csv \
  --checkpoint coverage_fold0_real_epoch10.pt \
  --events examples/coverage_synthetic_events.csv \
  --stay-id demo-1 --anchor '2100-01-01 02:00:00' --target bicarbonate
python -m scripts.predict_coverage_csv \
  --checkpoint coverage_fold0_real_epoch10.pt --schema
python -m scripts.verify_coverage_bundle --report COVERAGE_VERIFICATION.json
```

完整中英文指南：[COVERAGE_CANDIDATE_USAGE.md](COVERAGE_CANDIDATE_USAGE.md)。
此倉庫已附權重，不必照舊指南再次下載另一倉庫的release。

| 事件CSV欄位 | 填法 |
|---|---|
| stay_id | 本次住院代號，只用來選取病人，不是數值特徵 |
| source_kind | `lab_available`或`bedside_available` |
| source_id | 穩定原始事件ID；真正重測即使值相同也用新ID |
| event_time | 結果實際可知時間，不能假設等於抽血／charttime |
| measured_at | 真實採樣／測量時間，不可晚於event_time |
| variable | `--schema`列出的canonical名稱 |
| value | 物理單位的有限數字；未測則不新增假列 |
| unit | `--schema`指定的單位，大小寫一致 |
| quality_valid | 真正完成品質與單位檢核才填true |

包含 `whole_body_jepa/coverage_input.py`、`native_history_schedule.py`、
`native_reconstruction_examples.py`等真實builder。48h歷史、結果可知時間、
兄弟變數遮罩、mask／age與scaler遵守原訓練規格，不填dummy target。
這個介面不使用上面的311維人工向量。合成範例不是病人準確率證据。

## 完整研究權重 / Research release assets

[同倉庫Release: snapshot-20261005](https://github.com/oslian/Model/releases/tag/snapshot-20261005)

```sh
gh release download snapshot-20261005 --repo oslian/Model \
  --pattern 'research-weights-*.tar.gz' --dir downloaded-assets
python -m scripts.restore_research_assets --asset-dir downloaded-assets
python -m scripts.restore_research_assets --asset-dir downloaded-assets --extract
```

第一個Python命令只核對，第二個明確解壓，拒絕覆蓋不同內容的既有檔案。
`RESEARCH_ASSETS.json`列壓縮檔hash與內容；`RESEARCH_WEIGHTS_MANIFEST.json`
列原檔／匯出hash、移除欄位、候選狀態及排除項。

- 權重保留；`seen`、`pending`、病人queue、optimizer/RNG等訓練容器去除。
- 不是精確resume包。完整重跑仍需合法原始資料和private split/cache重建。
- 保存原相對路徑供查報告，不承諾所有舊audit能在沒有私有資料時一鍵重現。
- 可能保存訓練向量的reader與未核對容器列為排除，不把病人資料當權重上傳。
- 不含MIMIC/eICU/NWICU原始資料、波形、病人CSV或憑證；不提供差分隱私保證。
- 本次模型範圍是WholeBody-JEPA當前checkout，不是整台電腦全部歷史專案。

## 文件與程式地圖

| 路徑 | 內容 |
|---|---|
| whole_body_jepa/、osler_jepa/、根目錄Python | runtime、架構、共享state與契約 |
| extractors/ | 原始表格到canonical events/examples |
| scripts/ | 訓練、推論、評估、診斷、打包 |
| tests/、examples/ | 回歸測試與純合成範例 |
| models/ | 現役模型、heads、registry、selector、calibrators |
| artifacts/ | 可分享彙總結果；排除項列在publication manifest |
| ROOT_CAUSE_LEDGER.md | 唯一根因帳本，已做／未做與負結果 |
| PROJECT_STATE.md、EXPERIMENT_INDEX.md、graph.md | 日期化狀態與歷史索引 |
| PUBLICATION_MANIFEST.json、SNAPSHOT_SHA256.json | 快照來源、範圍與完整性 |

本次獨立副本完成release端到端驗證、188個fold-target合成parity檢查、
19項input/runtime/archive回歸。不是新的臨床准确率或外部泛化證據。
主線研究不由此發布任務續跑；未獨立核對的結果不自動算完成或promotion。

**Research only. Factual only. `causal_claim_allowed=false`.**
不可用於診斷、處方、治療選擇或病人照護決策。

## 輸入變數與現役格目錄

以下表格沿用現役輸入契約。實際是否移動仍以source selector及病人條件為準。

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
