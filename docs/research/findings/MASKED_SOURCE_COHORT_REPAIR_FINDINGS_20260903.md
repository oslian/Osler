# 遮蔽共享 State：來源抽樣修正

## 結論

**已修正一個確定的來源選取偏差，完成重新抽取、訓練和評估；沒有新增 promotion。**

原本的 5,000-stay canonical eICU cohort 不是跨醫院抽樣，而是排序 stay ID
後取前 5,000 筆。它只有 12 家醫院，最大一家占 46.54% 住院資料。不能把它
視為全 eICU 的代表性訓練樣本。

修正後同樣取 5,000 次住院，先選不同病人，再以固定 ID hash 在醫院之間
round-robin。涵蓋 208 家醫院、5,000 位病人，最大醫院占 0.50%。選取不使用
病情、檢驗結果、未來標籤或出院狀態。醫院 ID 只用於抽樣與稽核，不進 encoder。
均衡抽样也不等於母群代表性抽樣，本次目的為避免固定預算只取到少數醫院。

修正改善了部分重建能力，但**沒有證明來源抽樣是所有失敗的唯一原因**。
這輪只處理同 anchor 遮蔽重建，不是未來 forecast 的 41 格 promotion。

## 實際支援量

| 項目 | 原 ID 前段 cohort | 修正 cohort |
|---|---:|---:|
| 選取住院數 | 5,000 | 5,000 |
| 選取病人數 | 3,347 | 5,000 |
| 選取醫院數 | 12 | 208 |
| 最大醫院住院占比 | 46.54% | 0.50% |
| 可用 anchor 病人 | 3,291 | 4,761 |
| 可用 anchor 醫院 | 12 | 207 |
| Discovery 病人 | 1,668 | 2,404 |
| 可用 care-unit 類型 | 6 | 8 |

修正 cohort 不是每個病人每個 target 都有新鮮測量。每個讀取頭實際 fitting
支援為 1,146–2,155 位病人；不能把 5,000 當成每格的有效 N。
完整 raw metadata 有 200,859 次住院、139,367 位病人、208 個 hospital ID。

## 固定的模型與評估

- 同一個 packaged JEPA encoder、scaler、53 維生理輸入與初始 reconstruction heads。
- Encoder 完全凍結；沒有新 body encoder、organ graph 或 latent 容量。
- 七個 seed：7、19、31、43、59、71、83；等權平均七個 head。
- 每 target/head 32 次更新、batch 128、AdamW、learning rate 0.003。
- 相同容量、初始權重與預算的 placebo，訓練及評估各自打亂不同病人的 context。
- Source patient split seed 73；source-only fitting，不用 MIMIC/NWICU 選參數。
- 真值為當下 age <=1h 的實測；其他輸入 age <=6h。
- target、相關兄弟變數及所有歷史副本均在 encoder 前遮蔽；不能靠目標自己的舊值作答。
- 同樣的七個 patient split、patient-level paired bootstrap 和全部受支援 care-unit
  閘門。每 unit 至少 30 位病人、至少兩個 unit；門檻沒有放寬。
- 同樣的 MIMIC/NWICU 檔案 hash、固定 outcome-independent anchor 選取，最多每
  target/database 4,000 位病人。不是全量 row census。

這是保持模型及優化預算的**來源 cohort 修正比較**，不是嚴格隔離單一因素的
因果實驗。不同病人的數目與 care-unit 組成也變了；重新抽取使用目前的 ledger
程式，包含既有血氣/體重事件修正。新增的非 checkpoint 變數不进入 53 維 encoder，
但可能改變 regular anchor 起點。本次不能把所有 MAE 差異只歸因於醫院均衡。

MIMIC 和 NWICU 已反覆被研究使用，這次仍是探索性外部驗證，不是新 lockbox。
七個 patient split 彼此重疊，不是七個獨立臨床研究。CI 未作多重比較確認性調整。

## 逐指標結果

物理單位 MAE；「前」是上一輪七-seed frozen-readout ensemble，不是最早的原始 A。

| Target | MIMIC 前 | 修正後 | NWICU 前 | 修正後 |
|---|---:|---:|---:|---:|
| bicarbonate | 3.7881 | 3.4928 | 3.6204 | 3.4161 |
| BUN | 12.7545 | 12.3754 | 13.1944 | 12.8903 |
| chloride | 5.0838 | 5.1198 | 4.7919 | 4.7868 |
| creatinine | 0.7502 | 0.7245 | 0.7188 | 0.7099 |
| glucose | 35.5545 | 35.5021 | 39.2493 | 38.9697 |
| heart rate | 13.8745 | 13.7033 | 13.9209 | 14.0598 |
| hematocrit | 4.8765 | 5.5426 | 5.1474 | 5.3982 |
| hemoglobin | 1.6647 | 1.8128 | 1.8220 | 1.8699 |
| MAP | 13.2382 | 11.3146 | 11.8802 | 11.6799 |
| O2sat | 2.2117 | 2.2032 | 2.3752 | 2.3736 |
| platelets | 71.9348 | 73.3248 | 72.1106 | 71.5297 |
| potassium | 0.4432 | 0.4365 | 0.4152 | 0.4115 |
| respiratory rate | 4.2916 | 4.0825 | 4.4604 | 4.3937 |
| sodium | 4.0591 | 4.1007 | 3.8826 | 3.8550 |
| urine output | 66.4010 | 64.1032 | 不支援 | 不支援 |

- MIMIC：10/15 MAE point estimate 改善；完整診斷 **0/15**。
- NWICU：11/14 改善；完整診斷 **1/14，仍只有 creatinine**。
- 跨兩庫同時完整通過：**0 個新增**。
- 完整診斷前的七個 patient split：MIMIC 7 targets、NWICU 8 targets 全過；
  不能把它們當成已通過病房閘門。
- NWICU 尿量未有相容 label，屬 unsupported，不列為模型失敗。

## 失敗到底在哪

1. **整體有效，但局部仍不確定或退步。** MIMIC creatinine 的 pooled delta
   vs trained placebo = -0.05924，95% CI [-0.06668, -0.05177]；population
   病房閘門 9/9，但 trained-placebo 病房閘門 6/9。Neuro Stepdown 的 delta
   +0.03790，CI [-0.00246, +0.07735]。之前此 unit 顯著退步，現在跨 0，仍不是過關。
2. **有些不是只有 power 不足。** MIMIC chloride 對 population 和 placebo 的
   pooled CI 都完全大於零，表示在這套已遮蔽輸入下真的比較差。MIMIC Hgb/Hct
   同樣整體退步，不能靠「多醫院」或「有共同 state」當作放行理由。
3. **一些改善只是更好的族群中心，尚未穩定利用個人訊號。** glucose、尿量
   在 MIMIC 可勝過 population，但 trained-placebo pooled CI 跨 0；沒有完成個體
   context 增益的證據。Na 在 MIMIC 整體顯著輸 placebo。
4. **抽樣解決醫院集中，不會自動解決病房/生理族群不匹配。** 修正後可用 cohort
   有 3,623/4,761 位 Med-Surg ICU 病人，Neuro ICU 為 110 位；原 cohort 神經人群
   較多。需要區分來源支援與可轉移關係，不能把所有剩餘問題都診斷成同一個原因。

`artifacts/masked_source_repair_comparison_20260903.json` 對每個 target/database/control
保留 pooled CI、patient 通過數、care-unit 通過數、未過 unit 與「顯著更差」unit。
沒有把 CI 跨 0 和 CI 全為正混在一起。

## 工程修正與驗證

- `osler_jepa/event_adapters.py`：以 ID-only、order-invariant、patient-first
  hospital sampling 取代 `sort_values(stay_id).head(max_stays)`。
- `extractors/canonical_event_ledger_extract.py`：報告實際醫院覆蓋和集中度。
- `extractors/canonical_event_examples.py`：將 anchor 設定記入可重現報告。
- 遮蔽 audit 保留 hospital provenance 供檢查，但 encoder whitelist 排除它。
- 比較程式拒絕未完成 run、不同 release、不同外部資料 hash 或不同 evaluation 設定。
- 56 個相關測試通過；release checksum/schema/end-to-end validator 通過。
- 現役輸出仍為 **41 格，28 可移動、13 persistence**。沒有改 serving 權重或 calibrator，
  不把這些重建 head 餵回 forecast，也不替它們借用舊 forecast 區間。

## 重跑指令

先在已安裝專案依賴的 Python 環境中執行。輸出路徑不可覆蓋先前實驗；以下使用
新的目錄名稱。Raw 路徑需對應合法下載的 eICU；patient-level parquet 只留在本機。

```sh
python -m extractors.canonical_event_ledger_extract \
  --database eicu --data-root /path/to/eicu-collaborative-research-database-2.0 \
  --output-dir /private/tmp/eicu_balanced_repeat \
  --report artifacts/eicu_balanced_repeat_ledger.json \
  --max-stays 5000 --exclude-treatments

python -m extractors.canonical_event_examples \
  --event-data /private/tmp/eicu_balanced_repeat \
  --output /private/tmp/eicu_balanced_repeat_examples.parquet \
  --report artifacts/eicu_balanced_repeat_examples.json \
  --horizons 3 --lookback-hours 6 --anchor-step-hours 3 --maximum-episode-hours 96

python -m scripts.audit_masked_readout_ensemble \
  --eicu /private/tmp/eicu_balanced_repeat_examples.parquet \
  --mimic-current /path/to/mimic_current_examples.parquet \
  --nwicu-current /path/to/nwicu_current_examples.parquet \
  --output-dir artifacts/masked_readout_balanced_repeat --max-patients 4000 --threads 4

python -m scripts.compare_masked_readout_reports \
  --before artifacts/masked_readout_ensemble_20260903/report.json \
  --after artifacts/masked_readout_hospital_balanced_20260903/report.json \
  --output artifacts/masked_source_repair_comparison_repeat.json
```

本次聚合結果：`artifacts/masked_readout_hospital_balanced_20260903/report.json`。
候選重建 heads：同目錄 `readouts.pt`，綁定原 frozen encoder hash，不是獨立完整模型。
抽取報告：`artifacts/eicu_hospital_balanced_ledger_20260903.json`；examples 報告：
`artifacts/eicu_hospital_balanced_examples_20260903.json`。

沒有臨床、因果、完整隱藏人體狀態或新增測量時間預測宣稱。
