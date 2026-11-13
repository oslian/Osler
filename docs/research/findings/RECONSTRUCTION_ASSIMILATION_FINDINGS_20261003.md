# 已知內容的 state 路徑定位：E20

## 新證據與範圍

上一項 E19 只知道 last-observed event 比 final body 容易讀回同一個已知量測。
本項固定同一 encoder、病人、真值及讀取器規格，加入當次量測剛寫入 state
的快照；每個階段各自在 fitting 病人上訓練讀取頭。不是拿 final readout
直接讀中間 state，也不是未知 target 重建或未來 forecast。

**新判斷：可用性差距不只在最終讀取。當次 assimilation 後已有差距，
後續 state 路徑還有額外差距。** 這是在兩種有限讀取器下得到的證據，
不是資訊不可逆丟失的證明，也還沒有完成修復。

## 固定對照

- 原 E19 全 53 變數支持盤點；36 有支持，35 有完整五 fold，anion gap 僅 fold0/1。
- 每 fold 固定 hash 抽樣最多 2048 fitting、1024 selection 病人，每人一個 context。
- 同 177 variable-fold，新增 354 個 post-update 讀取器；共 88,122 查詢，
  2,048 位不同 selection 病人。不同 fold 有重複病人，不是五個獨立 cohort。
- 三種表示皆 64 維加相同 159 個 process covariates，Ridge100 與原 HGB100。
- post 在该變數最後實測事件後取樣；final 含之後合法事件及演化。
  process covariates 在共同 anchor 已知，因此這是回顧性保留診斷，
  不能將 post reader 當成較早時間的可部署預測器。
- 四個方向比較，53×4 多重比較；重複 fold 的 loss 先按病人平均，
  再按來源 component 平均。兩級 bootstrap CI 上界<0，且配對單尾
  p<0.05/(53×4)、至少30病人及3 components，才通過診斷 screen。
- 沒有更改來源模型、增加 outer 查詢、調參、挑有利目標或放寬 gate。

## 結果

以下為 normalized target-macro MAE，越低越好；不是原始臨床單位或準確率。

| 讀取器／等權方式 | 單事件 event | 剛寫入 post | 最終 body |
|---|---:|---:|---:|
| Ridge／病人 | 0.911219 | 0.999468 | 1.259259 |
| Ridge／來源組 | 0.962995 | 1.067097 | 1.291298 |
| HGB／病人 | 0.628675 | 0.686716 | 0.893199 |
| HGB／來源組 | 0.680428 | 0.746463 | 0.967497 |

36 個受支持變數的病人等權均值，兩種 reader 都是 event < post < final。
均值方向一致不代表36項均通過統計檢查。

| 比較 | Ridge screen | HGB screen | 兩 reader 共同 |
|---|---:|---:|---:|
| event 優於 post | 19 | 12 | 12 |
| post 優於 final | 19 | 17 | 13 |

即時差距共同12項：bicarbonate、chloride、glucose、heart_rate、MAP、
O2sat、platelets、potassium、PTT、respiratory_rate、temperature、WBC。

後續差距共同13項：AST、glucose、heart_rate、hematocrit、hemoglobin、
INR、MAP、O2sat、platelets、PTT、respiratory_rate、temperature、WBC。

兩段皆通過共同 screen 的9項：glucose、heart_rate、MAP、O2sat、
platelets、PTT、respiratory_rate、temperature、WBC。
以上均有完整五fold支持，但仍是同一開發資料的診斷，不是外部 promotion。

腎臟反例也必須保留：BUN 與 creatinine 的均值同樣逐段退步，
但沒有兩 reader 共同通過本次任一階段 screen。不能把全身均值結論
寫成腎臟的唯一已確認根因。Creatinine 的後續差距只有 Ridge 通過。

| 指標（HGB／病人） | event | post | final |
|---|---:|---:|---:|
| BUN | 0.355173 | 0.385171 | 0.420840 |
| Creatinine | 0.487809 | 0.519748 | 0.630285 |
| Bicarbonate | 0.393049 | 0.455500 | 0.526704 |
| Glucose | 0.460243 | 0.543446 | 0.688468 |
| INR | 0.301552 | 0.337155 | 0.803635 |
| Heart rate | 0.174499 | 0.252091 | 0.391713 |

## 第一次失敗在哪裡

- **表層現象：** 同一已知量測，沿 event → post → final 越來越難讀回。
- **已定位操作：** 對共同12項，差距在 event representation 混入先前 body
  的 assimilation 後已出現；共同13項在後續演化與新觀測更新後進一步出現。
- **尚未證實的機制：** gate 是否覆蓋舊內容、dynamics 是否漂移、座標變得
  難讀、以及原本只訓練未知 target 推論是否刻意犧牲已知值精確度。
  本項不能分開這些因素，也未把後續 advance 與後續 update 單獨干預。

因此不能說只需換 final head，也不能把「混合後難讀」改說為「資訊必然消失」。
E17 已發現簡單 event pooling 不能普遍取代 state 對未知目標的推論。
E16 的全域刪除 dynamics 也會傷害其他目標，這些反例仍有效。

## 執行修復與核對

首次執行在尚未完成任何 cache／reader 前，由 event 重現防線停止。
原因是 E19 編碼 unique last-observed events，E20 原先以全 history 批次
比較，同一 float32 encoder 的批次算術相差到1.6093e-6，超出原界限。
修正為原批次作事件重現，state replay 仍保持原完整歷史路徑；未放寬容差。
停止時只產生 contract，已封存在私有目錄，沒有覆蓋既有實驗。

完整執行後：

- 所有 context 的 event／final 與 E19 **重現誤差皆0**。
- 不同 batch shape 的 encoder 差異另記錄，最高6.914139e-6，
  不是此次模型成效差距的估計，也不作為模型失敗證據。
- 177 組模型、354 新 readers、88,122 預測全數重現；原 E19 所有欄位不變。
- 177 scalers 全部有限正尺度；76 個有極小負 variance，最低−3.626051e-38，
  均符合既有常數 roundoff 界並使用 scale=1。沒有非有限輸出。
- 24 個相關單元測試通過。訓練／驗證程序皆正常結束，CodeGraph 已同步。
- 工作樹另有既有未提交依賴；artifact 綁定實際工作樹 hash。
  這不是乾淨 checkout 的完整 release 可重現宣告。

## 修復方向與停止條件

下一個有效介入應在**同一個持續 body state**內測試保留觀測內容的
訓練／更新約束，同時保留原 masked-target 推論對照。先查既有訓練目標，
確認是否已有等價項目，再定義唯一改動；本項尚未執行這個修復。

只把數值抄進旁路可以完美讀回已知值，卻沒有證明能推斷未知生理，不能算成功。
只改善已知值而傷害 masked-target 推論也不能接管。還需要區分後續
advance 和 observation update 的貢獻，不能將兩者統稱成已證明的「遺忘」。
沒有更改現役權重、registry、區間或 serving；沒有新增 promotion。
完整泛化根因與修法仍未完成。

原始證據：`artifacts/native_reconstruction_assimilation_assessment_20261003.json`、
五份 `native_reconstruction_assimilation_foldN_20261003.json`、
`artifacts/native_reconstruction_assimilation_verification_20261003.json`。
私有逐病人資料、state、讀取器與預測不進版本控制。
