# E22：後續觀測更新比單純時間演化更影響舊量測讀回

## 結論與適用範圍

**新證據支持：在這個固定模型與已知內容任務中，後續觀測寫入是較明顯的讀回差距來源。**
不是再次泛稱「共享 state 有訊號但不穩」，也不是完整泛化修復。

E20把差距分成當次assimilation與後續路徑；本輪再把後續路徑分成
時間演化、觀測更新及完整組合。每個表示各自fit相同規格的reader，
避免直接用不適配的最終head評分中間state。

## 固定比較與規模

同E19/E20五fold、2048位不同selection病人、177個variable-fold、88122查詢。
53項全列支持，其中36項可完成比較、35項有全五fold；新增708個readers。
五fold病人有重複，先平均病人loss，不當作五次獨立replication。

在每個變數最後一次已知觀測寫入後，保留四條路徑：

- `post`：保留當次state快照，不繼續更新。
- `advance`：只做原時間演化，不寫入後續觀測。
- `update`：時鐘與gate的delta正常進行，只寫入後續觀測，不做latent dynamics。
- `body`：原本完整時間演化＋觀測更新。

四者都讀回同一個**最後已知量測值**，不是未量測的當下真值。
Ridge/HGB各自使用64维body＋同159流程欄位，fit角色、權重與預算相同。
原post/body預測不改。固定53×4×2家族，patient與component paired CI及p同時過關，
最低30病人／3來源組。詳見protocol；沒有外部新lockbox或runtime promotion。

## 實際結果

以下是normalized target-macro MAE，越小越好，不是準確率百分比。

| 路徑 | Ridge病人等權 | HGB病人等權 | Ridge來源組等權 | HGB來源組等權 |
|---|---:|---:|---:|---:|
| 當次post快照 | 0.999468 | 0.686716 | 1.067097 | 0.746463 |
| 只有時間演化 | 1.017629 | 0.727531 | 1.082523 | 0.803395 |
| 只有後續觀測更新 | 1.244731 | 0.885117 | 1.281753 | 0.956254 |
| 完整路徑 | 1.259259 | 0.893199 | 1.291298 | 0.967497 |

| 固定比較（左側誤差較小） | Ridge通過 | HGB通過 | 兩reader共同 |
|---|---:|---:|---:|
| post < advance | 1 | 3 | 0 |
| post < update | 14 | 13 | 10 |
| advance < body | 14 | 13 | 8 |
| update < body | 0 | 0 | 0 |

共同10項post<update：bicarbonate、hematocrit、hemoglobin、INR、MAP、platelets、
PTT、respiratory_rate、temperature、WBC。
共同8項advance<body：AST、hematocrit、hemoglobin、INR、platelets、respiratory_rate、
temperature、total_protein。

兩reader病人均值全部36項post<update，但**不是36項皆通過統計screen**。
相反地，時間演化也有部分均值差距；共同0項不證明它完全無害。
差中差僅描述性列在JSON，不能將macro差距百分比當成精確的因果責任占比。

### 腎臟反例

| 目標／HGB病人MAE | post | advance | update | body |
|---|---:|---:|---:|---:|
| BUN | 0.385171 | 0.402483 | 0.415012 | 0.420840 |
| Creatinine | 0.519748 | 0.539721 | 0.631524 | 0.630285 |

腎臟均值也有差距，但兩項都没有兩reader共同screen。
Creatinine完整路徑甚至略優於update-only；不能說所有時間演化都應關掉。

## 為什麼還不是完整修復

1. 舊量測不是當下真值；新觀測可能合理地改變對病人生理的判斷。
   保留舊值最好的state，不一定是未知值推論最好的state。
2. E21已實測內容保留loss，局部讀回改善未轉成未知值完整通過。
   因此本輪不能用「刪除update」或「再加同一保留loss」宣稱修好。
3. 各條路徑是診斷干預，reader的座標／擬合限制仍存在。
   不是不可逆資訊損失證明，也不是治療因果或器官生理因果。
4. 未測future forecast、個人線上更新、區間或未接觸外部驗證。

判斷改變：從「後續advance/update責任未拆」改為
「對部分已知內容的可讀性，後續update有更一致的干預證據」。
下一個修復必須證明改善寫入方式能同時保留有用歷史、利用新觀測、改善原masked推論。
不能再只增加記憶能力就推定全身預測變好。完整修法仍未交付。

## 完整性

- 原post/final state逐context重現最大誤差均0。
- 708保存模型逐筆預測重現，原E19/E20基線未改；620檔bindings核對。
- 配對統計與screen完全相同；重新載入Parquet後MAE彙總的浮點加總
  roundoff最大7.11e-15。修正verifier只對彙總容許64eps尺度誤差，
  配對p/CI、支持量與screen仍要求完全相同；沒有改gate或結果。
- 354個scaler皆通過數值防線；152個有極微負variance，最低-3.63e-38，
  均為常數roundoff及有限正scale，不是NaN預測。
- 37項新舊測試通過；訓練58603、最終核對27140均正常結束。
- 沒有修改現役權重、registry、區間、serving，沒有新增promotion。
- 私有states、病人預測與reader權重留本機；只保存aggregate研究產物於Git。

證據：`RECONSTRUCTION_SUFFIX_PROTOCOL_20261003.md`、
`artifacts/native_reconstruction_suffix_assessment_20261003.json`、
五份`native_reconstruction_suffix_foldN_20261003.json`、
`artifacts/native_reconstruction_suffix_verification_20261003.json`。
