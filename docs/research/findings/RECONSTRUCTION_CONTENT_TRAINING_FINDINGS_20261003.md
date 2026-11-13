# E21：內容保留監督沒有完成未知指標修復

## 新增了什麼證據

E20 定位的是已知觀測內容在事件寫入與後續 state 路徑的可讀性差距，
不是不可逆資訊損失。E21 首次按本 protocol 干預原 state 更新的訓練目標：
凍結原事件 encoder，只訓練原 state layer、masked readout 與新增內容讀取頭。
這不是再對凍結 body 換一種讀取器，也不是 future forecast。

**結果：部分已知內容可以更好讀回，但尚無未知指標完整勝過兩個對照。**
不能將 E20 的位置診斷與 E21 的局部內容改善合併宣稱「泛化根因已修好」。

## 固定比較

- 同 E19 五個 fold 的固定 hash context，每 fold 2048 fitting、1024 selection。
- 每位病人每 fold 一個 masked context；fold 有重複病人，先平均其 loss，
  不能把五 fold 當成五個獨立 cohort。Selection 共2048位不同病人。
- 原目標與其歷史、兄弟變數全部遮蔽；內容 loss 只用其他實際觀測。
  內容真值是最後已知量測，不是假定當下生理必然等於舊值。
- 三臂相同初始化、順序、容量、optimizer 與預算：primary、real、shuffled。
  Primary 也訓練內容頭，但在該 loss detach state，避免額外 head 訓練量混淆。
  Shuffled 逐變數旋轉 fitting 病人的內容標籤，保留缺失遮罩與值分布。
- 內容頭1 pass暖身，再2 passes joint training；權重0.25固定，沒有 sweep。
  原 epoch10 模型保留為 frozen baseline；其他比較控制額外訓練與 optimizer 重啟。
- 五fold共15個模型；5120個 masked query occurrences、89788個內容 query
  occurrences，共94908。全53欄位列出支持情況，不挑選有利指標。
- Masked46項有評分資料，其中27項達最低30病人／3來源組要求；內容47項
  有評分資料，其中41項達最低支持要求。「可評分」不是「通過」。
- 兩種任務、兩個主要比較、53欄位的固定多重比較控制；病人與來源組
  paired bootstrap 95% CI上界小於0，且單尾p小於0.05/(53*2*2)。
  所有資料仍是重複使用的 development，不是新外部 lockbox。

## 結果

下表是 normalized target-macro MAE，越低越好，不是準確率。

| 任務／等權 | 原 frozen | 同量 primary | 真內容 real | 亂序 shuffled |
|---|---:|---:|---:|---:|
| 未知 masked／病人 | 0.877001 | 0.875550 | 0.874808 | 0.875287 |
| 未知 masked／來源組 | 0.922048 | 0.920441 | 0.919490 | 0.920109 |
| 已知內容／病人 | 不適用 | 0.887196 | 0.885561 | 1.001899 |
| 已知內容／來源組 | 不適用 | 0.991882 | 0.990278 | 1.111248 |

- 未知推論相對同量 primary，病人 macro 改善 **0.0848%**；兩個主要比較
  均 **0項完整screen**。27/46項均值改善，不代表27項驗證成功。
- 已知內容相對 primary 改善 **0.1842%**；40/47項均值改善。
  Hemoglobin、respiratory_rate 兩項同時通過 real對primary與real對shuffle。
  Real對shuffle單獨通過16項；不能忽略較強的primary對照只報16項。
- Shuffled 內容頭本身用錯配標籤訓練，讀回較差是預期控制結果；
  因此單獨勝過shuffle不能證明共享state帶來額外價值。
- 此次支持集合、讀取頭與任務與E20不同，不直接把兩輪macro相減當修復幅度。

### 反例：記得更好不等於推得更好

| 指標 | 已知內容 primary → real | 未知 masked primary → real |
|---|---:|---:|
| Hemoglobin | 0.354554 → 0.349829 | 0.486077 → 0.486479 |
| Respiratory rate | 0.656781 → 0.653341 | 0.670778 → 0.670193 |
| BUN | 0.433629 → 0.430531 | 0.418176 → 0.419191 |
| Creatinine | 0.708119 → 0.703599 | 0.686746 → 0.686387 |

Hemoglobin內容讀回過兩對照，未知推論均值反而略差，該退步CI仍跨0；
不能說已證明有害，但更不能說已修好。BUN相對frozen有改善，
primary額外訓練已有更大改善，不能將它歸因內容監督。
呼吸率與creatinine未知推論的微小改善也未通過固定screen。

## 現在能定位到哪裡

- **已確認：** 固定介入可以小幅改善部分已知內容的可讀回性。
- **已否決的具體修法：** 此固定0.25權重、兩pass、凍結事件encoder的
  全域內容保留訓練，尚不足以成為未知值重建的通用修復。
- **仍未區分：** 可讀性、狀態演化目標取捨、事件encoder缺口和跨來源差異
  各自的責任。有限訓練預算不支持宣稱方法家族不可能有效。
- **不能推論：** 內容一旦保住，未知指標與未來預測自然就會改善。
  本輪沒有測future、個體在線自學習、區間或因果。

不因本次結果掃weight／seed／epoch，也不移除退步目標後聲稱成功。
下一研究須先提出能區分原因的證據，例如拆開後續advance與新觀測update
的同條件干預；這仍未由E21完成，亦不是在本報告中自動啟動的實驗。

## 完整性與交付

- 所有15360個fit/selection context的原body重現最大誤差0，原native預測亦逐例核對。
- 15個保存模型逐筆重算，94908×3個預測全部精確重現；encoder state_dict未改。
- Hash、支持、病人身份／真值、fit-selection隔離、三臂資料一致性核對通過。
- 24項相關測試通過；訓練59333與核對99496皆正常exit0。
- 私有病人cache、模型權重、逐筆預測不進Git；aggregate與程式保留。
- 工作樹既有未提交依賴按實際hash綁定，並非乾淨checkout完整release宣告。
- **現役權重、registry、區間及serving未改；0新增promotion。完整修法未完成。**

證據：`RECONSTRUCTION_CONTENT_TRAINING_PROTOCOL_20261003.md`；
`artifacts/native_reconstruction_content_training_assessment_20261003.json`；
五份`native_reconstruction_content_training_foldN_20261003.json`；
`artifacts/native_reconstruction_content_training_verification_20261003.json`。
