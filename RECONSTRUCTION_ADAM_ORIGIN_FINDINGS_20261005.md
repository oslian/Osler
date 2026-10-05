# E37：有害方向來自自身訓練，還是其他目標？

## 這次新增的證據

E36指出部分已保存修正方向在零起點就不利。這次重現E34同一個immediate
bridge的實際AdamW訓練，追蹤每個target梯度的動量如何累積成最終權重。
不是再取少量病人計算gradient cosine，也不是改參數再試一次準度。

固定五個fold紀錄、每個4096更新、原訓練順序和設定。全部原模型tensor與
epoch log精確重現；沒有覆蓋原checkpoint。再對E36全部20,480查詢，
計算零bridge起點的局部MAE方向貢獻。53個target來源、weight decay和
數值rounding餘項分開保存。重複fold先於病人內聚合，再作病人等權。

## 結果：不是一個問題

以下只統計**總初始方向使MAE均值增加**的指標，非顯著性或promotion數：

| 角色 | 總方向有害 | 自身有益、其他target有害且抵消 | 自身貢獻也有害 |
|---|---:|---:|---:|
| fit | 7 | 7 | 0 |
| selection | 18 | 8 | 10 |
| same-source query | 11 | 9 | 2 |

Selection的8項是AST、bilirubin、glucose、lactate、O2sat、phosphate、
potassium、TSH。自身也不利的10項是direct bilirubin、chloride、CPK、
free T4、hematocrit、lipase、PTT、triglycerides、troponin I、WBC。
這是診斷清單，不是可以部署的真值路由規則。

來源組等權的對應三列是13=13+0、20=8+12、17=14+3。
組別與病人權重會改變符號；來源component不能直接稱為獨立醫院。

**新增判斷：** 部分局部傷害可追到跨target更新貢獻互相抵消；
但另外一些target自己的訓練貢獻也不能轉移，不能只用共享衝突解釋所有失敗。
fit內亦有7項被其他target抵消，故也不能將所有問題歸咎外部資料差異。

## 保留反例，不全域切斷耦合

數字為標準化MAE的局部方向貢獻，不是臨床單位或預測改善百分比。

| target／角色 | 自身來源 | 其他target來源 | 總方向（含decay／rounding） |
|---|---:|---:|---:|
| glucose／selection | -0.002138330 | +0.002166933 | +0.000026456 |
| phosphate／selection | -0.002575976 | +0.009609705 | +0.007016721 |
| hematocrit／selection | +0.001405959 | +0.001171678 | +0.002572791 |
| BUN／same-source | -0.001484904 | +0.004058012 | +0.002568405 |
| creatinine／selection | -0.000449395 | -0.001011996 | -0.001457570 |
| hemoglobin／selection | -0.008394009 | -0.000485512 | -0.008854209 |

Creatinine和hemoglobin同時受益於自身與其他來源，是「把所有target隔離就好」
的反例。Glucose差很小，不能稱為顯著失敗。來源矩陣中WBC對glucose的貢獻
不利，也不表示WBC生理上造成血糖惡化；這只是共享參數的訓練來源歸因。

## 歸因的限制

- Adam動量會跨step延續，因此不能把整次更新歸給當前樣本的target。
- 本分解使用共同訓練軌跡下的second moment、clipping與梯度；刪掉一個
  任務後，這些都可能改變。**不是leave-task-out因果效果，也不是修復試驗。**
- 測的是零起點局部導數；不能保證移除某來源後，完整非線性replay會更準。
- fold0與fold2的來源向量相同，不是五次獨立複驗；結果按病人聚合，
  沒有把重複fold當作額外獨立病人或新的顯著性。
- 仍是當下遮蔽恢復。不是future forecast、自學習回放或生理治療因果。
- MIMIC／NWICU已反覆使用，沒有新的未接觸lockbox。

## 驗證範圍

- 25項新舊測試通過：動量延續、零梯度與decay、原fit精確重現、
  inference-mode隔離、零殘差非可加性、病人／來源權重與非法輸入。
- 正式98004自然exit0；五個原模型所有tensor與原epoch logs精確重現。
- 來源向量加總成原權重最大差1.110223e-16。
- 全20,480查詢起點值差0；來源加總對E36導數最大差1.110223e-16。
- 來源彙總與scalar/fsum最大差1.040834e-17；另行收尾核對總方向差6.938894e-18。
- 52項bindings、15角色全部原欄位／身份／支持量與角色互斥核對通過。
- 浮點來源分解的rounding向量norm約4.47e-5至5.37e-5，原權重norm
  約1.32至1.58；first-moment分解對原float32最大差2.11e-9。餘項顯式保留，
  不藏進任何target。這些數值核對不能替代準度驗證。
- 收尾第一次命令import名稱錯誤，修正後5376正常完成；不修改原模型或科學門檻。

## 修復還缺什麼

下一項干預必須同時面對兩類原因：保留有益的共同更新而避免有害抵消，
以及處理自身學到的方向不能轉移。不能僅根據selection清單刪來源後宣布成功。
應先核對已有protected-gradient等實驗，再固定不同於舊方案的干預及等預算對照；
必須評估完整幅度的真實誤差和原強raw基線，不能只改善局部導數。

本輪沒有修復介入、新promotion或runtime／現役權重／區間變更。
完整泛化修法仍未交付；E37新增的是更新來源的區分，不是再次重述「有訊號但不穩」。

## 產物

- [固定協議](RECONSTRUCTION_ADAM_ORIGIN_PROTOCOL_20261005.md)
- [完整aggregate與bindings](artifacts/native_adam_origins_20261005.json)
- `scripts/diagnose_adam_origins.py`、`tests/test_adam_origins.py`
- [E36參數路徑](RECONSTRUCTION_PARAMETER_PATH_FINDINGS_20261005.md)
