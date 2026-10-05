# E34：固定主幹仍不能讓修正穩定轉移

## 問題與對照

E33只能定位固定checkpoint的bridge推論責任，不能區分它與主幹共同訓練的責任。
本次沿E29同一epoch10起點、相同病人／遮蔽／資料順序／兩遍預算，
只更新bridge投影，完整encoder、state與readout逐tensor固定。
梯度仍穿過固定的歷史演化，不使用detach阻斷早期修正。

五fold、immediate／delayed／body三通道各28,672個可訓練參數，15個新模型。
Fit 10,240、selection 5,120、same-source 5,120次查詢；53項完整盤點，
三角色分別47／46／44項有評分支持。Selection只有27項達到最低病人／來源支持。
E29共同更新和E27無bridge對照重用已核對產物，沒有重訓或多給預算。

新增`bound_start`為原主幹搭配同一binding器官輸入的零bridge起點；
舊`frozen`使用native pooling，兩者不能混為同一基線。保留兩者，不改舊結果。

## 結果

下表為病人等權、target等權的normalized MAE，越低越好，不是準確率百分比。
不同角色的target支持不同，只能在同一列做配對比較。

| 角色 | 相同輸入起點 | 只學bridge | E29共同更新 | 已訓練無bridge |
|---|---:|---:|---:|---:|
| fit，47項 | 0.753682 | 0.747468 | 0.701781 | 0.705812 |
| selection，46項 | 0.876270 | 0.876812 | 0.876522 | 0.875949 |
| same-source，44項 | 0.781413 | 0.776376 | 0.779152 | 0.778867 |

Selection強raw對照仍為0.853856。只學bridge未追上，七個預定比較及完整screen均0。
固定的delayed／body控制selection均值為0.876337／0.877210；同來源為
0.777808／0.775485。不能看見結果後選最好的控制當正式模型。

相對相同輸入零bridge起點：

- Fit有39/47項均值改善，另8項未改善，不能說每個target都已學會。
- Selection有24/46項均值改善；18項在fit有益、到selection變成無增益。
- Same-source有29/44項均值改善；12項仍由fit有益轉為無增益。
- 這些是描述性符號計數，不是顯著通過／失敗數。BUN的fit增益僅約4.9e-7，
  被如實納入符號計數，但不能稱實質進步；沒有事後設定排除閾值。

## 新證據改變了什麼

**主幹共同移動不是這些反轉的必要條件；單靠凍結它不是充分修法。**
即使主幹完全不變，修正層仍可在訓練資料減少誤差、在新角色反而增加誤差。
因此不能再將全部退步解釋為「state/readout被一起訓練壞」。

但不能反過來說共同更新一定有害或器官資訊全無用：

- 同來源平均上，固定版本反而優於共同更新；不是所有增益都消失。
- BUN在selection的共同更新0.419631，比固定0.448811好；glucose為
  0.709293對0.720482，Hgb為0.486692對0.498449。禁止全域改用固定版本。
- Phosphate固定修正對起點在fit改善約0.005442，selection惡化0.007642，
  same-source改善0.002516。這個具體例子保留來源／支持差異的可能責任。
- Creatinine固定版selection病人等權較起點低0.000866，但來源等權反而高
  0.003125；不能只看一種加權就宣稱泛化成功。
- 同來源平均較好的body控制也是反例，不能把均值改善直接歸因器官内容。

已縮小的是「凍結主幹即可解決」這項假設，不是找到唯一全域根因。
不穩定条件關係、病人／觀測支持與有限fit預算仍未完全拆清。
這是一個相同起點的optimization-scope比較，joint的可訓練參數較多，
全域gradient clipping的有效影響也不同，不能當純粹單因素的訓練因果估計。

## 核對與限制

- 正式86599及獨立核對85808正常exit0，未中斷／重啟訓練。
- 全部15保存模型reload後，三角色正常forward與獨立顯式事件序列逐筆差0。
- 原encoder／state／readout逐tensor與起點相同；保存控制預測欄完全未改。
- 獨立核對器以scalar/fSum重算病人／來源等權誤差，以固定seed另算paired
  bootstrap與單側t檢定，重驗53×7家族門檻；最大數值差8.882e-15。
- 20項新舊測試通過，包含梯度、固定权重、正向screen、缺支持與篡改拒絕。
- 全量顯式神經回放在trainer內完成；獨立verifier核對保存權重／列／統計，
  不是再做第二次完整神經回放。原assessment中的verification=false保留歷史。
- Same-source多數人曾參與原encoder訓練，支持不均且部分很小；fold也不是
  五個獨立複驗。全部是已查看development資料，不是新外部lockbox。
- 本次是當下遮蔽恢復，不是future forecast、個人化回放或因果模型。

## 決策

不promote、不改現役權重／runtime／區間。關閉此固定bridge-only版本，
不在同一selection掃epoch、lr或seed。完整泛化修法仍未交付。
下一項研究必須有新機制對照，區分修正的條件關係與觀測支持，
不能把E33開關消融或本次凍結主幹再次列為未做，也不能只新增另一個讀取頭。

## 證據入口

- [固定協議](RECONSTRUCTION_FROZEN_BRIDGE_PROTOCOL_20261005.md)
- [全部評分](artifacts/native_frozen_bridge_assessment_20261005.json)
- [獨立核對與描述性診斷](artifacts/native_frozen_bridge_verification_20261005.json)
- `scripts/train_frozen_bridge.py`、`scripts/verify_frozen_bridge.py`
- 私有逐筆預測及權重留在本機data，不提交病人資料。
