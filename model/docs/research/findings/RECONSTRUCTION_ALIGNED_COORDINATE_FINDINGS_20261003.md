# 固定共享 state 的觀測對齊座標：完整結果

## 結論與範圍

**47 targets / 5 source folds 已完成，沒有完整通過項目。**
新增對齊座標的平均改善很小，未在預先固定的病人及來源組校正檢定下
勝過原 body HGB、raw HGB 或兩個等寬座標控制。
關閉本次固定線性對齊修復分支，不繼續掃 alpha、維度、seed 或 HGB 預算。
這不是證明所有資訊都丟失，也不是證明所有讀取方式都無效。

本次是遮蔽當下值重建，不是未來 forecast、自學習、因果或 runtime promotion。
沿用已反覆使用的 source development；沒有讀取 outer，也沒有新的獨立 lockbox。
現役權重、registry、calibrator 與服務行為均未改變。

## 與前次不同的地方

固定同一 frozen body64、病人、標籤及原始輸入。僅在 fit 病人上學習
body 到 raw265 摘要的 Ridge 映射，再以 body64 + 解碼座標265 訓練 HGB。
推論不增加 raw 觀測，因此仍是同一份 body 資訊。原 body64 保留，不做降維。
兩個控制同為329維：隨機單位向量座標、以 fit 內配對亂序學習的座標。
三種 heads 同100階段，合計705個；235組映射含470個Ridge擬合。
座標含觀測值及流程欄位，不能稱為生理真值。

## 完整評估

1,797,884 個 query occurrences，720,470 個去重 patient-target，29,580 位病人。
同病人的多fold輸出先平均誤差，再以病人／來源組等權評估，不把fold當獨立複驗。
原基線逐筆與聚合結果一致；已核對hash、完整235組產物、資料角色及有限值。

Normalized target-macro MAE，越低越好；不是百分比準確率：

| 讀取方式 | 病人等權 | 來源組等權 |
|---|---:|---:|
| Native neural | 0.882389 | 0.831593 |
| 原 body HGB | 0.857139 | 0.811241 |
| **觀測對齊 body HGB** | **0.854847** | **0.809938** |
| 隨機座標控制 | 0.856169 | 0.811406 |
| 亂序座標控制 | 0.858780 | 0.813392 |
| 原始觀測 raw HGB | 0.825737 | 0.787947 |

相對原 body，平均改善0.2675%／0.1606%。逐目標平均誤差比原body低者
為41/47病人等權、31/47來源組等權；這些不是顯著通過計數。
相對raw，兩種等權方式均45/47平均更差，並非只卡在統計門檻。

完整固定screen：CI上界小於0及單側p < 0.05/(47*7)，病人與來源組都要過。
CI來自5000次paired bootstrap，p值來自既有單側paired-unit t檢定，兩者不同。
各對照通過數：body 0、raw 0、random 0、shuffled 0、native 12、process 14、population 17。
七個對照的完整交集為0；較弱對照上的勝利不能代替完整screen。

## 第一次不足出現在哪裡

235個target-fold fit比較是描述性統計，彼此非獨立：

| 對照 | 對齊頭fit較好 | 對齊頭fit較差 | 平均fit誤差差值 |
|---|---:|---:|---:|
| 原body HGB | 229 | 6 | -0.012781 |
| raw HGB | 82 | 153 | +0.004309 |
| 隨機座標 | 138 | 97 | +0.000531 |
| 亂序座標 | 202 | 33 | -0.005948 |

- **已確認：** 相對raw的不足在153/235個fit比較已存在，不能全部歸咎
  外部資料分布或驗收規則。相對body的fit改善也不能保證來源組穩定。
- **已確認：** 線性映射可解碼部分其他觀測。235組selection的最新值／
  歷史平均重建MSE比值中位數為0.677674／0.674437，分母是fit均值基線，
  只評有觀測支持的數值；不是被遮target的重建能力或資訊保留百分比。
- **尚未區分：** 有用於目標的資訊是否被壓縮掉、線性反解是否失真、
  HGB如何使用這些座標，以及跨組關係差異各占多少。
- **不支持的完整修法：** 只替同一state加這組可解釋座標，便能彌補raw差距。
  微小均值改善不足以证明觀測對齊有超越等寬控制的穩定作用。

例如creatinine，病人均值對body改善0.008472，來源組改善0.012361，
但來源組p=0.0141未過固定校正要求；對raw卻平均退步0.044244／0.034053。
同一目標也須區分「對body的增益支持不足」和「對raw確實平均較差」。

## 數值與工程稽核

235映射／470scalers／705heads檢查通過。134個scaler含微小負variance，
共1030欄，最低-9.20178e-34；全部在既有浮點常數誤差界內且scale精確為1。
係數有限、原body保持不變、常數解碼欄為0；全部query預測有限。
因此本次sqrt警告不構成NaN預測失敗，亦不需要因警告重訓。
數值測試只能排除這個具體工程原因，不能證明生理有效。

## 決策

不promote，不修改門檻，不重試相同對齊超參數。保留同一共享state方向，
但下一個研究須先查既有中間表示／history消融，定位raw的優勢在何階段
減少；不能只把另一種讀取頭當成根因修復。尚未因此啟動新訓練。

## 可核對產物

- [凍結設計](../../../RECONSTRUCTION_ALIGNED_COORDINATE_PROTOCOL_20261003.md)
- [完整評分](../../../artifacts/native_reconstruction_aligned_coordinates_assessment_20261003.json)
- [數值稽核](../../../artifacts/native_reconstruction_aligned_coordinates_numerics_20261003.json)
- [Fit/selection診斷](../../../artifacts/native_reconstruction_aligned_coordinates_diagnosis_20261003.json)
- 五份 `artifacts/native_reconstruction_aligned_coordinates_fold{0..4}_20261003.json`
- `scripts/train_aligned_coordinates.py`、`scripts/score_aligned_coordinates.py`
- `scripts/audit_aligned_coordinate_numerics.py`、`scripts/summarize_aligned_coordinates.py`

私有逐筆結果與fitted heads留在本機，不納入版本控制。
