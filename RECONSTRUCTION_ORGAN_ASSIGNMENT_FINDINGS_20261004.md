# E26：器官內的數值與身份配對在平均聚合被消除

## 已確認的新證據

這次不是另一個讀取頭的負結果。`native_organ_observations.py` 的局部器官
proposal 存在可證明的交換不變性：

```text
organ = mean(phi(value, mask, age) + variable_embedding + module_embedding)
```

同一器官內觀測集合固定時，交換兩個變數的 `(value, mask, age)`，其數值
編碼的總和不變，身份編碼的總和也不變。因此器官 proposal 不知道哪個數值
屬於哪個變數。這裡交換的是 normalized 數值，不是把不同單位的檢验任意互換
當成合理病人；這是表示可辨識性的反例。

**首次可定位操作是 pooling，而不是最後的評分門檻或外部資料轉移。**
這是局部路徑的資訊合併，不再只是「某個有限讀取器讀不好」的證據。

## 已訓練權重核對

固定 E25 五fold映射，實際為四組不同的 epoch10 權重，fold2 重用 fold0。
53變數中的全部224對同器官變數，各測三組數值與兩種觀測集合，共1344案／映射。
float64、固定1e-10絕對容差；原器官公式独立重算最大差4.44e-16。

| 路徑 | 每組1344案可區分數值交換的數量 |
|---|---:|
| 原器官 proposal | 0 |
| 原器官更新後 state | 0 |
| 原器官演化6h後 state | 0 |
| 原 body proposal | 1344 |
| 候選非線性身份綁定 proposal | 1344 |
| 候選更新後／演化6h state | 各1344 |

原器官6h後最大差1.32e-15，屬roundoff；候選6h後最小差9.43e-5。
所有模型參數／buffers執行前後完全相同。與E25共同的7項來源綁定hash一致。
這不是五次獨立臨床驗證，不計病人準確率、p值或promotion。

## 真實訓練輸入的接觸機會

使用既有E19全部五fold fitting contexts，先套原target與兄弟變數遮蔽，
未依預測誤差選樣本，未使用selection或新外部結果。

- 10,240次context回放、3,094位不重複病人；fold間有大量重疊。
- 337,712個事件組出現次數，不是不重複原始事件數。
- 175,658次有同器官兩項以上觀測，其中173,525次有不同數值或年齡。
- 9,855／10,240次context有至少一次這種非平凡配對（96.24%）。
- 包括 lungs、kidneys、liver、bone_marrow等器官；分類沿現有canonical schema，
  不代表每項測量只由所映射器官生成。

**這是機制接觸率，不是錯誤比例或MAE歸因。** 不涵蓋全部訓練人口，也不證明
所有實際互換都是生理可行的。既有source cache、角色、scaling、context與payload
hash核對通過，僅輸出aggregate，沒有病人ID／數值／時刻。

## 已試的結構修復與限制

研究候選把身份與數值先結合，再聚合：

```text
organ = mean(SiLU(phi(value, mask, age) + variable_embedding + module_embedding))
```

沿用現有PyTorch運算，沒有增加可學參數。所有測試配對都不再碰撞。
這遵循 [Deep Sets](https://arxiv.org/abs/1703.06114) 的基本模式：先表示完整元素，
再對元素集合聚合。完整觀測換排列应不變；只換數值、不換變數身份應能區分。
單一SiLU並不保證任意輸入可逆，也不保證是最佳表示。

- 修復候選只在診斷脚本，不接到現役runtime，也不覆寫舊checkpoint。
- 已訓練的body路徑全部可區分這次交換，故不能把E26說成共享body丟失全部資訊，
  或宣稱E25的全部缺口已被解釋。
- 原器官層沒有body feedback，且此次起始state相同，因此其後演化不能重建
  已被合併的配對。若先前歷史不同、其他輸入不同，不能外推為同一結果。
- 候選改了器官輸入分布，不能用舊下游權重的未校準數字直接部署。
- 真實誤差修復尚未測：需要同角色、同訓練預算的原版／binding版訓練，保留
  強raw對照、等容量控制及原驗收，不以合成分辨能力代替當下重建或future gate。

## 驗證與決策

13個相關測試通過，涵蓋碰撞與候選分離、原生演化、完整觀測排列不變、相同值
控制、缺失NaN與實測零、未觀測器官不被update、真實支持計數條件。
未觀測器官的update已正確隔離，這次原先疑問未獲支持；查碼進一步找到的是
同器官內部的值／身份分離，而不是所有缺失器官都被覆寫。

決策：H2取得一個新的**局部確定機制**，其準度貢獻尚未量化。完整根因與
泛化修法仍未交付。Runtime／權重／區間／門檻保持不變，不promote。

## 可重現入口

- `RECONSTRUCTION_ORGAN_ASSIGNMENT_PROTOCOL_20261004.md`
- `RECONSTRUCTION_ORGAN_ASSIGNMENT_SUPPORT_20261004.md`
- `python -m scripts.audit_organ_assignment`
- `python -m scripts.audit_organ_assignment_support`
- `artifacts/native_organ_assignment_20261004.json`
- `artifacts/native_organ_assignment_support_20261004.json`

兩個audit要求新的輸出目的檔，避免覆寫已綁定證據；重現時應在乾淨工作副本
執行，不刪除原研究產物。
