# E35：多數退步不是把正確修正放得太大

## 範圍與核對

固定E34三種bridge通道、五fold、fit／selection／same-source三角色的全部
20,480次查詢，合計61,440個候選輸出；53項完整盤點，47／46／44項有支持。
沒有訓練、推論新資料、挑target、選alpha或改模型。對照是同binding輸入的
零bridge起點`bound_start`，不是native-pooling的舊`frozen`。

父assessment、verification與各fold輸出hash通過；原MAE與支持量逐项重現。
每列向量算式另用分段scalar計算，病人／來源等權另用fsum重算，最大差
3.553e-15。20項新舊測試通過，含零殘差、反向、跨真值、權重、非法身份、
有限差分與PyTorch SmoothL1核對。這是獨立算式核對，不是再次神經回放。
30項本輪bindings保存於aggregate；正式92832正常exit0。

## 核心結果

定義固定輸出線`起點 + alpha * (修正後預測 - 起點)`。
在起點的MAE右導數已正時，由凸性可知，任何統一正alpha都不能改善
這批樣本、這個target、這套固定權重的平均MAE。這不等於縮放recurrent
bridge權重，也不限制從輸入重新學一條不同修正方向。

以下是病人等權的描述性計數，**不是顯著失敗／promotion數**：

| 角色／通道 | 平均MAE退步 | 右導數已正 | 右導數負、全幅因跨真值退步 | SmoothL1改善但MAE退步 |
|---|---:|---:|---:|---:|
| fit／immediate | 8 | 7 | 1 | 6 |
| selection／immediate | 22 | 20 | 2 | 4 |
| same-source／immediate | 15 | 11 | 4 | 4 |
| selection／delayed | 23 | 19 | 4 | 4 |
| selection／body | 19 | 15 | 4 | 2 |

E34的18項fit有益→selection無益中，17項右導數已正，只有myoglobin屬於
右導數負而全幅有害；同來源12項反轉中為9項／3項。全部通道、角色及來源
等權數字均在JSON，不只保留immediate或有利結果。Selection immediate
來源等權有25項平均退步，其中20項右導數正、5項跨真值，仍非顯著計數。

### 修正方向與幅度不能只看總平均

Selection immediate的病人等權target-macro：

- MAE改變 +0.000541944。
- 起點右導數 -0.002294171，跨真值凸性項 +0.002836114，精確相加得到上述改變。
- Gross harm 0.015143845、gross benefit 0.014601902；93.48%的gross harm
  來自朝遠離真值的方向修正，6.52%來自跨真值。

這些事實不矛盾：總平均右導數仍有益，某個較小統一輸出尺度可能改善
總平均；但不會使那20項各自都勝起點。不能把「總平均可縮幅」包裝成
「所有未過target只差幅度」。本次不利用selection真值選尺度。

### 訓練損失與驗收差異確實存在，但不是唯一原因

SmoothL1(beta=1)和MAE使用同一套權重比較，selection immediate的前者
平均下降0.001308476，後者上升0.000541944。4個MAE退步而SmoothL1改善
的target為CPK、glucose、temperature、troponin I；其餘18個MAE退步
不能用這個有利SmoothL1的端點反例解釋。

| 指標／selection | MAE改變 | MAE右導數 | SmoothL1改變 |
|---|---:|---:|---:|
| BUN | +0.000445539 | +0.000147943 | +0.000295368 |
| glucose | +0.000048758 | +0.000048758 | -0.000179287 |
| phosphate | +0.007642283 | +0.006881807 | +0.003986301 |
| hematocrit | +0.004947207 | +0.002190906 | +0.004498963 |
| hemoglobin | -0.000922249 | -0.008759593 | -0.002265352 |

不要把glucose的極小平均差稱為顯著退步。BUN／phosphate兩種損失都退步，
反駁「只要換MAE就修好全部」；Hgb仍有益，反駁全域刪除器官修正。
這是固定保存端點的損失，不是optimizer沿途平均訓練loss、參數梯度或因果歸因。

## 舊證據與本次新增的區別

- E9早已發現teacher修正多數gross harm方向錯，不能稱本次首次發現一般規律。
- 舊2026-09-20 site-constraint reconciliation已看到SmoothL1／MAE不一致；
  `docs/research/notes/RECONSTRUCTION_MAE_SITE_READOUT_20260920.md`也已測exact MAE讀取修復，
  沒有完整新增通過。不能把換loss重新列為未試通用修法。
- 新增的是：E34直接用真值訓練且固定主幹，仍有逐target正右導數；因此
  這些target的反轉不能僅由teacher偏差、主幹共同移動或全幅跨真值解釋。
  加上同權SmoothL1對照，能分開部分目標取捨與兩種損失都失敗的target。

## 決策與仍未交付

不實作全域輸出縮幅、真值選路或事後放寬gate；不宣稱換loss已修復。
下一項介入必須解釋並檢驗「為什麼合法輸入對應的修正方向在新病人上失效」，
且與既有support selector、局部offset、raw/latent probe、exact-MAE修復區分。
本次沒有定位是哪項輸入、哪個更新或哪種條件關係造成方向失效，完整修法仍未交付。

這是CURRENT masked reconstruction，不是病人的未來上升／下降、治療效果、
個人化或自學習回放。來源與病人支持不均、原encoder暴露和反覆查看development
限制沿用E34。沒有新外部lockbox、promotion、現役權重／區間／runtime變更。

## 產物

- [固定協議](../../../RECONSTRUCTION_BRIDGE_GEOMETRY_PROTOCOL_20261005.md)
- [完整數據與核對](../../../artifacts/native_bridge_geometry_20261005.json)
- `scripts/diagnose_bridge_geometry.py`、`tests/test_bridge_geometry.py`
- [父實驗E34](RECONSTRUCTION_FROZEN_BRIDGE_FINDINGS_20261005.md)
