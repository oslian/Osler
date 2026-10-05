# E31：增量適應的退步不只出現在新來源

## 結論與判斷變化

E30無法分開「換病人」與「換來源」。E31固定原E29模型，補上同來源、
未參與E29兩遍適應的病人角色。原本22項fit增益在selection反轉的指標中：

| 新角色的描述性結果 | 指標數 |
|---|---:|
| 同來源亦未保留病人等權平均增益 | 13 |
| 同來源保留平均增益 | 8 |
| 無支持 | 1 |

**來源切換不是這13项樣本均值反轉的必要條件。**
但這不是13項顯著退步、不是因果歸因，也不是已證明過擬合。
同來源病人組成、不同target的支持量、原encoder暴露與抽樣變異仍未分離。
8項微小平均增益不等於通過任何新gate，不拿它們挑模型或改參數。

## 資料角色與支持

- 五fold、5120次查詢，1613位不同病人；重複fold不是獨立複驗。
- 53項完整盤點、44項可評分，其餘保留unsupported。
- 每fold排除所有E29適應病人，不只是排除同target；只選該fold適應來源。
- 5105/5120次查詢的病人曾出現在原epoch10更新紀錄；只可稱E29適應未見。
- 剩餘15次查詢也使用原fit pool的scaler，不能稱完全未接觸資料。
- 每位病人固定一個context、沿舊hash規則選人；沒有按結果挑選。
- 強raw reader已在此pool擬合，故不拿它當本角色的未見病人對照。

## 原22項反轉的完整清單

以下delta是normalized MAE的「immediate減unbridged」，負值較好。
不是實際生理單位、準確率百分比或新統計閘門。

| Target | 不同病人 | 來源群 | Delta |
|---|---:|---:|---:|
| alkaline_phos | 27 | 11 | +0.001743 |
| alt | 36 | 12 | +0.004134 |
| amylase | 5 | 4 | +0.000849 |
| ast | 37 | 13 | -0.003332 |
| bnp | 2 | 2 | +0.002985 |
| chloride | 55 | 17 | -0.000258 |
| cpk | 12 | 8 | +0.003514 |
| free_t4 | 3 | 2 | -0.000314 |
| hematocrit | 71 | 13 | -0.000217 |
| lipase | 8 | 4 | +0.000269 |
| map | 75 | 14 | -0.000165 |
| myoglobin | 0 | 0 | unsupported |
| o2sat | 58 | 12 | +0.000825 |
| phosphate | 31 | 11 | +0.005298 |
| potassium | 62 | 14 | +0.000200 |
| ptt | 23 | 12 | -0.000450 |
| temperature | 73 | 13 | +0.001667 |
| total_protein | 36 | 11 | -0.001702 |
| triglycerides | 17 | 9 | +0.001878 |
| troponin_i | 13 | 7 | -0.000597 |
| uric_acid | 2 | 2 | +0.014518 |
| wbc | 54 | 11 | +0.001267 |

44項病人等權target-macro MAE為：immediate 0.779152、delayed 0.779174、
body控制0.778997、unbridged 0.778867、frozen 0.781810。
不能直接與E29不同角色／target支持的macro相減來計算泛化損失。

## 核對結果

- 原推論29146與獨立核對14534均正常退出；沒有重訓原模型。
- 重新從fit feature SQLite選人、重建target context與原訓練暴露標記。
- 三種bridge使用獨立顯式更新序列；unbridged與frozen使用native API重播。
- 五fold所有保存預測差皆0；均值獨立加總最大差8.882e-16。
- 原E29權重、E31依賴hash、私有預測、資料角色、支持數与反轉摘要核對通過。
- 16項相關回歸通過，包含摘要篡改、重複身份、非有限值及病人等權反例。
- 原assessment不覆寫；獨立verification保存核對範圍與綁定。

## 尚未解決與下一個判別點

本次補上的是診斷對照，不是模型修復。E29的0完整screen不撤回。
對這些同來源反轉，下一個應核對的是**同fold、同target、共同來源內的
相對誤差與来源組成**；目前的macro不能區分這兩者。
只對有共同支持的部分做配對角色分解，缺支持明列，不能跨fold製造共同來源。
分解也不會自動證明某個病人的生理機制；要有可知輸入的機制證據及固定干預，
才有理由重新訓練。不得因這份結果掃epoch、bridge上限或依結果挑target。

這仍是遮蔽當下重建，不是future forecast、自學習、治療因果或新外部lockbox。
Runtime、權重、區間、promotion狀態均未改變；完整泛化修法仍未完成。

## 原始證據

- [固定協議](RECONSTRUCTION_ASSIMILATION_TRANSFER_PROTOCOL_20261005.md)
- [完整aggregate](artifacts/native_assimilation_transfer_20261005.json)
- [獨立核對](artifacts/native_assimilation_transfer_verification_20261005.json)
- `scripts/probe_assimilation_transfer.py`
- `scripts/verify_assimilation_transfer.py`
- `tests/test_assimilation_transfer.py`
- `tests/test_assimilation_transfer_verification.py`
