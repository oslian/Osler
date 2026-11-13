# E25：完整支持量器官讀取對照

## 結論與判斷變化

**完整實驗與獨立核對完成，但0項通過完整增量比較，不promote。**
這是當下遮蔽重建，不是future forecast、自學習或臨床驗證。

E24指出小樣本讀取器不足以測器官表示。本次補足固定五fold全部fit支持後，
真器官拼接相對body有小幅平均增益，但沒有指標完整勝過等寬body展開，
也沒有指標完整勝過raw。不能再把「完整fit支持尚未測」當成此版本的缺口。
核心結論仍未改變：完整泛化修法尚未找到。

## 固定對照與完成範圍

- 沿[原protocol](../../../RECONSTRUCTION_FULL_ORGAN_PROTOCOL_20261003.md)，不重訓encoder、不調參。
- 五fold、十角色共5,319,744次原始查詢提取完成；53指標盤點，47有支持。
- 相同合法歷史、target及兄弟遮蔽、角色與權重；四個新HGB100臂共940個讀取器。
- 原235個raw讀取器通過資料／權重／預測契約核對後重用，native為固定參考。
- Selection共1,797,884次查詢、29,580位病人、235個target-fold。重複fold先按病人平均，不能當獨立複驗。
- 不支援：ck_mb、crp、crp_hs、ionized_calcium、prealbumin、urine_output；不是六項模型失敗。
- MIMIC／NWICU仍為反覆查看的development資料，不是新外部lockbox。

## 誤差與固定研究閘門

數字是normalized target-macro MAE，不是百分比準確率；越低越好。

| 輸入／讀取器 | 病人等權 | 來源組等權 |
|---|---:|---:|
| Body + 相同流程欄位 | 0.854767 | 0.811346 |
| Body + 真器官 + 相同流程欄位 | 0.849663 | 0.804782 |
| Body + 打亂器官 + 相同流程欄位 | 0.857061 | 0.813491 |
| Body + 等寬body線性展開 + 相同流程欄位 | 0.853201 | 0.808770 |
| Raw完整讀取器 | 0.825737 | 0.787947 |
| 原生神經讀取 | 0.882389 | 0.831593 |

真器官比body平均低0.60%，但比raw平均高2.90%。
47項中39項病人均值勝body、42項勝等寬展開，**0項病人均值勝raw**。
平均改善與完整統計通過必須分開：

| 真器官對照 | 同時通過病人／來源組CI與固定多重比較的指標 |
|---|---|
| Body | hemoglobin、lactate |
| 打亂器官 | hematocrit、hemoglobin、ptt |
| 等寬body展開 | 無 |
| Raw | 無 |
| Native | alt、ast、bun、hematocrit、hemoglobin、inr、platelets、potassium、respiratory_rate |

每個比較仍要求至少30病人、3來源組，兩種95%配對bootstrap CI上界<0，
且兩種p_less<0.05/(53×5)。增量合取與全部對照合取皆為0。
這是固定研究screen，不是release gate；不能只選對native的9項說成promotion。

## 首次失敗能定位到哪裡

235個target-fold中，真器官在fit MAE勝body215組、勝等寬展開129組、勝raw123組。
因此**112組在fit已輸raw**；不能將全部差距歸咎外部資料或最後的顯著性門檻。
Selection跨fold病人聚合後，47項均未勝raw，證明不只是「誤差較低但CI太寬」。
這些是不同聚合層級，不能把123直接稱為123組跨域反轉。
所有五個HGB臂的實際平均葉數中位數均為15，不再是E23弱讀取器的近乎常數擬合。

已排除的是：**只補足fit支持，就足以讓這個固定器官拼接HGB100成為完整修法。**
仍未區分的是：座標可讀性、表示保留、擬合目標及條件關係轉移各負責多少。
這不是器官無資訊、所有架構無效或唯一根因已定位的證明。

## 核對與局部工程修復

全cache身份、key、hash、payload及native parity核對完成；獨立逐事件器官演化
回放仍只有首批1,024筆，不能說全部531萬筆都做過第二套trajectory回放。
正式評分重播fit／selection；獨立verifier另行組裝輸入、重播1,175個selection讀取器，
獨立按病人及來源組聚合，沿用固定統計原語與seed，全部離散判斷精確一致。

首次verifier在32個來源組p值上超過64eps，最大差7.7938e-14；沒有screen差異。
原因是獨立路徑的逐筆浮點加總累積誤差，改用標準庫math.fsum並新增大群組測試後，
完整重跑最大統計差2.9976e-15，原64eps容差未放寬。這是核對器數值修復，
不是模型泛化修復。五份相關套件共41項通過，不將歷次測試數累加。

## 決策與證據

關閉這個固定全支持器官拼接分支，不繼續掃seed、寬度或HGB參數。
重開需新機制證據和能區分原因的固定對照；本報告不授權下一輪訓練。
權重、registry、正式區間及serving沒有變更，完整根因／修法仍未交付。

- [全量提取快照](../../../artifacts/native_reconstruction_full_organs_progress_5319744_20261003.json)
- [完整評分與fit診斷](../../../artifacts/native_reconstruction_full_organ_readout_assessment_20261003.json)
- [獨立核對](../../../artifacts/native_reconstruction_full_organ_readout_verification_20261003.json)
- [操作進度與歷史](../notes/RECONSTRUCTION_FULL_ORGAN_PROGRESS_20261003.md)

原assessment保留當時`independent_statistical_verification_complete:false`，不覆寫歷史；
完成證據由另一份verification綁定其SHA256並回報true。
