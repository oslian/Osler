# E27：器官配對修正沒有帶來穩定的未知值重建增益

## 問題與固定對照

E26已定位同器官加法平均會抹掉變數身份與數值配對，且非線性binding
能區分受測交換。本次檢驗這項局部修正是否真正改善遮蔽重建，不重做E25拼接。

使用既有E19/E21五fold上下文，每fold2,048 fitting與1,024 selection，
相同epoch10初始化、角色、時間、target及兄弟遮罩。fold2重用fold0權重；
不能當五次独立複驗。固定兩遍訓練，只更新state layer與原readout，event JEPA凍結。

三臂相同可學參數量：

- native：`mean(phi(numeric) + identity)`。
- binding：`mean(SiLU(phi(numeric) + identity))`。
- postpool：`SiLU(mean(phi(numeric) + identity))`，控制非線性與尺度改變，仍有配對碰撞。

body事件表示不變。保留未再訓練的frozen版與既有完整fit raw HGB強對照。
Raw的fit支持量較大，不能用此比較宣布所有模型的公平通用排名；三個配對訓練臂
之間才是同樣本、同預算的局部介入。

## 結果

53項盤點，46項有selection資料，27項達最低30病人／3來源組支持量。
5,120次selection查詢、2,048位不重複病人；重複fold先按病人合併。
以下是normalized target-macro MAE，不是準確率百分比。

| 版本 | 病人等權 | 來源組等權 |
|---|---:|---:|
| native重新訓練 | 0.876430 | 0.920825 |
| binding重新訓練 | 0.875949 | 0.920407 |
| postpool重新訓練 | 0.876059 | 0.920504 |
| frozen原版 | 0.877001 | 0.922048 |
| raw完整fit對照 | 0.853856 | 0.871928 |

Binding相對配對native的均值改善約**0.055%**，不是5.5%。
對native、postpool、frozen、raw的固定screen各0項；機制增量及完整合取皆0。
保留所有未支持項，沒有挑target、seed、病房或改門檻。

## 首次失敗位置：不能全部推給外部轉移

由保存fit及selection預測做描述性拆分，未用拆分結果選新模型：

| Binding的對照 | 46項中fit均值改善 | selection均值改善 | fit改善但selection反轉 |
|---|---:|---:|---:|
| 同條件native | 23 | 27 | 9 |
| 同條件postpool | 24 | 24 | 11 |
| 未再訓練frozen | 46 | 26 | 20 |

Fit病人macro為native0.714634、binding0.714835、postpool0.714859、frozen0.761098。
因此：

1. 配對binding相對native的23項在fit就未改善，並非全部跨來源才出錯。
2. 相對frozen，46項fit改善卻有20項selection反轉，額外訓練收益不穩定轉移。
3. 三臂平均差很小，不能把binding的微小均值改善歸因已證明的配對資訊增益。

這些是固定候選的失敗分層，不是唯一全域根因或臨床正常／異常分組的證明。
E26的結構盲點仍成立；但解除局部碰撞不保證資訊能被下游用來推論未知真值。

## 核對器故障與修復

原核對40442在albumin處用整份dictionary逐位比較而失敗。診斷找到143處差異，
**全部只在normalized MAE**，最大1.7763568394002505e-15。
Parquet讀回後逐筆數字完全相同，但DataFrame欄位的記憶排列不同，影響浮點加總；
以原逐欄方式重建DataFrame後，全部53項與原assessment精確相同。
CI、p值、樣本数與screen均沒有差異，原assessment與訓練產物沒有重寫。

修復只在verifier：MAE使用已有E25的64eps相對機器roundoff檢查；
paired統計、支持與決策仍精確比較。這明確改正原版exact-mean核對契約，
**不是放寬模型的科學驗收門檻**。非有限值、實質誤差、p值及screen變更均有拒絕測試。

- 30項相關回歸執行通過（包含重用的測試案例），不代表30項臨床能力。
- 完整核對31688正常exit0；15保存模型在fit及selection的native API重播誤差全部0。
- 凍結encoder完全不變，原raw預測核對通過。
- 全部固定paired統計與summary重算通過，另用Python分組獨立核對MAE。
- 整體roundoff最大1.776e-15，原始實驗與科學決策保持不變。

## 決策與邊界

固定的「凍結event encoder、兩遍state/readout適應、pre-pool SiLU」分支未成修法，
不繼續在同一批selection上掃learning rate、epoch或seed以求過關。
若重開，必須提出先前介入未涵蓋的新機制證據及可反駁對照；
不是把已測版本重新稱為未測，也不是宣稱所有binding架構不可能有效。

尚未完成端到端event encoder binding訓練或全部cohort驗證；未做不等於應直接開跑。
MIMIC／NWICU已查看，不是新外部lockbox。這是當下未知值重建，不是future forecast、
自學習、治療因果或runtime promotion。現役權重、區間及registry不變。
**局部核對器錯誤已修復；完整模型根因與泛化修法仍未交付。**

## 證據

- [固定protocol](../../../RECONSTRUCTION_ORGAN_BINDING_PROTOCOL_20261004.md)
- [原評分](../../../artifacts/native_organ_binding_assessment_20261004.json)
- [完整核對](../../../artifacts/native_organ_binding_verification_20261004.json)
- [fit/selection拆分](../../../artifacts/native_organ_binding_fit_gap_20261004.json)
- [數值故障定位](../../../artifacts/native_organ_binding_statistics_diagnosis_20261004.json)
- `scripts/train_organ_binding.py`、`scripts/verify_organ_binding.py`
- `tests/test_organ_binding.py`

逐筆資料、病人身份、cache及候選權重留在忽略的私有儲存；研究文件僅含aggregate。
