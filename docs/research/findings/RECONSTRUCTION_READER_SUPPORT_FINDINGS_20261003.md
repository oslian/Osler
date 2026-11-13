# E24：小樣本讀取器限制了上一輪診斷，並非共享 state 已修好

## 已確認的新證據

E23的raw對照比native差，不能據此推翻早前完整fitting的raw優勢。
本輪在**完全相同**的E23留出查詢上，重用已保存的完整／抽樣raw讀取器。
沒有重新訓練、調參、選擇有利病人或更動模型權重。

五fold、131個target-fold、28有支持指標、4569次查詢／1851位不同病人。
兩者raw265／真值相同，抽樣fit為完整fit子集，fit與selection不重疊；
兩者使用同一HGB100設定。53項完整盤點，其餘25項保留unsupported。

| 方法 | 病人等權 normalized macro MAE | 來源組等權 |
|---|---:|---:|
| E23抽樣fit raw | 0.799944 | 0.947003 |
| E3完整fit raw，同一批E23查詢 | **0.638187** | **0.716334** |
| 原native神經讀取 | 0.715713 | 0.842483 |

完整raw的病人等權macro比抽樣raw低20.22%，比native低10.83%。
這是normalized MAE的相對差，不是臨床準確率提升20個百分點。
27/28指標在病人與來源組兩種均值都勝抽樣raw，20/28都勝native。

## 為什麼相同設定仍不是同樣的有效學習能力

| 實際支持／複雜度 | 抽樣fit | 完整fit |
|---|---:|---:|
| 每target-fold fit病人範圍 | 30–106 | 14,209–29,267 |
| Fit病人中位數 | 75 | 26,325 |
| 每棵樹平均實際葉節點 | 2.785 | 15.000 |

葉節點是結果出來後才做的描述性檢查，沒有加入新驗收門檻。
固定min_samples_leaf=20、max_leaf_nodes=15，在很小的fit池中限制了
可形成的分割；加上額外器官座標不會自動產生足夠的學習支持。
同時較大的fit池也改變來源組成與bounded weights，**沒有獨立證明
只有樣本數造成差異**，也不代表增加任意資料都有效。

## 完整通過與反例

三個預先固定逐target比較均0項通過53×3多重校正screen。
完整raw對抽樣raw有12項兩層未校正CI皆負，對native有6項；
但兩層adjusted p均未完整通過，3項還有最低支持不足。
不能改用未校正CI宣布promotion。

- Creatinine：病人MAE 0.795157 → 0.619772，native 0.686952。
- BUN：0.529335 → 0.459299，仍未勝native 0.446279。
- MAP：完整raw病人均值勝native，但來源組均值仍略差。

因此不是「所有負結果都是樣本數」，也不是「只要用完整fit就全部成功」。
較少的selection支持限制了這次screen；仍需真正足夠的獨立評估，
不能把反覆看過的資料改稱新lockbox。

## 修復了什麼，還沒修復什麼

**修正了診斷對照的判讀：** 同查詢完整raw恢復較強表現，E23弱raw不能
代表全量raw能力；不能用弱小樣本reader的失敗推論器官state無資訊。

**尚未修復模型：** 本輪沒有訓練完整支持量的器官readout，沒有改善
shared state本身、未來預測、自學習、區間或runtime。
先前E21的「已知內容讀回改善但未知推論沒過」亦沒有被推翻。

後續若重開器官readout，必須讓body、body+organs、容量控制與raw使用
相同且足夠的fit支持，保留原native強對照，將fit反轉與selection不確定
分開；不能再靠每target約75人的失敗宣判整個state。
這是後續實驗的必要設計條件，不是本輪已完成的修法。

## 核對

- 主執行43552 exit0：原raw／truth／角色核對、保存預測重現，763個檔案hash檢查。
- 獨立核對48925 exit0：262個保存reader預測精確重現；獨立病人／來源組
  loss聚合核對；paired統計／screens不變，MAE彙總roundoff最大8.88e-16。
- 初次核對32193在回放完成後把混合型assessment交給float-only helper而中止。
  僅修核對器並加入回歸測試；沒有重訓或改動原assessment。
- 12項相關測試通過。這不是模型準確度驗收。
- 現役權重、registry、interval、serving未改；不promote、不宣稱因果或臨床能力。

## 證據

- `RECONSTRUCTION_READER_SUPPORT_PROTOCOL_20261003.md`
- `scripts/audit_reader_fit_support.py`
- `scripts/verify_reader_fit_support.py`
- `tests/test_reader_fit_support.py`
- `artifacts/native_reconstruction_reader_support_20261003.json`
- `artifacts/native_reconstruction_reader_support_verification_20261003.json`

逐筆病人、輸入與預測留在ignore目錄，這裡只有彙總結果。
