# E33：部分退步定位到已訓練的修正路徑

## 做了什麼

固定全部15個E29模型，五fold、三臂、三個既有資料角色，共20,480次查詢。
只在模型副本清零bridge投影；其餘參數、觀測、遮蔽與時間不變，不訓練。
53項完整盤點；fit有47項、selection46項、same-source44項有支持。
這是當下遮蔽重建，不是未來預測；數字為normalized MAE，不是準確率。

## 新定位

immediate候選的病人等權、target等權平均誤差：

| 角色 | 原on | 同權重關bridge | 原無bridge對照 |
|---|---:|---:|---:|
| fit（47項） | 0.701781 | 0.706335 | 0.705812 |
| selection（46項） | 0.876522 | 0.875705 | 0.875949 |
| same-source（44項） | 0.779152 | 0.778731 | 0.778867 |

同一角色內才是配對比較；不同角色的绝對數字不能直接比較難度。
selection的raw對照仍為0.853856，關掉bridge仍未追上強raw。

固定其他權重後，immediate bridge在selection的26/46項、same-source的26/44項
使病人等權平均誤差較off高；其中25與24項在fit原本有益，到了新角色變有害。
這是描述性方向計數，**不是26項統計顯著失敗，也不是新增通過格數**。

每個角色將相對原無bridge對照的誤差差分成：

    on - unbridged = (on - off) + (off - unbridged)

其中每個名稱代表該預測的絕對誤差，而非預測值本身。

| immediate平均項 | fit | selection | same-source |
|---|---:|---:|---:|
| bridge推論效應（on-off） | -0.004554 | +0.000818 | +0.000421 |
| 其餘共同適應權重（off-unbridged） | +0.000523 | -0.000245 | -0.000136 |

正值代表較差。平均上，新角色的其他權重改變反而稍有幫助，bridge蓋過了它。
這將「E29哪條路徑導致部分退步」定位得更具體，不能再只籠統說domain shift。
但不能說已找到唯一的訓練根因：關bridge可能離開共同訓練分布，
第二項也不是獨立訓練因果效應。第一項含歷史修正的所有後續recurrent影響。

## 為什麼不能全部關掉

- selection關bridge後，仍有22/46項平均輸原無bridge模型，沒有全面修復。
- Hgb的bridge在selection有益（-0.001991），但其餘權重較差（+0.001793），兩者抵銷。
- Creatinine在selection的bridge有益（-0.001991），same-source卻略有害（+0.000121），其他權重在兩者皆較差。
- BUN在selection的bridge有益（-0.002015），same-source有害（+0.003700）；不能只用全體平均決定部署。
- 相同反轉不限器官輸入：delayed在selection26/46項on較off差，body控制28/46項。
- Patient/component等權下，immediate bridge的符號在selection有7項不同、same-source有2項不同。平均增益不足以取代來源穩健性。
- body控制在same-source的病人macro bridge項反而是-0.000085，否定「所有修正一定有害」的說法。

## 核對與限制

正式85498與獨立彙總核對19343正常exit0。
三臂、三角色、全部fold的原on預測重現及off與native回放差皆0。
模型副本只有投影被清零；完整回放前後其他權重／buffer逐項核對不變。
從私有保存列用fsum獨立重算病人／來源等權、三項誤差差與支持量，
最大roundoff3.553e-15；原預測欄位逐列未改，17項新舊測試通過。
回放核對在audit內完成；獨立verifier核對保存列與彙總，不冒充另一次完整神經回放。

同來源角色多數曾參與原epoch10訓練；各target支持不均且部分很小。
重複fold不是獨立複驗，全部都是已查看development資料，沒有新lockbox。
fit到query的兩項變化未再做來源組成校正，不混用E32的共同支持分解。
本次沒有新顯著性screen、promotion、權重發布或runtime／區間改動。

## 下一個修復假設，不是修復成果

已定位的局部問題是：共同適應後的bridge，其訓練增益未穩定轉移。
尚未分清是bridge學到不穩定條件關係，或它與state/readout共同移動後互相補償。
若繼續，先查重「同一已訓練無bridge起點，凍結主幹只學bridge」和
「相同起點／預算共同更新」的對照；必須保留delayed/body等容量控制與原強對照。
不能將本次較有利的off事後挑為正式模型，也不靠放寬gate宣告解決。
這一訓練對照尚未執行；若兩種都無法轉移，就否決共同移動作為充分解釋。

## 證據

- [固定協議](RECONSTRUCTION_BRIDGE_ABLATION_PROTOCOL_20261005.md)
- [全角色結果](artifacts/native_trained_bridge_ablation_20261005.json)
- [獨立彙總核對與角色變化](artifacts/native_trained_bridge_ablation_verification_20261005.json)
- `scripts/audit_trained_bridge.py`、`scripts/verify_trained_bridge.py`
