# E29：即時器官資訊寫回共享 state，未成為完整修法

## 結論與證據層級

**已完成訓練、評分、15 個保存模型的獨立更新序列重播與統計核對；0 完整研究 screen。**
這是同時刻 target／兄弟變數遮蔽重建，不是未來預測或現役 promotion。
結構上可接通最新器官更新，但本次固定干預没有穩定增加未知值恢復能力。
全域根因與完整泛化修法仍未交付。

新證據改變的判斷：E28 的同時刻器官支路限制，已由「真實誤差貢獻未測」
更新為「此固定即時 assimilation 介入已測，不能解決完整缺口」。
不否定所有器官資訊或 observation-update 方法，也不把這個負結果改寫成成功。

## 實際比較

依凍結 [protocol](../../../RECONSTRUCTION_ORGAN_ASSIMILATION_PROTOCOL_20261004.md)：
同一個 body、原 epoch10 初始化、凍結 event encoder、同 E27 binding proposals，
五 fold 各三臂、固定兩遍訓練，沒有依 selection 選參數。

- 即時：將本次 organ innovation 寫回 body。
- 延遲：下一筆真實觀測才寫回前次 innovation，不在 query 補寫、不假造時間。
- Body 控制：相同參數數目，但以 body innovation 為輸入。
- 無 bridge：E27 相同預算的 binding 適應；另保留 frozen 和完整 fit 的強 raw reader。

Selection：5,120 次重複 fold 查詢、2,048 位病人；53 項完整盤點中 46 項有評分資料，
27 項达到最低 30 病人／3 source components 支持。其餘不刪除、不宣稱可驗證。
下表為 normalized target-macro MAE，越低越好，**不是準確率百分比**。

| 方法 | 病人等權 | Source-component 等權 |
|---|---:|---:|
| 即時 organ innovation | 0.876522 | 0.921248 |
| 延遲 organ innovation | 0.876776 | 0.920984 |
| Body 等參數控制 | 0.877487 | 0.921441 |
| 無 bridge | **0.875949** | **0.920407** |
| Frozen | 0.877001 | 0.922048 |
| 強 raw reader | **0.853856** | **0.871928** |

五種固定比較各自 screen 均為 0，機制增量交集及完整交集皆 0。
即時對延遲的病人均值略好，但 source-component 均值反而較差，不能只報有利欄位。

## 首次失敗位置

以下是描述性均值分類，不是新 gate 或事後 promotion。

| 即時臂相對對照 | Fit 已無均值改善 | Fit 改善、selection 反轉 | 兩者均值改善 |
|---|---:|---:|---:|
| 無 bridge | 0 | 22 | 24 |
| 延遲臂 | 14 | 17 | 15 |
| Body 控制 | 28 | 10 | 8 |
| Frozen | 0 | 23 | 23 |

**因此不能說它完全學不到，也不能說全部只是閘門太嚴。**
相對無 bridge 的 22 項，已觀察到的失敗在 fit→selection；相對 body 控制的
28 項，器官輸入的相對優勢在 fit 內就沒有建立。兩個對照回答不同問題。
13 項相對無 bridge 的病人與 source-component 改善方向不一致。
Raw 沒有這份 bounded fit 角色的保存對照，不能補造 raw 的 fit-gap。

### 投影是否被上限卡住

三臂全部保存 hook 的 `abs(tanh) >= .99` 座標計數為 0。
即時臂 selection 平均絕對 correction 0.002390，最大 0.107346；既定 latent
上限 0.2。**不支持「大量碰到飽和上限所以失敗」的解釋**，但這不能證明
上限／正則化在任何意義下都沒有影響，也不能據此掃 bound。
這些是保存 hook 的彙總，包含零／缺測步驟，不是生理單位或獨立病人數。

### 已排除與仍未知

- 已核對：本次保存推論與獨立更新排程一致、原對照不變，不能把負結果歸因保存模型跑錯。
- 不支持：通道完全沒更新、普遍飽和，以及單靠接通同時刻器官支路就能修好本版本。
- 尚未區分：有限 fitting 支持、冗餘資訊、表示可讀性、訓練取捨及跨來源條件關係的責任。
- 本次沒有證明特定治療／器官關係造成反轉，也沒有證明人體無法預測。

## 核對與限制

- 訓練原程序正常結束，沒有為了結果取消或重啟。
- 完整核對正常 exit0：15 模型各 fit／selection 重播最大預測差全為 0；
  encoder 不變、原對照／hash／固定 paired 統計與 summary 核對通過。
- 最大均值 roundoff 為 1.776e-15；沿既有核對規則，未改科學門檻。
- 26 項新舊回歸測試通過，包含身份缺失拒絕、零／缺測區分、時間與病人隔離、
  保存重播、診斷分類及資料綁定。測試通過不等於準度通過。
- 延遲臂少一次有效 correction，body 控制有效輸入秩較低，因此不是純時間因果比較。
- Raw 使用更多 fitting 支持；它是保留的強基線，不是純架構公平排名。
- 五 fold 重複查詢不能當獨立病人；這是 reused development，不是新外部 lockbox。
- 此實驗的 event encoder 沒有端到端重訓；不得說所有端到端方法都已失敗。

原 assessment 的 `independent_verification_complete: false` 是評分當時快照，
為保留 hash 不覆寫。完成證據是獨立的 verification artifact，不把旧快照當即時狀態。

## 決策與重開條件

關閉此固定橋接方式／預算作為完整修法，不掃同一批 selection 的 seed、bound、epoch。
現役權重、registry、區間與 serving 均不變，`promotion_allowed: false`。
下一次介入必須先指出 **哪種 pre-anchor 條件關係在 fit 成立卻不能轉移**，
或提供先前未涵蓋的表示／目標機制證據；只有「再加容量／多跑幾遍」不構成准入理由。
此要求是待完成診斷，不是已找到新的唯一根因或保證有效的新模型。

## 原始產物

- [原始評分](../../../artifacts/native_organ_assimilation_assessment_20261004.json)
- [完整獨立核對](../../../artifacts/native_organ_assimilation_verification_20261004.json)
- [失敗位置／支持與更新幅度](../../../artifacts/native_organ_assimilation_diagnosis_20261005.json)
- `scripts/organ_assimilation.py`、`scripts/train_organ_assimilation.py`
- `scripts/verify_organ_assimilation.py`、`scripts/diagnose_organ_assimilation.py`

以上僅保存 aggregate；逐筆病人資料及候選權重維持私有。
