# E28：器官最新觀測在同時刻無法直接到達 body 讀取頭

## 已確認的局部限制

E27 三種版本、五個 fold 映射，共15個保存模型均重現：

1. 固定 prior state、body proposal、presence及時間，只改新器官proposal。
2. `update` 後器官state會變，但body state和當下預測完全不變。
3. Autograd顯示該輸出沒有通往這筆器官proposal的計算路徑；body proposal梯度非零。
4. 向後演化1小時後，器官proposal可影響body與輸出，梯度亦非零。

首次限制在 `PersistentPhysiologyStateLayer.update` 的分開更新：body更新不讀
新器官state。器官到body的通道只在 `transition`，但 `advance(0)` 不執行它。
E27當下重建只讀 `model.readout(state.body)`，所以零最後間隔時讀不到這次器官更新。
這是條件性的路徑限制，不是整條器官網路無梯度或所有歷史都沒作用。

## 真實 context 的條件支持

盤點既有E27全部context，不依誤差選人，也沒有重新評分或訓練：

| 角色 | context次數 | 最後事件與anchor同時 | 同時且僅一個事件 |
|---|---:|---:|---:|
| fit | 10,240 | 9,403（91.83%） | 404 |
| selection | 5,120 | 4,634（90.51%） | 282 |

上述零間隔context的最後一筆可見器官proposal，native與binding全部不同。
但是同時刻body-only讀取無法直接接收這個最後器官差異。
這些是跨fold重複查詢次數，不是獨立病人数；fold2原始權重重用fold0。
所有target保留在aggregate中，沒有支持也標為0。

## 什麼判斷改變了

- 原判斷：E27解除器官配對碰撞，訓練後沒有穩定增益，尚不清楚修正怎樣到达讀取頭。
- 新證據：器官支路的最新修正，在大多數既有查詢的同時刻讀取沒有直接路徑。
- 因而不能用E27小增益單獨推論「最新器官配對資訊沒有價值」。
- E27的負結果仍有效：此完整實作與固定訓練沒有通過，沒有新增promotion。

## 不能跨越的界線與修復條件

**相同原始觀測已經進入body encoder旁路。** 這不是90.5%人體資料丟失，
不是90.5%誤差可修好，也未證明新器官通道有超出body的獨立資訊。
本次數值擾動是合成latent的計算路徑測試，不是治療因果或生理反應測試。

E25曾直接把完整器官state給讀取器，仍未勝過全部對照。因此不能把
「直接加器官讀取頭」再次當成保證有效的新方案；E28也不否定E25。

若要測同時刻器官assimilation，必須保持真實時鐘與同一body state，
以等容量的立即／延遲路徑對照和body旁路控制分離作用，再作真值驗證。
不得偷偷加一小時、改future label，或把未校準的器官向量加到現役輸出。
同時刻通路的有效性與端到端準度修復尚未完成；本次沒有替換現役模型。

## 驗證與產物

- 兩項合成回歸通過，包含0/正時間、body旁路、權重不變、缺測target與非法時間拒絕。
- 15保存模型逐一載入並核對hash，五fold原cache/contract由E27既有入口核對。
- 完整程序正常退出；報告僅包含支路敏感度及時間支持量，未輸出病人識別碼。
- [固定protocol](RECONSTRUCTION_ORGAN_REACHABILITY_PROTOCOL_20261004.md)
- [aggregate結果](artifacts/native_organ_reachability_20261004.json)
- `scripts/audit_organ_reachability.py`、`tests/test_organ_reachability.py`

**新增局部機制證據，完整根因及泛化修法仍未解決。**
