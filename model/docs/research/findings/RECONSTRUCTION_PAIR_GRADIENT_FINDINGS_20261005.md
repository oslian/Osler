# E38：減少梯度衝突，仍未修好未知值重建

## 結論與新證據

固定配對PCGrad干預已完成訓練、評分及獨立核對；九個對照的校正screen
和完整交集均為0。**沒有新增promotion，完整泛化修法仍未交付。**

這次不是單純重述「有訊號但不穩」：投影確實大幅減少擬採用方向的局部
有害計數，但Adam實際更新仍可產生有害方向，完整誤差也沒有實質改善。
因此「把原始梯度衝突投影掉，就能修好共享state」不是此固定版本的充分修法。

## 固定比較

- E34相同起點與資料，保留同一body及bridge，只學28,672個投影參數。
- `mean`：兩例梯度平均後AdamW；`pcgrad`：不同target且內積負時先投影再平均。
- 每fold兩遍各2,048例，每遍1,024次更新；兩臂樣本、初始化、更新預算相同。
- 五份fold紀錄、十個模型；20,480次查詢分為fit10,240、selection5,120、
  same-source5,120。不是20,480個獨立病人。重複fold先在病人內聚合。
- 53項完整盤點；fit47、selection46、same-source44項有資料。Selection27項
  滿足至少30病人／3來源組，其餘不因低支持被隱藏。
- 九個強對照包含配對mean、E34 sequential fixed-immediate、E29 joint-immediate、
  fixed-delayed、fixed-body、匹配起點、unbridged、frozen及raw。
- E34 sequential每遍有2,048次更新，不能把與它的差別全部歸因投影；
  **PCGrad對配對mean**才是這次的匹配干預。

## 1. 局部梯度保護不等於實際更新保護

以下加總五fold每臂20,480次「當前樣本梯度與更新方向」檢查，並非
20,480位病人、獨立重複或真實損失增加次數。局部導數也不保證有限步結果。

| 訓練診斷 | 配對mean | PCGrad |
|---|---:|---:|
| 更新次數 | 10,240 | 10,240 |
| 梯度內積負的pair | 5,096 | 5,097 |
| 實施投影的pair | 0 | 4,909 |
| 擬採用負梯度方向的局部有害計數 | 670 | 67 |
| Adam實際參數更新的局部有害計數 | 2,197 | 1,921 |

前者下降90%，後者僅下降約12.56%。兩臂軌跡不同，不能把每次事件視為
逐步同狀態配對；PCGrad同target pair按協議不投影，且計數有數值容差。
不能聲稱它消除了所有任務衝突。

**首次能定位的剩餘限制：** 在這個訓練器內，梯度合成後的方向不等於
Adam最後實際走的方向。動量、逐座標尺度、decay與rounding共同介入；
本輪尚未分離各自責任，不能直接指定其中一個為唯一根因。
這個局部現象也不能單獨解釋全部未見病人的泛化失敗。

## 2. 完整真值誤差幾乎不變

病人等權、target等權的標準化MAE，越小越好；不是臨床單位或準確率：

| 角色 | 配對mean | PCGrad | E34 sequential | 強raw |
|---|---:|---:|---:|---:|
| fit | 0.748914 | 0.748895 | 0.747468 | 未評估 |
| selection | 0.876553 | 0.876528 | 0.876812 | 0.853856 |
| same-source | 0.777108 | 0.777131 | 0.776376 | 未評估 |

Selection相對配對mean僅約0.00275%描述性改善，同來源平均反而略差。
來源組等權結論亦相同：selection0.921718→0.921702，same-source
0.767089→0.767109。沒有顯著或臨床效益宣稱。

PCGrad對mean的均值改善數為fit31/47、selection23/46、same-source23/44。
Fit有改善但query沒有改善的項目，selection與same-source各15項。
不能將所有失敗都歸因外部資料库切換或門檻太嚴。

例：glucose selection MAE 0.720758→0.720656，BUN 0.449020→0.448698。
兩者病人層未校正CI在0以下，但來源組CI跨0且未過預定家族校正；
不是新通過格。Creatinine、hemoglobin等也沒有完整通過，保留全部反例。

## 3. 驗證與限制

- 正式54965與獨立核對60684皆自然exit0，沒有中止重跑或改門檻。
- 24項新舊測試通過，含極小梯度、相反／同向／零梯度、同target不投影、
  凍結主幹、同預算、病人／來源組權重、非法身份及固定screen反例。
- 檢查所有原始控制欄位、真值、角色支持與病人角色互斥，保存主幹逐tensor不變。
- 十模型全部20,480查詢重新載入後以明確state-step helper回放，最大預測差0。
- 獨立scalar/fsum與paired統計重算，最大roundoff1.832978e-13；未更改驗收容差。
- 原產物69bindings及核對74bindings收尾匹配；原assessment保留未核對時快照，
  完成證據以獨立verification為準，不覆寫原訓練紀錄。
- Fold0和fold2兩臂checkpoint皆完全相同，**不是五次獨立複驗**。
- 此干預只適用凍結主幹的bridge、固定兩例配對與預算，不否定所有多任務方法。
- 同來源query亦有原encoder暴露限制；MIMIC／NWICU已反覆使用，無新lockbox。
- 這是當下遮蔽恢復，不是future forecast、自學習回放、因果或完整人體理解。

## 決策與尚欠證據

關閉此固定PCGrad分支，不掃learning rate、配對方式或seed重試。
E37的「有來源相抵」已接上真正干預，但E38顯示局部改善不能直接轉為完整增益。
更新後的判斷是：**只保護原始梯度不足，實際optimizer更新與跨病人轉移仍要分開。**

若再研究更新機制，必須核對同一步驟在合成、動量、預條件化、decay各階段
的方向責任，不能僅換optimizer後把任何均值改善叫根因修復。
E37中自身target來源也有害的部分仍未解决；即使局部更新全部受保護，也
不能預設未知病人的誤差必然下降。新研究仍需獨立固定對照，現役不變。

## 證據入口

- [固定協議](../../../RECONSTRUCTION_PAIR_GRADIENT_PROTOCOL_20261005.md)
- [原assessment](../../../artifacts/native_pair_gradient_20261005.json)
- [獨立核對與診斷](../../../artifacts/native_pair_gradient_verification_20261005.json)
- `scripts/train_pair_gradient.py`、`scripts/verify_pair_gradient.py`
- `tests/test_pair_gradient.py`、`tests/test_pair_gradient_verification.py`
