# E23：直接加入器官 state 沒有完成未知值推論修復

## 結論

**這次具體的直接拼接修法沒有通過。**
目前遮蔽重建頭確實只讀body；但把同一條持續狀態軌跡的器官分量直接
接入固定讀取器，並沒有穩定增加留出病人的推論能力。
這不是已證明「器官沒有資訊」，也不是共享人體架構不可能成功。

不同於E19–22的已知內容讀回，本次真值是被完整遮掉target及其歷史／兄弟後
的當下量測值。沒有用記得舊值來代替未知值推論，也不是future forecast。

## 固定對照與支持量

同E21五fold、凍結epoch10、原2048 fit／1024 selection contexts。
全53項盤點，28項至少一fold有足夠支持；131個target-fold、1310讀取器，
4569查詢／1851不同selection病人。每target-fold的fit病人30–106，
中位75；selection 12–54，中位34。其餘25項不填造結果。
同人多fold先平均loss，不能將131組當獨立研究複驗。

五種輸入：body、body+organs、body+打亂器官、body+body隨機線性展開、raw。
三種增廣相同維度；process159一致。隨機展開不增加body以外的資訊，
但等維度不保證有效統計容量完全相同。Shuffle按masked target、資料角色分開，
不使用另一角色病人或label；沒有按source component配對，故不能隔離所有流程因素。
Ridge100與HGB100無調參；八對照固定53×8校正、病人與來源組paired bootstrap。

## 結果

下表為normalized target-macro MAE，不是百分比準確率。

| 表示 | Ridge病人等權 | HGB病人等權 | HGB來源組等權 |
|---|---:|---:|---:|
| Body | 1.014640 | 0.783467 | 0.937894 |
| Body + organs | 1.148590 | 0.789964 | 0.940866 |
| Body + shuffled organs | 1.152000 | 0.785817 | 0.939807 |
| Body + body線性展開 | 1.122101 | 0.779098 | 0.925987 |
| Raw | 1.356537 | 0.799944 | 0.947003 |

八個比較都0項通過固定screen，沒有新增promotion。
原native神經讀取的病人macro是0.715713，來源組0.842483，亦優於新讀取器。
它原本使用較完整的訓練資料與共享多target學習；本輪新reader只在抽樣fit池擬合。
**因此不能用這個較弱的小樣本raw對照，推翻先前完整樣本raw較強的結果。**

## 第一次退步在哪裡

| 描述性target-fold比較 | Ridge | HGB |
|---|---:|---:|
| 加器官後fit誤差降低 | 131/131 | 110/131 |
| 加器官後selection誤差降低 | 31/131 | 62/131 |
| Fit改善但selection反而惡化 | 100/131 | 56/131 |

- 已確認：本輪Ridge的失敗不在於無法擬合訓練資料；主要已觀察反轉發生在selection。
- 已確認：器官提取與原生replay一致，不是本輪把器官state抽錯。
- 合理推測：有限fit支持加大量器官座標，容易把樣本內關係擬合得更好卻無法轉移。
  與過擬合／條件關係不穩相容，但沒有用資料量對照證明唯一原因。
- 未確定：完整支持量下的受約束器官讀取是否有效、哪些訊號在state更新時被犧牲。
  不能把有限reader失敗當作資訊不存在或不可逆丟失的證明。

## 工程核對與修正

- 原訓練執行65691完成五fold擬合後，在joint screen彙總漏寫對照reader前綴而exit1。
  原已綁定訓練script與權重保持不變，作為實驗來源紀錄；正式結果由獨立
  `scripts/score_organ_compartments.py` 修正彙總並核對，不重訓、不修改門檻。
- 獨立評分53339正常結束：1310保存reader的fit／selection預測重現，655scalers核對，
  350檔bindings；原body／native誤差0。
- `scripts/verify_organ_compartment_replay.py` 最初有str/Path型別錯誤，未進入state比較；
  修正後23608正常結束。用原生replay_observations獨立重播15360個fit/selection context，
  body、organs、native point全部逐值精確相等。
- 7項新測試通過，包括完整state路徑、非法時間、合法shuffle、相同process、病人等權，
  以及彙總鍵名與全控制交集回歸；合併既有相關測試共39次測試執行通過。
  測試通過不等於能力通過。

## 決策

關閉這個bounded直接拼接＋固定reader版本，不掃相同資料的維度／alpha／seed。
本輪没有證明直接讀取器官能修好未知推論，更沒有修好未來預測。
下一研究若重開器官讀取，必須解決本輪較弱fit支持／多座標的混淆，且保留
已訓練native強對照；不能再用更弱的新reader失敗替整個state下結論。
現役權重、registry、區間、serving均未改，完整泛化修法仍未交付。

## 原始證據

- `RECONSTRUCTION_ORGAN_READOUT_PROTOCOL_20261003.md`
- `artifacts/native_reconstruction_organ_compartments_fold0_20261003.json` 至fold4
- `artifacts/native_reconstruction_organ_compartments_assessment_20261003.json`
- `artifacts/native_reconstruction_organ_compartments_verification_20261003.json`
- `artifacts/native_reconstruction_organ_compartments_replay_20261003.json`

病人列、latent與reader權重只在本機忽略目錄。上述均為既有development資料，
不是未接觸外部驗證；causal／clinical／runtime promotion均不允許。
