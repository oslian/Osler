# 修正洩漏後重新評估開發資料庫（2026-10-23）

Codex 修正順序第 2 步。凍結的部署形式（v8 的 recipe 與線上設定：affine_0.005、DtACI 20、θ0 0.15、eICU 網路共享）不變；資料與重播換成修正後版本：

- 資料：所有錨點保留並記錄量測時間差（MIMIC v5、NWICU v5、Zigong v2、Emory v2、`eicu_harmonized_20261022`）。
- 重播：共用部署時鐘、結果在 anchor + h + |量測偏移| 才可用、暖機住院不計分、選擇／校正病人分離。
- 5 個 MIMIC 抽樣，≥ 80% 通過才算穩健；舊結果檔保留為歷史。
- Emory 已於 2026-10-20 計分一次，此後視為開發資料（不再是 lockbox）。

結果檔：`site_adaptive_reeval_{eICU,NWICU,Zigong,Emory}_20261022.json`；新舊比較：`site_adaptive_reeval_comparison_20261023.json`。

## 穩健格數（凍結方法 point_and_dtaci_adaptive）

| 資料庫 | 可評估格 | 舊（有洩漏） | 新 |
|---|---|---|---|
| eICU | 53 | 45 | **9** |
| NWICU | 44 | 37 | 34 |
| Zigong | 21 | 16 | 15 |
| Emory（非獨立） | 43 | 42 | 38 |

各方法穩健格數（新）：

| 資料庫 | static v2 | 固定步長追蹤 | 點校正 + 固定步長 | 點校正 + DtACI（凍結） |
|---|---|---|---|---|
| eICU | 12 | 29 | 33 | 9 |
| NWICU | 32 | 33 | 36 | 34 |
| Zigong | 0 | 12 | 15 | 15 |
| Emory | 21 | 35 | 41 | 38 |

## 主要發現

1. **點預測大致存活。** eICU 46/53 格點預測仍穩健（DtACI 方法下）；NWICU 43/44、Zigong 16/21、Emory 40/43。失去的格子絕大多數是區間 gate，不是點預測。
2. **eICU 的崩落來自網路共享 DtACI 在真實延遲下發散（實作缺陷，不是資料結論）。** potassium@24h：網路層對數乘數中位 57、最高 399；每發出一個區間時平均有 1,184 個結果尚未解出。DtACI 以「發出當時」的門檻判斷漏涵蓋；成批解出的過期漏涵蓋訊號連續推高共享乘數。舊的洩漏重播在換住院時就解出結果，回饋幾乎即時，所以沒有暴露這個問題。同格改用各醫院自己的 DtACI：中位半寬 0.81（正常）。
3. **單一醫院資料庫的損失集中在 48h（與部分 24h）：** 區間過寬（覆蓋 0.94–0.97）。長時距結果晚到，追蹤器在串流結束前來不及從 θ0 = 0.15 的起始放寬收回。θ0 是在洩漏重播下選的。
4. eICU 點預測未通過的格子（如 potassium@6h）多半是醫院軸 CI 上界剛好跨 0；醫院等權 gate 依研究者決定維持不變。

## 結論

修正洩漏後，點預測的跨院泛化大致成立；區間的線上調適設定（DtACI 延遲回饋、θ0、網路共享）是在洩漏重播下選出的，需要在修正後的重播下重新開發。這些結果全部是開發資料上的結果，不是獨立驗證。

## 修正 DtACI 延遲回饋後再評估（2026-10-23，第 2b 步）

唯一改動：DtACI 在結果解出時以各專家「目前」的門檻判斷漏涵蓋（線上分位數追蹤），不再用發出當時的門檻。其他凍結設定（affine_0.005、DtACI 20、θ0 0.15、eICU 網路共享）不變，也沒有重選。結果檔：`site_adaptive_reeval_{db}_dtacifix_20261023.json`；三組對照：`site_adaptive_reeval_comparison_dtacifix_20261023.json`。

| 資料庫 | 可評估格 | 舊（有洩漏） | 修洩漏 | 修洩漏 + DtACI | 點預測穩健 | 區間穩健 |
|---|---|---|---|---|---|---|
| eICU | 53 | 45 | 9 | **40** | 46 | 44 |
| NWICU | 44 | 37 | 34 | **38** | 43 | 38 |
| Zigong | 21 | 16 | 15 | **14** | 16 | 17 |
| Emory（非獨立） | 43 | 42 | 38 | **41** | 40 | 43 |
| 合計 | 161 | 140 | 96 | **133** | 145 | 142 |

相對舊結果仍失去的格子：

- 點預測（CI 上界碰到或跨過 0；DtACI 修正不影響點預測，修洩漏後就已失去）：eICU glucose_poc@1h、platelets@3h、potassium@3h（醫院軸）；Zigong、Emory creatinine@48h（病人軸，贏 persistence／plain ridge 的幅度約 0.00–0.04 mg/dL）。
- 區間（現在是**涵蓋不足**，接近門檻）：eICU glucose@6h（醫院等權 0.84–0.90）、hematocrit@12h、potassium@6h；NWICU potassium@48h（照護單位 0.86–0.90）；Zigong bicarbonate@12h（照護單位 0.86–0.88）。
- 相對修洩漏版，DtACI 修正後新失去的：eICU platelets@1h（persistence 備援格）、urine_output@1h（病人 0.87、醫院 0.86）、Zigong bicarbonate@12h，全部是涵蓋不足。
- 相對舊結果新增：eICU urine_output@3h；NWICU platelets@1h、sodium@24h。

觀察：

1. 網路共享 DtACI 的發散修好了：eICU 從 9 回到 40。
2. 修洩漏版裡單一醫院 48h 的過寬（涵蓋 0.94–0.97）也消失了：Emory 區間 43/43 穩健。原先歸因於 θ0，實際主因是延遲回饋的更新規則。
3. 剩下的區間損失方向反轉成涵蓋不足，而且都貼近門檻；目前設定是在洩漏重播下選的，現在的偏差是小幅、系統性偏窄。
4. 點預測的損失與 DtACI 無關，是修洩漏後真實存在的邊緣格。

這些仍是開發資料上的結果，不是獨立驗證。線上設定是否在修正後重播下重選，由研究者決定；重選沿用原本預先宣告的開發程序（點校正與 DtACI：MIMIC 偽醫院；θ0：eICU/NWICU/Zigong 留一資料庫；網路共享：eICU 醫院對半），規則須在看結果前寫定，不能朝本表的數字調整。

## 在修正後重播下重選線上設定（2026-10-23～24）

協定：`online_config_reselection_protocol_20261023.json`（規則與候選沿用原本預先宣告的，未修改）。對照檔：`site_adaptive_reselected_comparison_20261024.json`。

| 步驟 | 舊 | 重選 | 依據 |
|---|---|---|---|
| 0 固定步長追蹤器（MIMIC） | weight_site_unit_0.02 | 不變 | 86/94 串流（舊 85） |
| 1 線上點校正（MIMIC） | affine_0.005 | **none** | 完整通過 84 vs 83；affine MAE 比 0.994 |
| 2 DtACI 學習率（MIMIC） | 20 | **5** | 166/188 vs 163 |
| 3 recipe（eICU/NWICU/Zigong） | — | 53 格改 7 格 | 5 格 MPR→hurdle；glucose@1h 等 2 格→persistence |
| 4 θ0（留一資料庫） | 0.15 | **0.10** | held-out 93 vs θ0=0 的 86 |
| 5 網路共享（eICU 醫院對半） | network_1.0 | 不變 | held-out 72 vs 71 |

穩健格數（point_and_dtaci_adaptive）：

| 資料庫 | 舊（洩漏） | 修洩漏 | 修 DtACI | 重選 | 重選：點穩健 | 重選：區間穩健 |
|---|---|---|---|---|---|---|
| eICU | 45 | 9 | 40 | **43** | 45 | 47 |
| NWICU | 37 | 34 | 38 | **36** | 41 | 36 |
| Zigong | 16 | 15 | 14 | **12** | 13 | 20 |
| Emory | 42 | 38 | 41 | **38** | 35 | 43 |
| 合計 | 140 | 96 | 133 | **129** | 134（修 DtACI 時 145） | 146（修 DtACI 時 142） |

發現：

1. 區間變好（142 → 146）：DtACI 5 與 θ0 0.10 修掉了大部分涵蓋不足。
2. 點預測變差（145 → 134），原因是第 1 步不再採用線上點校正：相對修 DtACI 版，161 格中 148 格 MAE 變差，各庫中位 +1.2%～+1.9%；Emory map@1h、platelets@3h、Zigong potassium@24h、sodium@24h 因此不再贏 persistence。
3. 第 1 步的決定只差 MIMIC 94 條串流中的 1 條完整通過；四個外部資料庫一致指向線上點校正有幫助。這是看到開發資料庫結果後才得知的，任何依此修改規則的做法都必須揭露為事後修改。

## 事後修正：點校正改回 affine_0.005 後重跑第 2–6 步（2026-10-24）

研究者決定（事後、已揭露）：`site_adaptive_point_amendment_20261024.json`；協定 `amendments` 一節。第 2–5 步規則不變。

| 步驟 | v8 | 第一輪重選 | 修正後 |
|---|---|---|---|
| 1 點校正 | affine_0.005 | none | **affine_0.005**（事後修正） |
| 2 DtACI 學習率 | 20 | 5 | **5**（164/188，與 1 同分，較窄） |
| 3 recipe | — | 改 7 格 | **相對 v8 改 3 格** |
| 4 θ0 | 0.15 | 0.10 | **0（不採用）**：held-out 88 vs θ0=0 的 88，未嚴格勝出 |
| 5 網路共享 | network_1.0 | 不變 | **不變**：held-out 70 vs 63 |

穩健格數（point_and_dtaci_adaptive）：

| 資料庫 | 舊（洩漏） | 修洩漏 | 修 DtACI | 第一輪重選 | **修正後** |
|---|---|---|---|---|---|
| eICU | 45 | 9 | 40 | 43 | **40** |
| NWICU | 37 | 34 | 38 | 36 | **36** |
| Zigong | 16 | 15 | 14 | 12 | **15** |
| Emory | 42 | 38 | 41 | 38 | **41** |
| 合計 | 140 | 96 | 133 | 129 | **132** |

相對修 DtACI 版：Zigong 多 bicarbonate@12h；NWICU 少 bun@24h、sodium@24h（通過比例都由 0.8 掉到 0.6，邊緣格）；eICU、Emory 總數相同。結果檔：`site_adaptive_amended_{db}_20261024.json`。

結論：在修正後重播下，重選出的部署形式與 v8 + DtACI 修正幾乎相同（132 vs 133），差別只有 DtACI 學習率 20 → 5、θ0 0.15 → 0、3 格 recipe。四個開發資料庫對第 1 步的事後決定是樣本內，只有 SICdb/HiRID 能驗證。

## JEPA gate：第 0 號 MIMIC 抽樣（2026-10-25，第 3 步）

規則 `jepa_gate_rule_20261022.json`（2026-10-24 改指向修正後線上設定）；錨點 `four_db_jepa_examples_harmonized_anchor_20261025.parquet`（recipes 20261024）；7 seeds × 5 個配對模型；測試庫 eICU/NWICU/Zigong（JEPA 事件樣本不含 Emory）。結果：`jepa_gate_sample0_20261025.json`。只有 1 個抽樣，不能晉升（需 5 個抽樣、≥ 80%）。

執行中修正：第一次啟動在預測階段 OOM（10 次預測各保留每列約 2 KB 的耦合診斷、匯出時複製整張 1,069 欄表）；改為只有 full 模型測試預測保留診斷、匯出只取 6 個欄位。6,000 住院實測：該階段記憶體增量 16 → 2 GB；錨點不變；同 seed 重跑本來就有約 0.02（數值約 72）的執行緒非決定性差異。

7/7 seeds 通過的格（8/90）：eICU bicarbonate@48h、heart_rate@6h、map@6h、potassium@1h；NWICU bicarbonate@48h、map@6h、potassium@1h；Zigong bicarbonate@48h。

| 資料庫 | 格數 | 7/7 | 5–6/7 | 0/7 |
|---|---|---|---|---|
| eICU | 36 | 4 | 11 | 11 |
| NWICU | 36 | 3 | 8 | 12 |
| Zigong | 18 | 1 | 8 | 6 |

- 有選出路徑時，JEPA 贏凍結錨點 424/460 個 seed-格：表徵本身有用。
- 主要失敗：選擇集上 organ_route 沒有贏 system_only（無路徑可選，170/630）；選出較深路徑後在測試集輸給配對消融（organ_route 96、system_only 105）；區間 46。
- creatinine 幾乎全滅（72/126 無路徑可選），器官層對 creatinine 沒有可分辨的增益。
- 不穩定主要在 seed 之間：每個 seed 通過 29–45 格，但同一格 7 個 seed 都過的只有 8 格。

## JEPA gate：五個 MIMIC 抽樣（2026-10-27，第 3 步完成）

Protocol：`jepa_mimic_samples_protocol_20261025.json`（看過第 0 號結果後、建第 1–4 號資料前寫下；gate rule 未改）。
結果：`jepa_gate_samples0to4_20261025.json`。第 1–4 號抽樣訓練皆一次完成（無 OOM）。

**穩定 cell（5 個抽樣中 ≥4 個 7/7 seeds 通過）：4 個，全部是 5/5**

| cell | anchor MAE | JEPA 改善 | 主要路徑 | 最差 CI 上界（vs anchor） |
|---|---|---|---|---|
| eICU:potassium@1h | 0.377 | 0.048（12.6%） | body_mediated | patient −0.045 / hospital −0.048 |
| NWICU:potassium@1h | 0.354 | 0.031（8.8%） | body_mediated | patient −0.033 |
| eICU:bicarbonate@48h | 3.166 | 0.198（6.3%） | body_mediated（偶爾 organ_route） | patient −0.102 / hospital −0.080 |
| NWICU:bicarbonate@48h | 2.924 | 0.159（5.4%） | body_mediated（偶爾 organ_route） | patient −0.056 |

- 兩個穩定 target×horizon 的 anchor 都是 persistence（`potassium@1h`、`bicarbonate@48h` 不在 `deployment_recipes_20261024.json`）。有正式 expert recipe（ridge）的 cell 沒有一個穩定：`eICU:map@3h` 2/5、`eICU:heart_rate@6h` 2/5、`eICU:map@6h` 1/5、`NWICU:map@6h` 1/5。
- Zigong 沒有穩定 cell：`Zigong:bicarbonate@48h` 3/5（第 3、4 號 6/7）。
- 每個抽樣 7/7 cell 數：8、8、6、4、6。第 0 號的 8 個中有 4 個沒有重現。
- 失敗原因（每抽樣 630 seed-cells）：沒有選到路徑 170–230；輸給 organ_route 88–114；輸給 system_only 68–127；輸給 anchor 25–80；interval 35–46。有選到路徑時，贏 anchor 的比例 82–94%。
- `agents_complete_gate: false`（time axis 未評估）；`causal_claim_allowed: false`。
