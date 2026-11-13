# 自適應模型一次性驗證 protocol v10（2026-10-27 凍結）

機器可驗證版本：`adaptive_model_lockbox_protocol_v10_20261027.json`（sha256 `4b43aa2c…`）。

- 雜湊值涵蓋 62 個程式／設定檔，以及 6 個 MIMIC-IV v5 訓練資料檔。
- 取代 v9（`docs/research/protocols/ADAPTIVE_MODEL_LOCKBOX_PROTOCOL_V9_20261027.md`）。v9 下沒有認領過任何步驟，也沒有評分過任何資料庫。
- v9 的其餘內容照舊：問題、部署形式、範圍、gate、主要終點、判讀規則、開發庫參考值都相同。

## 與 v9 的唯一差異（Codex 第二輪 #6）

`site_adaptive_external_audit.py` 的 `lockbox_step`：

| | 評分前要求 addendum 登記的 horizon 檔案 |
|---|---|
| v9 | 6 個 horizon 檔案（1/3/6/12/24/48h）只要有 1 個 |
| v10 | 6 個全部，且全部沒改過 |

- 檢查在認領一次性步驟 `continuous_<db>` 之前完成。
- 缺少檔案時，雜湊檢查會先報錯，不會再把那幾格當成「不可評估」。
- 部署形式、cohort 和分數都不變，所以開發參考值仍然有效：eICU 40、NWICU 36、Zigong 15、Emory 41。
- 新測試用暫存資料夾驗證三件事：
  - 只登記 1 個檔案時拒絕評分；
  - 6 個都登記時才認領；
  - 第二次評分會被拒絕。

## 揭露的 replay 性質（Codex 第二輪 #3–#5，照原設計凍結）

開發參考值是用同一套 replay 算出來的，所以這三點不修，只揭露：

1. **到達順序**：住院依隨機（固定 seed）的 Poisson 順序到達。
   - 沒有跨病人的日曆時間。
   - 同一病人較晚的入院可能先到。
   - 這個部署形式沒有病人層級的自適應，歷史特徵只在同一次住院內計算。
2. **只放有標籤的 anchor**：replay 的資料流只包含有合理標籤的 anchor。
   - 模擬到達率和暖機名單取決於「未來有沒有量到」，從不取決於量到的數值。
   - 沒有標籤的 anchor 本來就不能更新狀態。
3. **回饋順序**：兩次預測之間到期的結果，依預測順序更新，不依解出時間的順序。Codex 的例子差約 0.0002。

另外修正了 replay 的描述：結果的可用時間是 anchor + h + |標籤偏移|，這是保守的截止時間，不是量測時間。

## 評分（每個資料庫一次）

`site_adaptive_external_audit.py --database <db> --external-template <addendum 中的模板> --protocol adaptive_model_lockbox_protocol_v10_20261027.json --output site_adaptive_lockbox_<db>_v10.json`，其他選項一律用預設。

`causal_claim_allowed: false`；`clinical_promotion_allowed: false`。
