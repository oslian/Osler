# 事件型預測在洩漏修正後的重新評估（開發庫，2026-10-28）

這份補上 2026-10-22 修正順序第 2 步漏掉的部分：當時只重新評估了連續值格，二元事件格沒有。v10 protocol 的 `out_of_scope` 也記錄了這一點。

## 範圍與方法

- **格：** `hypoxemia_event@1h`、`hypoxemia_event@3h`。
  - 事件定義：t + h 時的 SpO₂ ≤ 92%。
  - 這是 catalog 裡全部的二元格。
- **模型：** 沿用不變。
  - logistic regression（C = 0.20），特徵是 o2sat 的 measurement-pure 子集；
  - 在 MIMIC 校準集上選擇向盛行率收縮的 shrinkage；
  - 不做任何線上更新。
- **Gate：** 沿用不變（`multi_sample_external_audit.py`，已被 v10 雜湊凍結，沒有修改）。
  - Brier 分數必須顯著贏過兩個基準：
    - 沿用目前 SpO₂ 的 0/1 判斷；
    - MIMIC 盛行率。
  - 判斷層級：eICU 看病人和醫院兩層；單一醫院的庫只看病人層。
  - 穩定的定義：5 個 MIMIC 樣本中至少 4 個通過。
- **資料：** 都是修正後的版本。
  - MIMIC v5；
  - eICU `eicu_harmonized_20261022`；
  - NWICU v5、Zigong v2、Emory v2。
  - 歷史特徵由完整 anchor 串流建立，之後才依標籤篩選（`normalize_cohort`）。
- **跟洩漏修正的關係：**
  - Codex 找到的洩漏裡，只有「歷史特徵在篩標籤後才建」會影響這兩格，現在已經修好。
  - 線上更新、day-one 權重和 JEPA anchor 這三項，這兩格都沒有用到。
- **程式：**
  - `binary_event_reeval_20261028.sh` → `multi_sample_{db}_binary_leakfix_20261028.json`
  - `binary_event_metrics_report.py`：用同一個凍結的 `evaluate_event_cell` 重算 Brier 分數。它的通過旗標必須和稽核結果一致，否則直接報錯。輸出為 `binary_event_metrics_{db}_leakfix_20261028.json`。

## 結果

| 庫 | 格 | 列 / 病人 / 醫院 | 事件率 | Brier（模型） | 沿用目前判斷 | MIMIC 盛行率 | 5 樣本通過 |
|---|---|---|---|---|---|---|---|
| eICU | @1h | 15,025 / 1,204 / 140 | 10.5% | 0.0789 | 0.1193 | 0.0949 | 5/5 |
| eICU | @3h | 15,014 / 1,027 / 131 | 11.5% | 0.0892 | 0.1415 | 0.1029 | 5/5 |
| NWICU | @1h | 15,020 / 2,161 / 1 | 10.2% | 0.0801 | 0.1215 | 0.0923 | 5/5 |
| NWICU | @3h | 15,002 / 2,026 / 1 | 10.7% | 0.0863 | 0.1356 | 0.0963 | 5/5 |
| Emory | @1h | 15,006 / 2,973 / 1 | 5.4% | 0.0458 | 0.0694 | 0.0513 | 5/5 |
| Emory | @3h | 15,006 / 2,960 / 1 | 5.9% | 0.0498 | 0.0768 | 0.0555 | 5/5 |
| Zigong | 兩格 | — | — | — | — | — | 不可評估：沒有 SpO₂ |

Brier 都是 5 個樣本的中位數。shrinkage 依樣本落在 0.65–1.0 之間。

**和修正前比較：** 修正前用 v4 資料，eICU、NWICU、Emory 都是 2/2 穩定；修正後結論不變。

## 判讀

- **Brier skill 對 MIMIC 盛行率是 10–17%，對「沿用目前判斷」是 34–37%。**
  - 「沿用目前判斷」是硬性的 0/1 預測，在 Brier 下本來就吃虧，所以 34–37% 偏高。
  - 比較有資訊量的是對盛行率的 10–17%。
- **Gate 沒有檢查的部分：**
  - 校準：校準斜率與截距；
  - 鑑別度：AUROC；
  - 單一醫院資料庫的病房層級。
- Emory 的事件率只有 MIMIC 的一半左右，模型仍然贏過 MIMIC 盛行率。不過這次沒有計算它在 Emory 上的校準。
- **這些都是開發庫的 in-sample 結果。** eICU、NWICU、Emory 之前都用過，不是獨立證據。

## 和 lockbox 的關係

- v10 的 `scored_task_types` 只有 `continuous_value`，ledger 步驟是 `continuous_<db>`。
- SICdb 和 HiRID 在 v10 只評過 `o2sat@12h`。1h 和 3h 的未來 SpO₂ 雖然在 extraction parquet 裡，但沒有被任何評分程式讀過，也沒有被看過。
- 所以下一版 protocol 只要把範圍限定在二元格，並用獨立的 ledger 步驟（例如 `binary_<db>`），就能在 SICdb 和 HiRID 上各評一次。
- 要不要這樣做、gate 要不要補上校準檢查，都需要使用者決定，而且必須在讀任何標籤之前凍結。

`causal_claim_allowed: false`；`independent_validation: false`。

## 校準檢查（使用者選 B，2026-10-28）

- **方法：** logistic recalibration（Cox 1958；Van Calster 2016、2019）。
  - 校準斜率：logit(p) 的係數。
  - 校準截距 CITL：以 logit(p) 為 offset 的截距。
  - O/E 與 AUROC 只報告，不納入通過條件。
- **信賴區間：** 依 point gate 的層級，用 cluster bootstrap 200 次。eICU 包含病人和醫院兩層，其他庫只有病人層。
- **預測值重建：** 程式依凍結的 `evaluate_event` 重建預測值，Brier 必須和前面的報告完全一致（差 ≤ 1e-12）。
- **程式：** `binary_event_calibration.py`，測試 4 個全部通過（`test_binary_event_calibration.py`）。輸出為 `binary_event_calibration_{db}_20261028.json`。
- **事前提出的候選規則：**
  - 單一樣本通過：斜率落在 [0.8, 1.2]，而且每一層 CITL 的 95% CI 都包含 0；
  - 一格穩定：5 個樣本中至少 4 個通過。

### 結果（5 個 MIMIC 樣本的範圍）

| 庫 | 格 | AUROC | O/E | CITL | 斜率 | 通過樣本 |
|---|---|---|---|---|---|---|
| eICU | @1h | 0.79–0.80 | 1.10–1.21 | +0.14 至 +0.29 | 0.76–0.93 | 0/5 |
| eICU | @3h | 0.76–0.78 | 1.20–1.28 | +0.25 至 +0.32 | 0.84–1.05 | 0/5 |
| NWICU | @1h | 0.80–0.81 | 1.19–1.44 | +0.22 至 +0.48 | 0.89–1.13 | 0/5 |
| NWICU | @3h | 0.75–0.78 | 1.24–1.58 | +0.27 至 +0.55 | 1.02–1.27 | 0/5 |
| Emory | @1h | 0.80–0.83 | 0.81–1.20 | −0.25 至 +0.23 | 0.98–1.23 | 1/5 |
| Emory | @3h | 0.74–0.79 | 0.90–1.08 | −0.11 至 +0.09 | 0.94–1.40 | 2/5 |

### 判讀

- **鑑別度不錯**：AUROC 0.74–0.83。
- **eICU 和 NWICU 的風險被系統性低估**：觀察到的事件數比預測多 10–58%。
- **Emory 隨訓練樣本兩邊擺動**：同一批 Emory 病人，平均預測機率在不同 MIMIC 樣本之間從 0.045 到 0.066。也就是說，MIMIC 訓練樣本本身就會讓截距晃動，包括 MIMIC 盛行率和 shrinkage（0.65–1.0）。
- **在候選規則下，三個庫都是 0/2 穩定。** 原本的 Brier gate 會通過，是因為「比盛行率好」不需要機率準確。
- **「CITL 的 CI 包含 0」對 15,000 列的靜態模型幾乎不可能達到**，CI 寬度只有大約 ±0.07。外部驗證文獻也指出，換院後 calibration-in-the-large 通常會偏移，標準處理方式是在新院做截距或斜率更新（Steyerberg 的模型更新；Debray 2015）。
- **結論：** 這兩格的靜態機率**不能當作準確的風險值使用**。如果要通過校準檢查，需要像連續值格一樣，在新院用已揭曉的結果做線上校正，例如 online Platt scaling（Gupta & Ramdas 2023）。這屬於新的開發工作，不能直接凍結。

## 線上校準的 MIMIC 開發（使用者選 B，2026-10-28）

- **方法：** online Platt scaling（Gupta & Ramdas 2023，附錄 Algorithm 1），實作在 `osler_jepa/online_platt.py`。
  - 重新校準後的機率是 sigmoid(a·logit(p) + b)，初值 (a, b) = (1, 0)。
  - 用 Online Newton Step 學習 (a, b)：γ = 0.1，A0 = 100·I，限制 ‖(a, b)‖ ≤ 100。這些全是論文的設定，沒有調參。
  - 結果只有在共享部署時鐘走過揭曉時間之後，才會拿來更新。
- **Gate：** `osler_jepa/event_calibration.py`。
  - 在看到任何線上校準結果之前就已固定：CITL ∈ [−0.2, 0.2]，斜率 ∈ [0.8, 1.25]，都是點估計。
  - 這個容許範圍是在靜態檢查的結果出來之後才定的，屬於事後決定，在此揭露。
- **測試：** `test_online_platt.py` 5 個，全部通過。
- **開發腳本：** `site_adaptive_event_dev.py`，只用 MIMIC。
  - 情境：S1 年代、S2 心臟加護病房。
  - MIMIC 盛行率這個比較基準也套用同樣的線上校準，這樣 Brier gate 仍然是同條件比較。
- **事前宣告的採用規則：** OPS 必須比 static 多通過至少一個資料流，而且 Brier 比值的中位數 ≤ 1，才採用。

### 結果（`site_adaptive_event_dev_20261028.json`）

| 格 / 情境 | static：CITL / 斜率 | OPS：CITL / 斜率 | Brier static → OPS |
|---|---|---|---|
| @1h S1 年代 | −0.014 / 0.90 | 0.003 / 1.00 | 0.063 → 0.062 |
| @1h S2 心臟加護 | −0.054 / 0.90 | −0.010 / 0.99 | 0.064 → 0.063 |
| @3h S1 年代 | −0.044 / 0.99 | 0.005 / 0.99 | 0.067 → 0.067 |
| @3h S2 心臟加護 | −0.158 / 1.21 | 0.001 / 0.98 | 0.063 → 0.063 |

- 兩種方法都是 4/4 個資料流通過。OPS 的 Brier 比值中位數是 0.998。
- **依事前規則，結果是 static。** MIMIC 內部的模擬轉移太溫和，static 已經全部通過，這套規則沒辦法分出兩者的差別。
- OPS 讓 4 個資料流的 |CITL| 都更接近 0（最大 0.010），斜率也落在 0.98–1.00，Brier 沒有變差。
- **要不要事後修改採用規則**，由使用者決定。如果修改，需要揭露，而且開發庫的結果對這個選擇來說就變成 in-sample。

## 開發庫評估：OPS（事後修正，2026-10-28）

- **修正紀錄：** `site_adaptive_event_amendment_20261028.json`。由使用者在看到 MIMIC 開發結果與靜態校準檢查之後、看到任何開發庫上的 OPS 結果之前決定。
- **程式與輸出：**
  - 程式：`site_adaptive_event_audit.py`，由 `site_adaptive_event_audit_20261028.sh` 執行。
  - 輸出：`site_adaptive_event_{eICU,NWICU,Emory}_20261028.json`。
- **設定：** 每家醫院各自是一條部署資料流，每條最多 60,000 列，跑 5 個 MIMIC 樣本。static 在同一批計分列上評分。

| 庫 | 格 | 醫院數 | static 通過樣本 | OPS 通過樣本 | OPS 的 CITL | OPS 的斜率 | OPS 的 O/E |
|---|---|---|---|---|---|---|---|
| eICU | @1h | 153 | 3/5 | **5/5** | −0.017 至 0.012 | 0.89–0.92 | 0.99–1.01 |
| eICU | @3h | 159 | 0/5 | **5/5** | −0.006 至 0.026 | 0.89–0.96 | 1.00–1.02 |
| NWICU | @1h | 1 | 0/5 | **5/5** | −0.001 至 0.003 | 0.99–1.00 | 1.00 |
| NWICU | @3h | 1 | 0/5 | **5/5** | −0.003 至 0.003 | 0.99–1.00 | 1.00 |
| Emory | @1h | 1 | 3/5 | **5/5** | −0.003 至 0.004 | 0.99–1.00 | 1.00 |
| Emory | @3h | 1 | 2/5 | **5/5** | −0.003 至 0.004 | 0.99–1.00 | 1.00 |

- **穩定格數：** static 是 0/2，三個庫都一樣；OPS 是 2/2，三個庫都一樣。
- **Brier：** OPS 每個樣本的 Brier 都通過，比較基準包括 persistence，以及同樣經過 OPS、等於追蹤各院事件率的盛行率。
  - 每位病人平均的 Brier 差：
    - eICU：−0.021 至 −0.033；
    - NWICU：−0.009 至 −0.014；
    - Emory：−0.003 至 −0.008。
  - Emory 的差距最小，因為它的事件率只有約 5.5%。

**限制：**
- 三個開發庫對這次的事後選擇都是 in-sample。
- 校準是整個資料流合併計算的，沒有逐院檢查。eICU 的斜率（0.89–0.96）比單院資料庫低，可能是因為小醫院自己的結果很少、校準調整得慢。
- OPS 只學整體的 (a, b)，不處理病房層級的差異。
- 下一步需要使用者決定：凍結只涵蓋事件格的下一版 protocol，在 SICdb 和 HiRID 上各評一次（ledger 步驟 `binary_<db>`）。

## Codex pre-freeze review and v2 rerun (2026-10-29)

- **Codex review:** the verdict was "freeze after fixes"; all 7 findings have been fixed. See `CODEX_REVIEW_EVENT_PREFREEZE_20261029.md`.
- **Rerun with the fixed code:** `site_adaptive_event_audit_20261029.sh` produces `site_adaptive_event_{db}_20261029.json` (schema v2).
  - Pass/fail results are identical to v1.
  - CITL differs only at the 1e-13 level, caused by the new root-finding bracket.
- **Diagnostics (OPS, not gated):**

| DB | Cell | Patient bootstrap CITL CI (sample 0) | First-third slope | Last-third slope | Estimable hospitals / outside tolerance |
|---|---|---|---|---|---|
| eICU | @1h | −0.050 to 0.042 | 0.75–0.83 | 0.92–0.96 | 53–55 / 15–19 |
| eICU | @3h | −0.025 to 0.055 | 0.77–0.86 | 0.93–0.97 | 49–54 / 16–24 |
| NWICU | @1h | −0.039 to 0.038 | 0.98–0.99 | 0.97–1.02 | single site |
| NWICU | @3h | −0.042 to 0.030 | 0.98–1.00 | 0.98–1.01 | single site |
| Emory | @1h | −0.049 to 0.036 | 0.98–1.00 | 0.98–1.02 | single site |
| Emory | @3h | −0.048 to 0.040 | 0.97–1.02 | 0.98–1.02 | single site |

- **Care-unit CITL (sample 0, OPS):**
  - NWICU: −0.66 to 0.37;
  - Emory: −0.24 to 0.14;
  - eICU: −0.29 to 0.13.

  OPS learns only a site-level (a, b). It does not correct care-unit differences.
- **Draft v11:** `ADAPTIVE_EVENT_LOCKBOX_PROTOCOL_V11.md`. Builder: `build_event_lockbox_protocol_v11.py` (dry run: 52 code files + 2 training files; covers every file loaded by the scoring program and the addendum tool). **Not frozen yet.**
