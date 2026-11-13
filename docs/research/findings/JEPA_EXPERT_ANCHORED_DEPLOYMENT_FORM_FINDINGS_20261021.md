# 三層 JEPA（系統 ↔ 器官 ↔ 人體）以凍結專家為錨點：部署形式比較（2026-10-21，開發診斷）

## 設定

- 資料：MIMIC-IV 5,000 個住院（訓練）＋ eICU 8 月版 canonical 事件範例（測試）；6 個核心目標 × 6 個 horizon。
- 錨點：凍結的部署 recipe（`deployment_recipes_20261020.json`，MIMIC v4 抽樣 0 擬合）；catalog 外的格錨點 = 現值（`build_expert_anchor_columns.py`、`expert_anchor_registry_core6_20261021.json`）。
- 模型：`teacher_anchored_joint_jepa_audit.py` 預設 `adjacent_three_level`，14 個解剖器官，system → organ → body 依序訓練，零初始化殘差；每格路徑依 MIMIC 校準列選定。1 個 seed（7）。
- 部署形式（`jepa_deployment_form_dev.py`）：錨點與 JEPA 各自走凍結的線上點校正 + DtACI（θ₀ 0.15、網路共享）；兩邊都用均勻校準權重（事件資料沒有第一天共變數）。

## 結果（eICU，36 格）

| | 靜態（無上線校正） | 部署形式（校正 + DtACI） |
|---|---|---|
| JEPA / 錨點誤差中位數 | 0.975 | **0.990** |
| JEPA 顯著勝過錨點（病人與醫院兩軸） | 33 / 36 | **32 / 36** |
| 區間 gate 通過 | — | 錨點 36 / 36，JEPA 36 / 36 |

- 靜態優勢大半被上線校正吸收：glucose@1h 0.854 → 0.990、heart_rate@6h 0.916 → 0.989。三層殘差在靜態下學到的有一大部分是跨資料庫偏差，而上線校正本來就會修掉。
- 校正後仍保留的改善：bicarbonate@1–48h 1–6%、potassium@1h 12%、potassium@6h 2.6%、glucose 1–2%、MAP 1–1.4%、HR 0.5–1%、creatinine 約 0.8%。
- 錨點 = persistence 的格（bicarbonate@1/3/6/48h、potassium@1h、creatinine@1/3h）：JEPA 提供了專家格原本沒有的預測能力。
- 未通過：heart_rate@1h、heart_rate@12h、map@12h、map@24h（醫院軸區間跨 0）。

## 判讀

- 統計顯著但幅度小：多數格的絕對改善很小（creatinine 約 0.003 mg/dL、glucose 約 0.5 mg/dL、potassium@6h 約 0.01 mmol/L）；大樣本使微小差異也顯著。臨床上有意義的主要是 bicarbonate 與 potassium@1h。
- 尚未達到 AGENTS.md 的接管條件：只有 1 個 seed；尚未做 7-seed、醫院／病房／時間切分、匹配消融的完整 gate；eICU 事件資料為 8 月版定義（glucose 與 MIMIC 不完全一致）；僅在 eICU 開發庫比較，尚無獨立資料庫。

independent_validation: false；causal_claim_allowed: false。

## 更正：基礎 JEPA 未收斂（2026-10-21）

第一次結果所用的基礎 JEPA 基礎損失 263.6（8 月為 3.74），latent SVD 未通過（有效秩 5.6 → eICU 2.25）。原因：MIMIC 的腫瘤標記極端長尾（AFP 穩健 z² 約 7e8；CA 19-9、CA-125），以及 eICU 8 月版事件資料的游離鈣單位錯誤（mg/dL vs mmol/L）與 lipase 試劑不可比。`prepare_jepa_event_examples.py` 修正後：基礎損失 1.63，有效秩 15.0 → 14.6（比例 0.97），SVD 通過；部署形式結果與修正前幾乎相同（33/36、中位數 0.989）。

## 7-seed gate（`jepa_7seed_gate_rule_20261021.json` 預先宣告；`jepa_7seed_gate_result_20261021.json`）

7 個 seed 都是 MIMIC 訓練 → eICU 測試；每格需 7/7：部署形式下病人與醫院兩軸勝過凍結專家錨點、JEPA 區間 gate 通過、選中路徑勝過 system-only（或本身為 system-only）、SVD 通過。7 個 seed 的 SVD 全部通過。

**通過 22 / 36 格**（JEPA ÷ 錨點，7 個 seed 的範圍）：

| 系統 | 通過的格 | 改善幅度 | 路徑 |
|---|---|---|---|
| 酸鹼（bicarbonate） | 1h、3h、6h、48h | 3–6% | 人體中介 / full / 直接器官 |
| 腎（creatinine） | 1h、3h、6h、24h | 0.6–1.6% | 多為 system-only |
| 血糖 | 1h、3h、6h、12h、24h、48h（全部） | 1–2.4% | 人體中介 / full |
| 心率 | 3h、6h、24h | 0.3–1.4% | full / system-only |
| MAP | 1h、3h、48h | 0.4–1.7% | full / 直接器官 |
| 鉀 | 1h、6h | 11–13%、2.6–3.3% | full |

未通過 14 格：bicarbonate@12h/24h、creatinine@12h/48h、heart_rate@1h/12h/48h、map@6h/12h/24h、potassium@3h/12h/24h/48h（多為部署形式下有幾個 seed 不顯著，或選中路徑未勝過 system-only）。

## 判讀

- 依 AGENTS.md 的完整 gate，三層 JEPA 在 22 格是凍結專家之上的穩定改善；其中 bicarbonate@1/3/6/48h 與 potassium@1h 的錨點原本只是 persistence。
- 多數格的絕對改善小（約 1–2%）；臨床意義較明顯的是 potassium@1h（約 12%）、bicarbonate（3–6%）、potassium@6h（約 3%）。
- 這仍是開發候選：只用 eICU（開發庫、8 月版事件定義）；下一步是 NWICU/Zigong 開發檢查，最後 SICdb/HiRID 一次性驗證；registry 變更需研究者同意。

## 四資料庫開發檢查（2026-10-22；`jepa_7seed_gate_rule_four_db_20261021.json` 預先宣告；`jepa_four_db_gate_result_20261022.json`）

同一個 MIMIC 訓練模型（7 seeds）同時在 eICU、NWICU（5,000 住院）、Zigong（約 3,000 住院，僅檢驗）測試。NWICU／Zigong 事件資料由 `osler_jepa/event_adapters.extract_nwicu_events / extract_zigong_events` 以 10 月統一定義建立。開跑前依各庫中位數檢查再修正 eICU CRP 單位（mg/dL→mg/L），並自 JEPA 移除 BNP（MIMIC 為 NT-proBNP）與 PTH。7 個 seed 基礎損失 1.39–1.65，SVD 全部通過。記憶體不足（exit 137）造成的中斷以 `--resume` 接續，未改變任何設定。

| 資料庫 | 可評估格 | 7/7 通過 |
|---|---|---|
| eICU（重跑，含 CRP 修正） | 36 | 22（與先前 22 格重疊 20 格；新增 map@24h，失去 creatinine@24h、heart_rate@24h） |
| NWICU | 36 | 15 |
| Zigong（僅檢驗目標） | 18 | 7 |

**接管候選 13 格**（eICU 通過，且在 NWICU、Zigong 凡可評估者皆通過；全部也在先前的 eICU-only gate 中通過）：

| 格 | JEPA ÷ 錨點（eICU / NWICU / Zigong，7 seed 範圍） |
|---|---|
| potassium@1h | 0.874–0.883 / 0.911–0.924 / 0.899–0.925 |
| potassium@6h | 0.965–0.973 / 0.965–0.976 / 0.954–0.981 |
| bicarbonate@3h | 0.961–0.966 / 0.965–0.972 / 0.957–0.963 |
| bicarbonate@6h | 0.962–0.967 / 0.953–0.959 / 0.954–0.960 |
| creatinine@1h、@3h | 約 0.99 / 0.995 / 0.99 |
| glucose@3h、6h、12h、24h、48h | 0.974–0.991（eICU、NWICU；Zigong 無 glucose） |
| heart_rate@3h、@6h | 0.988–0.997（eICU、NWICU） |

未成為候選：MAP 全部（NWICU 多數 seed 不顯著）、heart_rate@1h/12h/24h/48h、potassium@3h/12h/24h/48h、bicarbonate@1h（Zigong）、@12h/24h/48h、creatinine@6h（Zigong 5/7）/12h/24h/48h、glucose@1h（NWICU 6/7）。

## 判讀

- 13 格在三個不同的外部開發庫、7 個 seed 都穩定改善凍結專家；跨庫一致，不是 eICU 特有。
- 臨床意義最明確：potassium@1h（約 8–13%）、bicarbonate@3h/6h（約 3–5%）、potassium@6h（約 2–5%）；其餘約 0.5–2.5%。
- 仍是開發候選（eICU、NWICU、Zigong 都已用於開發）；獨立證據需凍結後在 SICdb／HiRID 一次性驗證；接管（registry 變更）需研究者同意。
