# 專案工程規則

## 強制執行方式

- 開始任何分析、規劃、修改、測試或命令執行前，必須先完整閱讀本檔。
- 若工作目錄內還有更具體的 `AGENTS.md`，也必須一併閱讀並遵守。
- 除非與系統或開發者層級指令衝突，以下規則均為強制要求。

## 核心規則

1. 不保留向後相容。過時的實作、介面或格式直接刪除；不新增相容層、不寫 migration、不保留舊版 fallback。本條針對舊版相容，不得誤刪目前需求明確要求的安全防線，例如 fail-closed 或經驗證的 persistence fallback。
2. 選擇能滿足目前需求的最簡單實作。不要建立預防性抽象、沒有必要的配置層或未被目前需求使用的擴充點。
3. 系統可以有清楚且長期可維護的分層，但實作順序必須先跑通最小端到端版本，再逐層增加能力。不得為了尚未完成的複雜度拆掉已經能正確運作的流程。
4. 元件保持模組化並維持關注點分離。每個模組應有單一清楚責任，跨層依賴必須明確。
5. 優先使用成熟、持續維護且已有實務驗證的函式庫。沒有明確理由，不自行重寫既有成熟能力。
6. 新增套件或自行實作前，先檢查專案目前的依賴、標準函式庫與既有工具是否已能完成需求。不得預設現有依賴做不到。
7. 架構決策以長期維護為準。不接受「先暫時這樣，以後再全部換掉」的方案；最小實作也必須位於正確的長期架構方向上。
8. 設計新功能前，先研究成熟產品、主流函式庫或已驗證系統如何解決相同問題。優先採用已證實有效的模式，不從零發明。

## 任務開始檢查

開始動手前，確認已完成以下事項：

- 已閱讀本檔與適用範圍內的其他 `AGENTS.md`。
- 已閱讀相關程式碼、測試、依賴與現有架構。
- 已確認哪些舊實作應直接刪除，而不是包裝成相容層。
- 已找出可交付的最小端到端路徑。
- 已確認現有依賴是否足以完成需求。
- 已參考成熟實作或已驗證模式；若未採用，已能說明具體原因。
- 已確認方案同時符合目前需求與長期架構。

## 研究記憶（每次任務強制）

- 每次任務先讀唯一帳本「一頁交接」中的簡明摘要，同時核對最近完成的證據ID與待驗證方案的完成證據，並使用其六欄開場／收尾模板；在 `graph.md` 明列本次是舊結論、新增診斷、局部修復還是能力通過。它是既有完整閱讀規則的入口，不是省略原始證據的替代品。若本次只是整理文件，明寫無新研究證據，不自行續跑歷史實驗。
- 固定必讀入口：[根因帳本的一頁交接](ROOT_CAUSE_LEDGER.md#一頁交接回答這不是早已知道了嗎)。每次任務先對照「舊結論／新增證據／局部修復／完整能力驗證」，再依下列規則閱讀與留回執。不得將舊 `graph.md` 的「目前任務」或歷史待辦當成新授權。
- 每次任務（含文件、狀態回報及中斷續接）先完整閱讀唯一根因帳本 `ROOT_CAUSE_LEDGER.md`。先核對「目前交接」、「已做／未做速查」、「必讀決策表」與「一頁交接」，完成第1節「任務前檢查表」；聊天摘要不能代替。必須區分已寫程式、測試完成、訓練完成、獨立核對完成與能力通過。
- 開場依帳本「任務開場回執與交付底線」，在 `graph.md` 記錄實際讀取範圍、最近證據 ID、本次唯一差異、成功標準與未核對事項；不能只寫「已讀」或把未讀文件記為已讀。任務結束更新同一帳本，不建立平行結論檔。
- 行動前列明：最新要求、最接近的既有實驗、已確認／已否決／未區分／未完成、本次唯一差異。歷史「下一步」必須查後續結果；文件任務寫「無新實驗」，不據此重啟研究。
- 若 `graph.md`、state或摘要與完成報告矛盾，先核實版本與原始證據並修正入口。提取計數須附保存快照日期，不冒充即時執行或模型成效；方案、程式完成、實驗完成與能力通過分開記錄。
- 結束依帳本第1節回報新增證據、判斷變化、首次失敗位置、修復範圍及未解事項。沒有新證據就明寫「核心結論未變」，不得再以「有訊號但不穩」或實驗次數當新突破。
- 更新既有帳本條目，不追加另一份同義摘要；保留原始負結果與重開條件，禁止只換名稱、seed或參數重做已關閉分支。
- 任務前必讀帳本「研究交接的責任」：先區分本次是重述舊結論、增加診斷證據、修好局部工程問題，還是通過完整能力驗證；不得用前三者冒充第四者。此規則同樣適用只問狀態的任務。
- 必須核對帳本最新「待驗證方案」：協議、訓練程式或測試檔存在，不代表測試已執行、訓練已完成或能力已通過；缺完成證據時保留待驗證，不沿歷史待辦自動續跑。

- 每個實質行動階段（規劃、修改、訓練、評估、發布）前，先讀取並更新 `graph.md` 的證據、目的、狀態與下一步，再執行；不能把 `.codegraph` 索引或舊聊天摘要當作當前研究結論。
- 每個實質行動階段也先同步並查詢 CodeGraph。若 PATH 找不到，使用已確認的 `/Users/chunyouchang/.local/bin/codegraph` 執行 `sync` 與 `explore`；同步失敗須保留錯誤並報告，不得只寫 Markdown 就聲稱索引已更新。
- 每次開始分析、規劃、修改或測試前，必須先調用 `whole-body-research-memory` skill，並完整閱讀 `PROJECT_STATE.md`。
- 執行研究前，依 `ROOT_CAUSE_LEDGER.md` 的六行模板記錄假設、既有實驗、唯一差異、可反駁結果、固定驗收及禁止宣稱。若所有結果都無法區分原因，先不要訓練。非研究任務標明不改模型／不執行實驗。
- 先核對歷史文件中「未做／下一步」是否已被後續實驗完成或否決；已關閉分支不得只換名稱或參數重做。使用者暫停後，文件整理或自動續接不構成恢復訓練授權。
- 提出或執行新實驗前，必須查閱 `EXPERIMENT_INDEX.md` 與相關 findings/protocol，確認沒有重做已完成或已否決的實驗。
- `.codegraph/` 存在時，理解程式結構、符號與呼叫關係應先使用 CodeGraph；文字與實驗結果查找再使用 `rg`。
- 現役能力數字必須以 `python -m scripts.validate_release` 的即時輸出為準；舊 findings、README 或 manifest 不得凌駕於 validator。
- `candidate_only`、負結果、部分通過與正式 runtime promotion 必須分開陳述。未通過完整 gate 的能力不得描述成現役能力。
- 完成一個實驗時，必須在同一個 commit 更新 `PROJECT_STATE.md` 或 `EXPERIMENT_INDEX.md`，記錄問題、資料角色、方法、結果、決策與 artifact 路徑。
- 不得將逐筆病人資料、病人識別碼或未去識別的原始醫療資料寫入研究記憶或版本控制。

## Whole-Body JEPA 架構決策

- 現役 release 維持已外部驗證的 target router、interval gate 與 persistence safety anchor。
- 舊的「系統層 -> 器官層 -> 人體層」latent 階層已被取代，不得新增或恢復獨立 system latent。資料來源中的 functional system 只負責觀測契約與 provenance，不是模型內的平行生理狀態。
- 全身候選只能有一個持續更新的 body state、明確的 anatomical organ states 與稀疏 organ graph。所有 target 必須由同一條狀態軌跡生成，不得再新增彼此無關的全身平行模型。
- future trajectory 必須沿同一 recurrent state 逐步演化，不得把每個 horizon 當成互不相關的獨立樣本。
- recurrent physiology state 表示病人的絕對內部狀態；JEPA future target 表示未來實測變化。兩者必須經由明確的 state-change readout 連接，禁止直接相加、混合或以 residual gate 掩蓋 latent 語義錯位。
- persistent candidate 直接接受 JEPA future-latent loss；不得以近零 gate 阻斷新狀態路徑的梯度。現役 release 的 persistence fallback 仍是部署安全邊界，不是候選訓練捷徑。
- persistent state 必須以零初始化 residual 增強已訓練的 JEPA future trajectory；不得用隨機 readout 整條取代已學會的 future latent。
- persistent residual 必須由 body state change、owner-organ state change、target actual delta_t、anchor value 與 observation mask 直接解碼成 target value delta；不得恢復已證明不穩定的 shared-future-latent 再轉 target-latent 雙重投影。
- 每個 target 的 persistent residual 必須先在完全未參與 fitting 的 development patients 上顯著勝過同規格 owner-organ JEPA，且達到預先指定的最小實質增益，才可啟用；未通過時輸出必須精確等於 owner-organ JEPA。
- 共享 state 可使用全部合法觀測做當下重建，但 target future head 預設只能讀 anatomical owner organ。非 owner organ 只能經過跨 patient、跨 hospital 且 no-edge ablation 通過的稀疏 adapter 影響該 target；禁止無條件 all-organ forecast mixing。
- 守恆律應約束有明確定義的隱藏 store。非同步 ICU 檢驗的 anion gap、osmolality 等公式只能作一致性 audit，不得直接覆寫模型輸出。
- 「理解人體」候選至少必須通過：遮蔽恢復、多步狀態一致性、共享狀態多 target 增益、跨器官 ablation／敏感度一致性。
- 共享人體 state 的遮蔽恢復必須是同 anchor 的純當下任務：完整遮掉 target、其全部歷史與近決定性兄弟變數，只允許其他合法生理觀測恢復 target。future forecast 不得代替這項證據。
- 共享 state 重建只可使用指定 measurement age 內的實測真值；訓練遮蔽必須包含與外部測試一致的 target-aware block mask。promotion 必須同時勝過族群中心與等容量亂序 placebo，並通過 patient 與 care-unit 外部閘門。
- 共享人體 state 可包含經驗證的低秩線性 physiology modes，作為同一 body state 的可稽核分量，不得建立第二套平行人體模型。低秩 rank 必須在 discovery data 決定；每個讀取 modes 的 target 仍須遮掉所有代數／單位換算兄弟後獨立過閘門。
- 結構測試通過不等於能力 promotion。每個 target × horizon 仍須獨立通過 patient、hospital/care-unit/time、paired bootstrap、conformal 與負遷移閘門。
- 所有輸出仍是 factual observation/forecast；禁止把輸入擾動敏感度描述成治療因果效應。

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
