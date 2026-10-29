# Codex 審查：HiRID extractor（addendum 前，2026-10-27，gpt-6.1-sol，唯讀、不含任何 HiRID 檔案）

## Claude 處理

開發時用的是使用者提供的 rar 測試副本，正式評分會改用官方檔案。

1. **Glucose 不對應。** 字典把三個項目都標成 Serum or Plasma，沒有區分中央實驗室、床邊血糖或血氣分析。依 protocol v10，`glucose` 與 `glucose_poc` 格不可評估，欄位留空，報告記錄原因。
   - 支援數量和時間戳記顯示 24000585 很可能是床邊血糖：52.9 萬筆，與 pH 同時間的只有 4%。
   - 依 Codex 意見，這只是佐證，不能拿來重新分類。
2. **住院結束時間改為「最後一筆有效心率」。** 心率和其他量測一樣，先取最新版本，再做 plausibility 驗證。
   - 刪除「監測結束即出院」的說法，改稱 monitored-window proxy。
   - 計數欄位改名為 `patients_without_valid_heart_rate`。
3. **空輸入。**
   - `selectable` 收到空的量測或空的 anchor 時，回傳有型別的空表。
   - 沒有任何量測時，回傳固定欄位的空表。
   - 有選入住院但沒有 anchor 時，build 會回傳空的轉移表。
4. **版本與作廢規則。** 先取最新輸入的版本（entertime，再 C > F > P，再取較小值），最新版本若被旗標或沒有值，整筆撤回，不退回較舊版本。
   - 支援數量：實驗室項目有 1,136,941 / 4,251,877 個 key 有多個版本。
   - 其中「最新版本被作廢、較舊版本有效」只有 37 個。
5. **Docstring 揭露補齊。**
   - 病人與住院相同：bootstrap 和 patient-equal 都以入院為單位。
   - 病房只有一組，所以病房覆蓋率等於整體覆蓋率。
   - 採最終值、以觀測時間重建，不代表前瞻情境下當時就拿得到。
   - 尿量時間點沒有文件說明。
   - pH/PaCO2 不保證來自同一檢體。
   - PEEP 的設定值與實測值、各量體溫部位，都不保證可以互換。
   - 凍結的 replay 性質照原樣適用。
6. **測試增為 10 個。** 新增：
   - 不合理的心率不會延長住院；
   - 最新版本被作廢就撤回；
   - 分批與不分批的結果完全相同（跨批邊界）；
   - 選入住院但沒有 anchor 的情況。

**沒有改的：** 用 `datetime` 而不是 `entertime`。凍結的 MIMIC 合約就是這樣，改成 `entertime` 會偏離合約，所以只做揭露。

**待驗證：** 官方 `observation_tables_parquet` 解開後的路徑，下載後確認。

---

**Verdict: proceed after the listed fixes. Do not freeze the current extractor and outputs.**

The principal blocker is glucose mapping. I also found a stay-end validation defect and reproducible empty-input failures. The row-pruning optimization is sound under its stated invariants.

This review used only this folder. No files were modified, and no patient-level data were inspected. Dictionary identities and source semantics below rely on the supplied brief; the distribution itself is absent.

**1. High — Glucose must be declared not evaluable under the supplied evidence. Confirmed protocol mismatch.**

`hirid_observation_transition_extract.py:32–35,106` treats all three serum/plasma glucose items as chemistry-laboratory glucose. The inference “serum/plasma, i.e. not capillary point-of-care” is unsupported: specimen or reported concentration terminology does not establish the measurement workflow.

Protocol v10 explicitly distinguishes chemistry, point-of-care and blood-gas glucose, and makes `glucose` and `glucose_poc` unevaluable when the database does not distinguish them (`adaptive_model_lockbox_protocol_v10_20261027.json:47`).

The supplied counts and pH coincidences establish neither that item 24000585 **is** point-of-care nor that the other two **are** central-laboratory measurements. They cannot replace the required dictionary distinction.

Before freezing:

- Exclude these ambiguous items from the canonical glucose mappings unless dictionary documentation establishes their measurement types.
- Ensure both `glucose` and `glucose_poc` cells are actually unevaluable. With the current scoring path, leaving their contract columns entirely missing-valued accomplishes this without changing frozen scoring code.
- Do not relabel 24000585 as `glucose_poc` or select the smaller items as chemistry based on frequency or timestamp patterns.

A report flag alone does not enforce exclusion: `normalize_cohort()` selects finite current/future values, and `evaluability()` checks support rather than the extractor’s availability flags (`mimic_to_eicu_49_cell_external_audit.py:109–133`; `external_value_conditional_interval_audit.py:307–315`). The current six-hour report contains **88,612 glucose pairs**, so the mapping has a material scoring consequence.

**2. Medium — Stay end is not the last plausibility-validated heart rate. Confirmed defect.**

`load_stays()` takes `max(datetime)` after checking only non-null value, status bits and admission time (`hirid_observation_transition_extract.py:141–149`). Plausibility cleaning occurs later, inside `measurements_for()` (`184–185`).

Consequently, a heart-rate value subsequently rejected by cleaning can still:

- extend `outtime`;
- admit a stay;
- create additional anchors;
- permit other measurements into the stay window.

Using the actual `load_stays()` query against an in-memory SQLite fixture, heart rates **80 at hour 10** and **−1,000,000 at hour 30**, both status zero, produced an end at **hour 30**. This verifies the predicate defect; it is not a DuckDB integration test.

Resolve heart-rate versions consistently and apply the contract’s heart-rate validity rules **before** computing the endpoint. Count admissions without a retained valid heart rate. The existing test checks an invalidation flag, not implausible numeric heart rates (`test_hirid_observation_transition_extract.py:25–26,70–74`).

The endpoint itself requires a separate interpretation:

- **Confirmed:** it uses future monitoring availability to define a retrospective observation window.
- **No demonstrated additional numeric-label leak:** it does not inject future heart-rate values into anchor features merely by taking their timestamp maximum.
- **Difference from MIMIC:** MIMIC uses administrative ICU `outtime`; HiRID additionally conditions the window on heart-rate recording. Monitoring cessation is not demonstrated to equal discharge.

Retain this as an explicitly defined monitored-window proxy. Remove the unsupported statement “monitoring ends at discharge” (`hirid_observation_transition_extract.py:18–19`).

**3. Medium — Empty-input handling fails before producing contract-shaped output. Confirmed and reproduced.**

Two in-memory executions of the supplied functions failed:

| Case | Result |
|---|---|
| Empty measurements passed to `selectable()` | `ValueError: No objects to concatenate` |
| One ten-hour stay, six-hour horizon, hence no eligible anchors | `MergeError` between object and integer `stay_id` keys |

The first originates at `hirid_observation_transition_extract.py:195–202`. The second follows from untyped empty anchors in `mimiciv_observation_transition_extract.py:594–607` entering `selectable()` at `hirid_observation_transition_extract.py:200`. Zero selected stays also leaves `pieces` empty at `226`.

Handle these cases in the **HiRID code**, returning appropriately typed empty results before invoking the frozen shared functions. Do not change the frozen SICdb helper or MIMIC contract. Add empty-build regressions.

**4. Time alignment and corrections — acceptable for the frozen observation-time comparison, with a material availability limitation.**

The implementation uses `datetime` for matching and `entertime` only to choose versions (`hirid_observation_transition_extract.py:168–176`). This matches the visible MIMIC extraction’s use of `charttime` without a result-entry availability check (`mimiciv_observation_transition_extract.py:278–298`).

However, the resulting features are **not an entry-time reconstruction**. A measurement observed at hour 12 and corrected at hour 15 can contribute its corrected value to the hour-12 state. Similarly, replay’s `anchor + h + |offset|` deadline does not account for a later database entry or correction. The fixture already includes a final result observed at hour 12 but entered at 12.5 (`test_hirid_observation_transition_extract.py:32–35`).

**Medium — disclosure required:** final-value, observation-time reconstruction is defensible for this frozen retrospective comparison, but does not establish that those values were available in prospective deployment. “MIMIC does the same” explains comparability; it does not remove the limitation. Do not silently switch to entry-time matching or alter frozen replay timing.

The `QUALIFY` ordering correctly implements:

1. latest `entertime`;
2. correction before final before preliminary for equal entry time;
3. smallest numeric value for otherwise tied records.

The third rule is deterministic but undocumented (`hirid_observation_transition_extract.py:171–172`).

**Medium, conditional risk — invalidation/version semantics remain unresolved.** Status filtering occurs **before** version selection (`170–172`). Thus, an older unflagged entry survives if a newer same-item/time entry is invalidated or null. The SQL behavior is clear; the supplied facts do not establish whether such a newer entry withdraws the earlier measurement or merely represents a rejected version. Document and test the intended policy using source semantics before changing this ordering.

The dropped mask is correctly **226 = 2 | 32 | 64 | 128**. Excluding invalidated, unmeasured and censored values is appropriate for exact numeric targets. Retaining bit 1 and then applying protocol plausibility cleaning is defensible. No supplied definition requires dropping connection, event, compression or mandatory flags.

**5. `selectable()` — exact under the post-collapse contract; batching is conditionally exact.**

For a query time, the backward candidate and forward candidate contain every possible nearest match. Applying tolerance afterward preserves both matches and missingness. Keeping candidates without tolerance therefore retains extra rows but does not change selection.

Equal-distance ties are preserved because both candidates remain for the same `assemble_state()` implementation to resolve. Exact timestamps remain available. Duplicate source timestamps are resolved upstream through per-item version selection and per-variable averaging (`hirid_observation_transition_extract.py:171–185`; `sicdb_observation_transition_extract.py:194–198`).

I executed the extracted, unchanged functions in memory:

- **400** full-versus-pruned and full-versus-batched comparisons passed across all six horizons.
- A deliberate equidistant tie selected the earlier measurement in both paths.
- Collapsed duplicate values and exact-time matches also agreed.

This supports exactness **after collapse**, not arbitrary uncollapsed duplicate input.

Part grouping is exact if each stay’s complete observation history resides in its indexed part. Each stay then belongs to one batch, and all cleaning, averaging, derivation and selection are stay-local (`hirid_observation_transition_extract.py:214–226`). Missing index entries fail closed.

The supplied index description supports that assumption, but no source files were available to verify it. **Low — verification gap:** tests should compare actual multi-group `build()` output against an unbatched reference, including a group boundary. The existing fixture does not exercise that (`test_hirid_observation_transition_extract.py:47–50,114–126`).

**6. Other variable mappings — no demonstrated conversion defect.**

Relative to the supplied units, the factors in `hirid_observation_transition_extract.py:85–116` are dimensionally consistent, including:

- glucose ×18.016, creatinine ÷88.42 and urea-to-BUN ×2.8013;
- bilirubin ÷17.104;
- calcium ×4.008, magnesium ×2.431 and phosphate ×3.097;
- hemoglobin/albumin ÷10, fibrinogen ×100;
- troponin T ×0.001 and cortisol ÷27.59.

The source dictionary is absent, so this checks the stated conversions rather than independently authenticating every item’s unit.

Pooling potassium, sodium and hemoglobin is consistent with the multiple-item MIMIC mappings (`mimiciv_observation_transition_extract.py:58–69`). Lactate pooling is consistent with the supplied mapping description. Temperature and PEEP pooling are structurally consistent with the MIMIC vital mappings (`108–118`). This does not establish interchangeability of specimen types, temperature sites, or set versus measured PEEP.

Measured-only MAP is correctly implemented. Validation precedes simultaneous averaging and bicarbonate derivation. Derived bicarbonate is validated again, and retained measured bicarbonate takes precedence (`hirid_observation_transition_extract.py:184–185`; `sicdb_observation_transition_extract.py:224–234`).

Excluding CK-MB activity, NT-proBNP, unavailable hematocrit and minute volume is appropriate under the supplied definitions. Hourly urine is already expressed as mL/h; no additional division is indicated. Its collection-window timestamp convention remains unverified.

**7. Stay identity, care unit, replay and reports — usable with explicit limits.**

**Medium — admission-level identity must not be described as verified person-level independence.** Setting subject equal to stay follows the absence of readmission linkage (`hirid_observation_transition_extract.py:151–155`). Bootstrap clusters and “patient-equal” coverage therefore operate on **admissions**. Repeated admissions from the same person cannot be linked, so their dependence cannot be accounted for.

The single `"ICU"` group accurately reflects the supplied department-level information. The care-unit gate remains computable under the frozen HiRID axes (`external_value_conditional_interval_audit.py:62–67,180–184`), but with one group its care-unit coverage equals overall row-weighted coverage. It provides no evidence of robustness across distinct care units.

The frozen protocol already discloses random stay arrivals, label-selected replay and forecast-order feedback (`adaptive_model_lockbox_protocol_v10_20261027.json:141–145`). These require disclosure, not another extractor-side replay redesign. The scorer and deployment-clock implementation are absent here, so their execution was not independently verified.

The six aggregate reports passed internal consistency checks. In particular:

- paired counts do not exceed output rows;
- subject and stay counts agree;
- selected-stay counts exceed represented stays as expected when selected short stays produce no anchors;
- timing settings and status masks agree with the code.

For example, the six-hour report has 33,215 selected stays but 31,596 represented stays (`results/hirid_observation_transitions_6h_v1.json:147–156`). That difference is not itself lost-anchor evidence.

**Low — report/disclosure improvements:** record glucose’s exclusion reason, admission-level identity, final-value timing assumptions, and the complete numeric tie rule. Clarify that measurement-row counts precede pruning and include selected stays without anchors.

**8. Documentation, tests and freeze boundary**

The docstring is **not yet complete or fully accurate**. Correct the glucose assertion and discharge-equivalence claim, and add:

- retrospective entry/correction availability limitations;
- the admission-level bootstrap limitation;
- the one-group care-unit gate’s limited interpretation;
- explicit final tie and invalidation policies;
- lack of proven specimen pairing for derived bicarbonate;
- hourly urine’s window-timing assumption;
- the frozen replay properties.

Also reconcile the brief’s `observation_tables_parquet` layout name with the hard-coded `raw_stage/observation_tables/parquet` path (`REVIEW_BRIEF.md:9`; `hirid_observation_transition_extract.py:129–132`). This is an unresolved packaging check, not a demonstrated archive incompatibility. Per-part row-count equality alone does not prove a lossless CSV conversion.

All **seven Python files parsed**, and all **four available frozen-code manifest entries matched their hashes**. The full unittest invocation failed during import because `duckdb` is unavailable; other required project imports are also absent from this folder. The synthetic checks therefore do not certify DuckDB execution, parquet integration, official-file compatibility or scoring.

Before the addendum, fix glucose exclusion, validated stay-end selection and empty-input handling; add targeted regressions and correct the disclosures. Regenerate all six outputs from the intended verified source, then bind the HiRID extractor, mapping and every output in the one-shot addendum. Any necessary change to a protocol-hashed file requires a new protocol version (`adaptive_model_lockbox_protocol_v10_20261027.json:48,66`).