# Codex 審查第二輪：HiRID extractor（addendum 前，2026-10-28，gpt-6.1-sol，唯讀；送審資料夾不含任何 HiRID 檔案）

## Claude 處理

Codex 結論：**修正後再進 addendum**。第一輪的問題已確認解決；glucose 在評分器中會正確變成「不可評估」。

1. **（中）同等排序但旗標不同的版本，結果取決於輸入順序。** 已修。
   - 排序改為 `entertime DESC, C > F > P, withdrawn DESC, value`，其中 `withdrawn = value IS NULL OR (status & 226) <> 0`。
   - 同等排序的版本中，只要有一個被作廢或沒有值，這筆量測就撤回；否則取較小值。
   - 心率（住院結束）和其他量測走同一個 `current_values()`，所以同樣適用。
   - Docstring 和報告的 `simultaneous_values` 已同步更新。
   - 報告新增 `availability` 欄位：最終值、依觀測時間重建，不代表前瞻情境下當時拿得到（Codex 提的低嚴重度文件缺口）。
2. **（中）測試不足。** 新增 5 個，共 15 個，全部通過：
   - 版本規則：C > F > P、較新的空值撤回、平手取較小值、同等排序一筆作廢或空值則撤回。正反兩種輸入順序都測。
   - 住院結束心率套用相同版本規則：較新的作廢版本、同等排序衝突。
   - 空輸入：`selectable` 收到空量測或空 anchor、`measurements_for` 收到空 stays、`build` 沒有任何住院入選，都保留欄位。
   - Glucose 整合：三個 glucose item 都不對應，對應欄位全為 NaN，配對數為 0。評分器在「欄位全空」和「欄位不存在」兩種情況下，都回傳 `insufficient_external_support`。
   - `selectable` 邊界：完全同時、等距、容許範圍兩端、look-back 邊界、剛好超出容許範圍，含 offset 欄位都相同。
   - **反向驗證：** 把排序改回舊版後，新增的兩個版本測試會失敗，確認測試真的抓得到這個問題。
3. Codex 的環境沒有 duckdb，無法執行測試。本機 `.venv` 已跑完整套測試。
4. **已過時：** 目前 6 份 HiRID 輸出是用修正前的 extractor 從測試副本產生的。正式萃取會用官方檔案和這版 extractor 重跑。
5. 不需要修改受 protocol 凍結的檔案：SICdb helper 和評分器都不用動。

---

**Verdict: proceed after the listed fixes.** The original glucose, stay-end and empty-input defects are addressed in the implementation. One version-selection ambiguity remains, and the regression suite needs additional coverage before the one-shot freeze.

This review used only this folder. No files were modified or patient-level data inspected. Source identities and dictionary semantics remain supplied evidence; official-file compatibility is unverified.

**1. Medium — Equal-ranked versions with conflicting status can be retained or withdrawn nondeterministically.**

In [hirid_observation_transition_extract.py:158](/private/tmp/claude-501/-Users-chunyouchang-Desktop-llm-project/cd050cbe-38bb-4f06-b4b5-ef224e0457b1/scratchpad/codex_review_hirid_r2/hirid_observation_transition_extract.py:158), `row_number()` orders by:

```sql
entertime DESC, type_priority DESC, value
```

The outer predicate then rejects the selected row if its status contains a dropped bit. Two rows can share **patient, item, datetime, entertime, type and value**, but disagree on status. Nothing resolves that tie.

For example, two otherwise identical potassium records with value `4.0`, one status `0` and one status `2`, have different consequences depending on which receives row number 1.

**Confirmed:** the SQL leaves this case unspecified. An equivalent in-memory SQLite window query retained the measurement with insertion order `(0, 2)` and withdrew it with `(2, 0)`. This is a counterexample to deterministic selection, **not a DuckDB reproduction**. Whether such conflicts occur in HiRID is unknown.

Before freezing:

- Define a deterministic policy for tied candidates whose withdrawal outcomes differ, such as withdrawing an ambiguous measurement.
- Document the policy and add reversed-input-order tests.
- Apply the same policy through `current_values()` for both measurements and heart-rate endpoints.

This is upstream of pruning: `selectable()` can remain exact while the observations supplied to it change.

**2. Round-1 disposition**

“Resolved” below describes the inspected implementation; the complete DuckDB suite did not run here.

| Round-1 finding | Severity | Round-2 disposition |
|---|---|---|
| **1. Glucose mapping** | High | **Resolved in code and reports.** No glucose item remains in `ITEMS` (`hirid_observation_transition_extract.py:100–131`). The docstring explicitly excludes the ambiguous items (`41–45`), and report fields explain exclusion (`285–287`). Downstream handling is correct, as detailed below. |
| **2. Unvalidated heart-rate endpoint** | Medium | **Resolved for ordered versions.** `load_stays()` calls `current_values()`, applies `contract.clean_measurements()`, then takes the maximum timestamp (`169–176`). The fixture now includes implausible late heart rate (`test_hirid_observation_transition_extract.py:25–27`), and the endpoint/count assertions cover it (`73–78`). The equal-rank issue above remains shared with this path. |
| **3. Empty-input failures** | Medium | **Resolved in the inspected paths.** `selectable()` returns early (`214–215`); empty pieces produce a measurement table with required columns (`250–251`). In-memory checks of the unchanged functions passed empty measurements, empty anchors, zero selected stays, and selected stays without anchors. Committed regression coverage remains incomplete. |
| **4. Observation-time alignment and withdrawal semantics** | Medium | **Mostly resolved.** Final-value retrospective timing and prospective-availability limitations are explicit (`26–36`). Version selection now precedes withdrawal (`150–160`), and the newer-flagged case is tested (`test…py:40–41,87–92`). Newer-null withdrawal and equal-time priorities lack direct tests; conflicting equal-ranked statuses remain unresolved. The policy is documented, but its source-system interpretation is not independently authenticated here. |
| **5. Pruning and part-group equivalence** | Low verification gap | **Test added; algorithm remains sound under its invariants.** The new `build()` comparison crosses two parts with group sizes 1 and 10 (`test…py:131–144`). Complete stay history residing in the indexed part remains a source-layout assumption, unverified without official files. |
| **6. Other mappings and conversions** | Low verification limitation | **No new demonstrated defect.** Conversions and exclusions remain consistent with the supplied dictionary descriptions (`hirid…py:46–73,100–131`). Cleaning, collapse and bicarbonate derivation retain the intended order (`203–207`). This does not independently verify dictionary units or specimen interchangeability. |
| **7. Identity, care unit, replay and reports** | Medium disclosures; low report improvements | **Substantially resolved.** Admission-level clustering, the single-unit limitation and replay properties are disclosed (`16–36,75–79`). Reports include identity, endpoint, exclusion and pre-pruning-count explanations (`275–287`). **Low remaining documentation gap:** aggregate JSON reports do not themselves record the final-value availability limitation; it currently resides in the extractor docstring. |
| **8. Documentation, tests and freeze boundary** | Medium verification gap | **Partly resolved.** The requested docstring disclosures are present. Official archive layout, complete integration execution and the final official-source artifacts remain unverified. The handling notes explicitly leave layout confirmation pending (`CODEX_REVIEW_HIRID_EXTRACTOR_20261027.md:36`). |

The evidence changing the round-1 judgment is concrete: glucose mappings were removed, endpoint cleaning moved before aggregation, empty selection gained an early return, and the missing disclosures and group-boundary fixture were added.

**3. Regressions and contract compliance**

I found **no demonstrated regression in `selectable()` or the stay-local batching logic**.

The optimization retains each query’s predecessor and successor after collapse. These include the backward state match and both candidates for the nearest future label. The unchanged contract then applies tolerance and resolves equal-distance ties (`mimiciv_observation_transition_extract.py:611–659`).

Using extracted, unchanged functions in memory:

- **720 comparisons** of full, pruned and stay-batched inputs passed across all six horizons.
- **18 directed cases** passed for exact timestamps, equidistant matches on the tolerance boundary, and matches just outside tolerance. Label offsets also agreed.
- Empty-input orchestration returned contract columns without the previous concatenation or merge failures.

These checks support the pandas selection logic. They do not certify parquet loading, DuckDB execution or actual official-part completeness.

The heart-rate change correctly removes implausible endpoint observations before determining `outtime`. The monitored-window selection limitation remains explicitly disclosed; this change does not establish an administrative discharge time.

**4. Glucose becomes not evaluable, rather than failed or silently passed**

The extractor, reports and scorer agree.

The implementation path is:

1. No ambiguous glucose item enters the mapped measurements.
2. `assemble_state()` creates all-NaN contract columns for absent variables (`mimiciv_observation_transition_extract.py:627–638`).
3. `normalize_cohort()` returns an empty cohort when required target columns are missing, or filters away rows without finite current/future pairs (`mimic_to_eicu_49_cell_external_audit.py:109–133`).
4. `evaluability()` returns `insufficient_external_support` for that empty cohort (`external_value_conditional_interval_audit.py:307–315`).
5. The scorer appends an explicit unevaluable cell and continues before fitting (`site_adaptive_external_audit.py:269–274`). Only evaluable cells contribute to robust counts (`340–354`).

The resulting cell entry is:

```json
{
  "cell": "<catalog cell>",
  "evaluable": false,
  "reason": "insufficient_external_support"
}
```

It receives neither a failed gate nor a successful gate.

I checked **36 synthetic normalization cases**: both glucose targets, six horizons, and three conditions—missing columns, all-null columns, and disjoint finite current/future observations producing zero pairs. All became unevaluable. Executing the actual scorer `run()` with synthetic catalog and I/O/configuration/lockbox stubs also skipped fitting and produced zero robust counts for these cells.

Missing glucose as a **covariate for other targets** is handled by `_safe_float()` and the observation-mask construction in `covariate_matrix()` (`target_specific_domain_conformal_audit.py:89–93`; `site_adaptive_interval_dev.py:83–96`). Synthetic missing/all-null checks produced finite matrices.

All six aggregate reports match the brief’s stay/row counts, contain zero glucose/glucose_poc/glucose_bg pairs, and contain no glucose measurement counts. Report flags explain exclusion; the scorer enforces unevaluability through the actual transition columns.

**5. Medium — The 10 tests are insufficient for the complete fixed behavior**

The existing tests cover useful nominal cases, including the implausible endpoint, newer flagged withdrawal, multi-part batching and selected stays without anchors. Before freezing, add:

- **Version-policy tests:** strictly newer null withdraws an older valid value; equal-entry-time `C > F > P`; smaller-value tie resolution; conflicting equal-ranked statuses in both input orders. Include an endpoint heart-rate version being withdrawn.
- **Empty-path tests:** direct empty measurements and empty anchors in `selectable()`; empty `measurements_for()`; zero selected stays in `build()`. Assert required output columns and relevant dtypes, beyond row count.
- **Glucose integration regression:** all three excluded item IDs remain unmapped; resulting glucose columns are missing-valued, pair counts are zero, and report flags agree. Exercise the scorer’s missing-column and zero-pair unevaluable branches.
- **Pruning boundary regression:** preserve the directed exact-time, equal-distance, tolerance-boundary and outside-tolerance cases, including offsets.

The full unittest command **failed during import because `duckdb` is unavailable**. Therefore, none of the 10 committed tests can be reported as passing in this environment. Additional project imports required by the extractor/scorer are also absent from this review bundle.

**6. Remaining freeze requirements**

Resolve finding 1 and add the targeted regressions, then run the suite in the dependency-complete environment before consuming the one-shot addendum. The final six extraction artifacts must come from the intended checksum-verified official source, with the actual unpacked layout confirmed; the current aggregate reports cannot authenticate that provenance (`REVIEW_BRIEF.md:14`).

All **18 Python files parsed**, and all **15 locally present protocol-hashed files matched** their manifest hashes. No change to the frozen SICdb helpers or scorer is needed for glucose exclusion or the identified version-selection fix. Any change to a protocol-hashed file would require the protocol’s refreeze rule (`adaptive_model_lockbox_protocol_v10_20261027.json:66`).

**Proceed after these fixes and verification steps; do not freeze the current extractor as is.**