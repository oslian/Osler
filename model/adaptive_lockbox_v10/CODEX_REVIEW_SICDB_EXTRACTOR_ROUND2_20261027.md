# Codex 審查第二輪：SICdb extractor 與 v9 replay（addendum 前，2026-10-27，gpt-6.1-sol，唯讀、不含任何 SICdb 檔案）

## Claude 處理

**Extractor：已修正**（extractor 尚未凍結，可以直接改）

- **#1 入 ICU 病房。** 改成只看剛好在 `ICUOffset` 那一秒、屬於 4 個病房的 unitlog 紀錄；拿掉原本 −60 秒的容許範圍。
  - 同一秒有多筆時，取 id 最後的那一筆（as-of 語意）。
    - unitlog 的 id 隨時間遞增：在 105,362 組相鄰紀錄中，沒有一組的 id 倒序。
    - 依據這點，最後一筆就是那一秒結束時的狀態。
  - 查核時只看時間戳記與病房代碼，沒有讀任何數值。結果如下：
    - 2,606 個 case 在入 ICU 那一秒記了兩個不同病房，大多是 INBD+INID。
    - 這 2,606 個 case 的 LogState 全是 0，字典也沒有可以分辨的欄位，所以這項規則只做揭露。
  - 有 22 個 case 在那一秒沒有病房紀錄，直接排除並計數，不再歸成 "unknown"。
  - `OffsetAfterFirstAdmission` 缺值時直接報錯（fail-closed）。實際資料中沒有缺值。
- **#2 MAP。** 刪除 (SBP + 2 DBP)/3 推算，只用實測平均壓（703、706）。
  - 依據：v9 protocol 明文規定「MAP = measured mean pressure」。
  - 這樣一來，未驗證的 SBP/DBP 輸入、以及動脈優先順序的問題都一起消失。
- **Docstring 揭露。** 補上以下幾點：
  - 病人時間軸沒有跨病人的共同原點；replay 的到達順序是隨機的，同一病人較晚的入院可能較早到達；
  - 病房固定在入院當時，不追蹤後續轉床；
  - 同一時間的 pH/PaCO2 平均值不保證來自同一檢體；
  - 1h 標籤可能涵蓋 anchor 之前的分鐘數；availability 是保守的截止時間，不是量測時間。
- **測試。** 增為 9 個，涵蓋：
  - 入院前 30 秒的病房、同一秒兩個病房、沒有病房紀錄的 case；
  - 只用實測 MAP；
  - 缺少時間軸時直接報錯。

**v9 replay（#3–#6）：沒有改**

這幾項的程式碼是 v9 已經雜湊凍結的檔案，改動就要凍結 v10，交由使用者決定。

- **#3、#4、#5 是設計層面的性質。** 開發參考值（eICU 40 / NWICU 36 / Zigong 15 / Emory 41）是用同一套 replay 算出來的。
  - #3 隨機的到達時鐘本來就是凍結的設計。
  - #4 replay 只包含有標籤的 anchor。
  - #5 待處理的回饋依預測順序更新，不依標籤可用的時間順序。
- **#6 addendum 只檢查有沒有交集，是真實的強制執行漏洞。** 不過分數不會因此改變。

---

**Verdict: proceed after the listed fixes. Do not freeze the current extractor and outputs.**

The new replay code confirms that changing stays no longer releases pending labels. That resolves an important round-1 uncertainty. However, the review found reproducible care-unit and MAP defects, incomplete addendum enforcement, and unresolved replay-order and cohort-selection issues.

This review used only this folder. No files were modified, and no patient-level data or target values were inspected.

**Round-1 findings**

| Finding | Status | Severity | Assessment |
|---|---|---|---|
| Care unit at ICU admission | **Partially resolved** | Medium | Terminal-unit routing was replaced, but the join can choose a different unit before admission instead of the documented entry exactly at `ICUOffset`. Requires a fix, not disclosure alone. `sicdb_observation_transition_extract.py:141–151`. |
| Chronology across readmissions | **Partially resolved** | High | Extracted timestamps now incorporate `OffsetAfterFirstAdmission`. The visible replay does not establish that this chronology survives scheduling. Requires evidence or a protocol resolution. `sicdb_observation_transition_extract.py:147–157`; `site_adaptive_interval_dev.py:123–132`. |
| Order-invariant duplicates | **Resolved for the reported counterexamples** | Low | Measurements are averaged before matching; SBP and DBP are separately averaged before pairing. Reversing simultaneous potassium `[6, 4]` still produced `5` in my synthetic check. Averaging and specimen-pairing limitations require disclosure. `sicdb_observation_transition_extract.py:178–182, 223–225, 250`. |
| Validate before deriving | **Partially resolved** | Medium | Bicarbonate inputs and measured MAP availability now follow cleaning. SBP/DBP inputs bypass that cleaning, and MAP source preference precedes derived-output validation. Requires fixes. `sicdb_observation_transition_extract.py:219–243, 249–251`. |
| Hourly aggregate boundary | **Resolved** | Low | The query now requires the entire source hour inside the stay and timestamps it at completion. Remaining interval-label semantics need disclosure only. `sicdb_observation_transition_extract.py:161–170, 195–204`. |
| Tests | **Partially resolved** | Medium | Eight tests now include duplicate reversal, invalid bicarbonate inputs, aggregate boundaries and `build()`. They still miss the counterexamples below and only check that offset columns exist, rather than reconstructing label timing. `test_sicdb_observation_transition_extract.py:54–101`. |

**Findings requiring resolution**

1. **Medium — The admission-unit join does not implement its stated definition. Confirmed and reproduced.**

   The filter admits every ICU-ward entry from `ICUOffset - 60` onward, then selects the earliest (`sicdb_observation_transition_extract.py:143–144`). It does not prefer an exact match or impose an upper bound.

   My synthetic case contained:

   - CWIN at offset **3570**;
   - INBD at `ICUOffset = 3600`.

   `load_stays()` selected **CWIN**, although the supplied definition requires **INBD**.

   Under the supplied guarantee that every case has an ICU entry exactly at `ICUOffset`, this does **not** introduce future information: the selected entry is no later than that exact entry. It nevertheless assigns the wrong subgroup and adaptation key in this counterexample.

   Use the exact admission entry. Reject or explicitly resolve conflicting units at that timestamp; sorting by entry `id` is deterministic but does not establish the correct unit. Missing exact entries should not silently select a later transport or become `"unknown"`.

2. **Medium — MAP still derives from unvalidated inputs and can discard a valid fallback. Confirmed and reproduced.**

   `measurements_for()` cleans mapped direct measurements, but passes the original signal rows into `add_derived_map()` (`sicdb_observation_transition_extract.py:249–251`). SBP and DBP are averaged directly from those rows (`223–225`). The shared derivation checks only `SBP > DBP` (`osler_jepa/derived_measurements.py:28–31`).

   In an instrumented execution of the actual functions, **SBP 300 / DBP −30** reached the output cleaner as **MAP 80**. Cleaning the resulting MAP cannot identify those invalid source pressures.

   Separately, arterial preference is applied by `drop_duplicates()` **before** derived-output cleaning (`sicdb_observation_transition_extract.py:227–230`). With arterial SBP `inf`, DBP 60, and a valid noninvasive 120/60 pair, only the arterial **`inf`** reached the cleaner; the noninvasive **80** had already been discarded.

   Validate individual pressure inputs before averaging and derivation. Validate each derived candidate before selecting arterial over noninvasive. Retain the existing validated-measured-MAP priority.

3. **High — Extraction chronology is improved, but chronological deployment across readmissions remains unverified.**

   `OffsetAfterFirstAdmission` correctly establishes within-patient admission timestamps when present (`sicdb_observation_transition_extract.py:147–157`). It supplies no shared calendar origin across patients.

   The `subject_id, anchor_time` sort is therefore not a hospital-wide chronological order (`mimic_to_eicu_49_cell_external_audit.py:146`). More importantly, replay subsequently calls:

   ```python
   deployment_clock(stay_id, hours_since_onset, seed=seed)
   ```

   It does not pass `subject_id` or `anchor_time` (`site_adaptive_interval_dev.py:129–132`). The clock implementation is absent, as is the implementation of `add_history_features()` imported at `mimic_to_eicu_49_cell_external_audit.py:21–22`.

   **Confirmed:** the supplied files cannot demonstrate that later admissions remain later on the deployment clock, or independently verify stay-isolated histories. The visible documentation describes simulated Poisson stay arrivals (`site_adaptive_interval_dev.py:9–11`).

   **Not established:** actual readmission reversal or numeric-label leakage caused by that missing implementation.

   Before freezing, provide the clock and history implementations with a repeated-patient regression. If replay deliberately uses synthetic admission ordering, explicitly establish its readmission policy and restrict conclusions to that simulated ordering. Network merging only orders the clocks it receives; it cannot recover missing calendar chronology (`site_adaptive_interval_dev.py:135–142`). Network adaptation is inactive for single-site SICdb (`site_adaptive_external_audit.py:192`).

4. **High — Future-label availability selects the deployment stream before replay. Confirmed future-information selection.**

   `normalize_cohort()` removes rows lacking finite future labels before replay (`mimic_to_eicu_49_cell_external_audit.py:127–133`). The filtered cohort is then used for prediction, deployment scheduling and warm-up weighting (`site_adaptive_external_audit.py:268–292, 163–168`).

   Thus, whether an anchor participates in the simulated workload—and potentially which stays constitute warm-up—depends on a measurement that has not yet occurred at forecast time. Keeping unlabeled anchors in the extraction does not resolve this downstream selection.

   This is distinct from prematurely reading a label’s numeric value. It is retrospective selection using future measurement availability and plausibility.

   Replay eligible prediction anchors independently of future-label presence. Determine scoring eligibility separately, and let only resolved, observed outcomes update adaptive state. If a retrospective paired-observation replay is intended instead, that requires explicit reconciliation with the stated online deployment claim.

5. **Medium — Pending updates are processed in forecast order, rather than outcome-resolution order. Confirmed and reproduced.**

   The trackers scan pending lists in insertion order (`osler_jepa/site_adaptive_point.py:72–79`; `osler_jepa/site_adaptive_conformal.py:136–143, 238–245`). When several outcomes become available between consecutive forecasts, a later-resolving outcome can update state before an earlier-resolving one.

   This matters for DtACI because each update uses the tracker’s **current** thresholds and weights (`osler_jepa/site_adaptive_conformal.py:213–236`). A synthetic example with forecast times `[0, 0.1, 10]` and resolution times `[5, 1.1, 20]` produced third widths:

   - current implementation: **1.052321434**;
   - resolution-time ordering: **1.052510096**.

   This does not use unavailable outcomes, but it does differ from updating as outcomes arrive. Process due feedback in resolution-time order with a declared tie rule, or explicitly justify this batching policy in the frozen method.

6. **High — The addendum check requires only one matching horizon file. Confirmed and reproduced.**

   `lockbox_step()` checks whether the requested six-file template has **any intersection** with recorded addendum inputs (`site_adaptive_external_audit.py:240–242`). It does not require every consumed extraction file to be recorded.

   With protocol verification, hashing and ledger operations mocked in memory, a receipt containing only `fake_1h.parquet` was sufficient to reach the scoring claim for `fake_{horizon}h.parquet`.

   Every extraction file consumed by scoring must belong to the addendum and pass its recorded hash check. Missing required horizon files must fail before claiming or opening labels; currently `external_cohort()` can instead return an empty cohort for a missing file (`external_value_conditional_interval_audit.py:297–299`).

   The atomic claim itself is implemented correctly through exclusive creation (`osler_jepa/frozen_protocol.py:58–72`). The defect is incomplete input binding, not claim concurrency.

**Replay leakage assessment**

The visible point, adaptive-interval and DtACI loops now resolve feedback only when `resolve_time <= anchor_hours[row]`; none flushes outcomes on a stay change (`osler_jepa/site_adaptive_point.py:75`; `osler_jepa/site_adaptive_conformal.py:139, 241`).

Synthetic executions confirmed that changing a pending outcome did not change predictions or widths before its resolution time, and did change them afterward.

Availability is computed as `horizon + abs(label_offset)` (`mimic_to_eicu_49_cell_external_audit.py:122–125`). This conservatively delays an early nearest measurement until a closer competing measurement could no longer arrive. It is not always the measurement’s actual timestamp.

A correctly assigned ICU-admission unit, retained throughout the stay, introduces **no demonstrated future-information leak**. It represents admission-unit membership throughout subsequent transfers, rather than current ward occupancy. That distinction needs explicit disclosure.

These findings establish the local feedback guards. They do not certify the missing clock/history implementations or eliminate finding 4’s future-dependent stream selection.

**Other implementation checks and disclosures**

- `_in_stay(duration_s=3600)` correctly enforces complete-hour containment, including equality at admission and discharge (`sicdb_observation_transition_extract.py:168`). I found no new boundary-filter defect.
- Bicarbonate now uses cleaned measurements and validates its derived output (`sicdb_observation_transition_extract.py:237–243, 250–251`). The original PaCO₂ **201** counterexample no longer generated bicarbonate in an extracted-function check using the supplied PaCO₂ bound.
- `collapse()` removes the demonstrated row-order dependence. Its policy weights records equally, including repeated records; it does not distinguish duplicate records from distinct simultaneous measurements.
- **Low — Empty-anchor handling still fails in the shared assembler.** A synthetic stay lasting ten hours with a six-hour horizon passes stay selection but yields no anchors. With retained measurements, `assemble_state()` raised a `MergeError` because empty-anchor `stay_id` had object dtype while measurements used integers (`mimiciv_observation_transition_extract.py:594–605, 640–648`). This is inherited rather than introduced here, but should have an empty-build regression.

The extractor docstring is **not yet complete or fully accurate**. Revise it to cover:

- the actual admission-unit selection after fixing finding 1;
- synthetic cross-patient chronology and the verified readmission policy;
- fixed admission-unit grouping after transfers;
- same-timestamp pH/PaCO₂ averaging without proven specimen pairing, and hourly SBP/DBP aggregation without proven simultaneous raw samples;
- one-hour labels potentially including pre-anchor minutes when their nearest endpoint precedes the requested future time;
- the actual validation of MAP source pressures;
- the conservative nearest-label resolution deadline, rather than simply “measurement time.”

The sample-time laboratory limitation, arterial-item selection, hourly urine interpretation and intermediate-care population are already disclosed (`sicdb_observation_transition_extract.py:12–55`). Missing admission offsets are silently replaced with zero and also need a validated policy (`147`).

Dictionary identities, units and `LaboratoryType` semantics remain independently unverifiable here. No new conversion error was demonstrated. The WBC/platelet comments now explicitly explain the intended count units (`sicdb_observation_transition_extract.py:98–101`).

Also reconcile the frozen protocol’s **“MAP = measured mean pressure”** requirement with the extractor’s derived fallback (`adaptive_model_lockbox_protocol_v9_20261027.json:47`; `sicdb_observation_transition_extract.py:288`). The current wording does not clearly authorize the surrogate.

**Verification and release boundary**

All **14 Python files parsed successfully**. All **12 available files named in the frozen-code manifest matched their hashes**. The six support reports were inspected; they cannot establish output-value correctness or provenance.

The full unittest invocation failed at import because `duckdb` is unavailable. Additional required modules—including the deployment clock, history builder and addendum implementation—are absent. My synthetic checks exercised extracted functions with explicit instrumentation or mocks; they do not constitute complete CSV/SQL or scoring integration tests.

Resolve the findings, add targeted regressions, regenerate the six extraction outputs, and verify complete addendum coverage before opening labels. Changes to hashed replay or enforcement files require a **new protocol version**, as required by `adaptive_model_lockbox_protocol_v9_20261027.json:66`; the extraction addendum cannot silently override that rule.