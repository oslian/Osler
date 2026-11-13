# Codex 審查：SICdb extractor（addendum 前，2026-10-27，gpt-6.1-sol，唯讀、不含任何 SICdb 檔案）

Claude 處理：#1 病房改為入 ICU 時的病房（unitlog 在 ICUOffset 的那筆；每個 case 都有，約 5% 與最後病房不同）；#2 replay 以模擬時鐘排序、歷史特徵依 stay 計算，另以 OffsetAfterFirstAdmission 讓同一病人的住院依真實先後排列；#3 同一時間的多個值先驗證再平均；#4 先驗證再推算，推算值再驗證；#5 只用整小時都在住院期間內的聚合值；#6 測試增為 8 個，包含 Codex 的反例與 build() 整合測試。揭露事項寫在 extractor 的 docstring。

---

**Verdict: proceed to the addendum only after the fixes and evidence checks below. Do not freeze the extractor as is.**

The main problems are retrospective care-unit assignment, unresolved chronology across admissions, order-dependent duplicate handling, and deriving measurements before validating their inputs. These are separate from the unit-conversion arithmetic, where I found no demonstrated error.

This review used only this folder. No files were modified and no patient-level data or target values were inspected. Documentation-dependent conclusions below are conditional on the facts supplied in `REVIEW_BRIEF.md`; the source dictionary itself is absent.

**Findings requiring resolution before the addendum**

1. **High — The final care unit is presented as the first care unit and cannot establish the ward known at an anchor.**  
   `sicdb_observation_transition_extract.py:112–119` copies `HospitalUnit` into both `first_careunit` and `last_careunit`. The supplied documentation defines it as the **last** unit using the case (`REVIEW_BRIEF.md:16`). The consumer then uses `first_careunit` as `careunit` (`mimic_to_eicu_49_cell_external_audit.py:135–138`), and SICdb’s interval gate includes that axis (`external_value_conditional_interval_audit.py:62–67`).

   **Confirmed:** these are terminal-unit groups, not first-unit or anchor-time-unit groups. They can support a retrospective terminal-unit subgroup analysis, with that name and limitation. They do not establish performance in the ward occupied at prediction time.

   **Unverified but consequential:** v9 specifies hospital/ward adaptive intervals (`ADAPTIVE_MODEL_LOCKBOX_PROTOCOL_V9_20261027.md:19`). Its actual replay implementation is absent here. If it routes predictions or adaptive updates using this terminal unit, it uses future information.

   **Required:** resolve and document the online care-unit assignment before freezing. Keep terminal unit as explicitly retrospective metadata. “Unitlog is difficult to interpret” does not make the last unit an anchor-time substitute. Any departure from the frozen ward adaptation needs an explicit protocol resolution.

2. **High — Case-relative timestamps do not establish chronological order across readmissions or patients.**  
   All cases use the same `BASE_TIME`, and admission/discharge times are constructed solely from within-case offsets (`sicdb_observation_transition_extract.py:57`, `109–123`, `136`). Two admissions of the same patient therefore occupy overlapping artificial timelines. The consumer preserves the shared patient identity and sorts by patient and anchor time (`mimic_to_eicu_49_cell_external_audit.py:116–121`, `146`).

   **Confirmed:** within-case timing is represented correctly; real inter-admission and hospital chronology is not represented. The aggregate report shows more stays than subjects, so repeated subjects exist (`results/sicdb_observation_transitions_6h_v1.json:155–156`).

   **Unverified:** whether history construction crosses stays, and how v9 schedules patient, hospital and ward updates. Those implementations are missing.

   **Required:** establish that histories and adaptive error state cannot cross admissions in an incorrect order. If real chronology is unavailable, freeze an explicit synthetic replay ordering and reset policy, and restrict the interpretation accordingly. These timestamps cannot substantiate a real calendar-time validation claim.

3. **Medium — Duplicate handling changes outputs when input row order changes.**  
   Direct measurements retain simultaneous rows from multiple mapped items (`sicdb_observation_transition_extract.py:145–148`, `161–163`). The shared assembler selects with `merge_asof` without first resolving duplicate `(stay_id, variable, charttime)` keys (`mimiciv_observation_transition_extract.py:628–648`).

   MAP additionally performs a potentially many-to-many SBP/DBP merge and then keeps the first derived result (`sicdb_observation_transition_extract.py:167–175`).

   **Reproduced using synthetic inputs and the extracted functions:**

   - Reversing two simultaneous potassium rows changed the selected label from **6 to 4**.
   - With simultaneous SBPs `[120, 150]` and DBPs `[60, 90]`, reversing the rows changed derived MAP from **80 to 110**.

   Actual duplicate prevalence is unknown; no patient data were available.

   **Required:** freeze a deterministic, dictionary-justified policy for duplicate records and simultaneous sources before as-of matching or derivation. Distinguish duplicate records from genuinely different measurements. Preserve enough source information to apply that policy. A stable sort alone does not establish a meaningful source preference.

4. **Medium — Derivation occurs before plausibility cleaning, allowing invalid inputs to produce valid-looking outputs or suppress valid replacements.**  
   Bicarbonate and MAP are derived before the single `clean_measurements` call (`sicdb_observation_transition_extract.py:164–188`, `199–200`).

   **Reproduced:**

   - pH **6.7** and PaCO₂ **201** generated bicarbonate **24.566**. PaCO₂ exceeds the contract’s upper bound of 200 (`mimiciv_observation_transition_extract.py:123`), but derivation has already occurred. The complete bicarbonate bounds are not supplied, so final retention was not verified.
   - An infinite measured MAP prevented derivation from an otherwise valid 120/60 pair. The function returned the infinite measured value.

   Similarly, a present but subsequently rejected bicarbonate prevents calculation (`sicdb_observation_transition_extract.py:185–186`).

   **Required:** validate source inputs before derivation and determine measured-value availability after validation. Validate derived outputs afterward. Freeze how same-time pH/PaCO₂ pairs are selected: independently averaging all pH and PaCO₂ values (`183–184`) can combine different specimens rather than an actual paired sample.

5. **Medium — The stay filter checks an hourly aggregate’s end, allowing part of its source window to precede ICU admission.**  
   Signals are shifted by one hour and then filtered using that shifted timestamp (`sicdb_observation_transition_extract.py:134`, `155–160`).

   For example, an aggregate covering offsets 0–3600 is retained when `ICUOffset=1800`, because its end lies inside the stay. Half its source window precedes ICU admission.

   **Confirmed:** this is a boundary-window issue, not post-anchor leakage. Point-timed pre-ICU labs are correctly excluded.

   **Required:** either require the complete aggregate window to lie inside the stay, or explicitly authorize and disclose boundary-straddling aggregates before freezing. Retain interval-start provenance if needed. Dropping aggregates that finish after discharge is conservative.

6. **Medium — Existing tests do not cover the failure conditions needed for an irreversible freeze.**  
   The five tests cover stay selection, one conversion/calculation example, end-of-hour timing, ordinary MAP fallback, and variable-name membership (`test_sicdb_observation_transition_extract.py:39–64`). They do not exercise `build`, anchor retention, future-label offsets, duplicate-order invariance, invalid derivation inputs, boundary-straddling hours, or repeated admissions.

   **Required regression checks:** the counterexamples above, exact stay/anchor boundaries, unlabeled-anchor retention, label-offset reconstruction, empty inputs, and admission isolation. Include synthetic CSV integration tests for the actual DuckDB queries.

**Time alignment and limitations that need disclosure**

- **Low — No post-anchor measurement-time leakage was found in the `_t` assembler.** SICdb requests backward matching with a six-hour lookback (`sicdb_observation_transition_extract.py:202`); the contract matches by stay and timestamp (`mimiciv_observation_transition_extract.py:640–653`). End-of-hour dating is appropriate for completed hourly aggregates under the supplied documentation. My synthetic check excluded a later observation from `_t`.

- **Medium — Lab sample time is not result-availability time.** Labs become usable at `Offset` without a reporting delay (`sicdb_observation_transition_extract.py:142–148`). An anchor can therefore use a sample collected before the anchor whose result became available afterward. MIMIC likewise uses `charttime` here (`mimiciv_observation_transition_extract.py:279–285`), so this is a shared contract limitation. Disclose an idealized measurement-time replay; actual clinical availability cannot be verified. Do not invent a delay without evidence.

- **Low — Future-label offsets are correctly retained and consumed.** Labels use nearest matching with tolerance `min(2, horizon/2)` (`sicdb_observation_transition_extract.py:195`, `203`). The contract records the signed measurement-time offset (`mimiciv_observation_transition_extract.py:655–658`), and the consumer uses `horizon + abs(offset)` for availability (`mimic_to_eicu_49_cell_external_audit.py:122–125`). This conservatively waits until an equally nearer observation could no longer arrive. Actual v9 scheduling remains unverified.

- **Medium — Hourly labels represent intervals, not instantaneous measurements.** An hourly signal label describes the preceding hour. At the one-hour horizon, an earlier nearest endpoint can give a label window overlapping the pre-anchor period when anchors and aggregate boundaries are misaligned. Freeze this endpoint-based interpretation explicitly; it is not a future point measurement.

**Variable mapping assessment**

| Mapping | Assessment |
|---|---|
| Chemistry and blood-gas pooling | Consistent in principle with the multiple-item MIMIC definitions for electrolytes, hemoglobin, hematocrit and bicarbonate (`mimiciv_observation_transition_extract.py:56–70`). Simultaneous-source resolution still needs finding 3’s fix. |
| Bicarbonate 451/456/666 | Appropriate **if** the comments correctly identify actual bicarbonate. Excluding standardized bicarbonate is sensible. Dictionary identities are not independently verifiable here (`sicdb_observation_transition_extract.py:67`). |
| pH 538/688; PaCO₂ 443/687; PaO₂ 444/689 | No demonstrated mapping error, conditional on the stated arterial, uncorrected definitions (`64–66`). MIMIC’s extraction applies no specimen-type restriction, so arterial-only SICdb selection is a disclosed sampling difference rather than proven exact specimen equivalence. |
| Urea → BUN | `28.0134 / 60.056` is correct for urea **mg/dL** → nitrogen **mg/dL** (`78`). |
| Phosphate, calcium, magnesium | Factors **3.097**, **4.008**, **2.431** are correct for the stated mmol/L inputs and corresponding MIMIC mg/dL conventions; phosphate’s factor expresses phosphorus mass (`72–75`). |
| CRP; troponin T | **×10** for mg/dL → mg/L and **×0.001** for ng/L → ng/mL are correct (`88–90`). |
| Heart rate; respiratory rate | ECG heart rate is appropriate in principle. Respiratory rate pools three items, as MIMIC also does (`94–95`; MIMIC `109–110`). The supplied comments do not establish each respiratory item’s exact semantics; retain dictionary evidence before freezing. |
| MAP | Measured arterial/noninvasive means match the MIMIC approach. Derived MAP is a surrogate, requiring findings 3–4’s fixes. Equal hourly timestamps do not prove SBP and DBP were measured in the same minutes. |
| Urine output | A documented one-hour volume sum is numerically mL/h and should **not** be divided by the gap between available hourly rows. MIMIC divides recorded volume by its preceding collection interval and excludes the first event (`osler_jepa/urine_output.py:28–32`, `67–86`). Units align; averaging intervals and measurement processes differ. |
| Temperature | Passing item 709 through unchanged is correct conditional on its stated degrees-C unit (`97`). |
| Glucose, FiO₂, PEEP | Keeping chemistry glucose separate from blood-gas glucose, leaving POC glucose unavailable, and excluding FiO₂/PEEP with undeclared units are appropriately conservative (`21–23`, `34`, `62–63`). |

**Low — Correct the WBC/platelet unit comments.** `g/L` denotes mass concentration; a gigacount convention should be written unambiguously as `10^9/L` (`sicdb_observation_transition_extract.py:79–82`). This is not evidence that the numerical factor is wrong.

**Medium — Mapping verification remains incomplete.** The dictionary, including units and LOINC evidence, is absent. `LaboratoryType` is ignored (`sicdb_observation_transition_extract.py:142–144`); its significance cannot be determined from this folder. Retain dictionary-only evidence for mapped identities, specimen distinctions and this field’s meaning before freezing. Do not change factors based on target-value distributions. The four retained CRP measurements and zero six-hour pairs are support observations, not proof of a mapping error (`results/sicdb_observation_transitions_6h_v1.json:32`, `87`).

**Stay and subject assessment**

CaseID as stay and PatientID as subject match the supplied definitions. Joining observations by CaseID prevents routine cross-case attachment (`sicdb_observation_transition_extract.py:116–117`, `133`). Keeping PatientID across readmissions is appropriate for patient clustering; it makes finding 2’s chronology check necessary.

The window includes intensive **and intermediate** care. It represents the documented case window, not reconstructed continuous occupancy of a particular ICU ward. Disclose that population boundary.

`selected_stays` exceeding emitted `stays` is explainable: eligibility requires only the horizon, while the first anchor also requires six hours of history (`sicdb_observation_transition_extract.py:114`; `mimiciv_observation_transition_extract.py:601–602`). That discrepancy alone does not demonstrate lost anchors.

**Verification and release boundary**

All **11 Python files parsed successfully**. Synthetic in-memory executions reproduced the duplicate and derivation failures above. These executions used extracted functions; they did not validate DuckDB CSV parsing or the complete build.

The full tests could not run: this environment lacks `duckdb`, and the folder lacks imported modules including `eicu_body_system_configs` and `eicu_sepsis_transition_extract`. The complete cleaning bounds, history builder, v9 replay, machine-readable protocol and addendum implementation are also unavailable here. I therefore cannot certify integration behavior or one-shot enforcement.

Freeze only after deterministic measurement resolution, validated derivations, an explicit aggregate-boundary policy, and chronology/care-unit replay checks pass. Then regenerate and hash the final outputs before opening labels. The remaining sample-time, interval-measurement and population differences require explicit disclosure; they do not establish clinical availability, causal validity or promotion eligibility.