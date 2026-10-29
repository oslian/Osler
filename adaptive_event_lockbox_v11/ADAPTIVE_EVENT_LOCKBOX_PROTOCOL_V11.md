# Event Lockbox Protocol v11 (frozen)

> **Status: frozen** on 2026-10-29 with the user's approval (frozen_at 2026-10-29T16:48:11Z).
> - Machine-verifiable version: `adaptive_event_lockbox_protocol_v11_20261029.json`, sha256 `4ff309754d5d1ad308f5891416bffb31f22b879085f32a59ccb28d32af5d2ac0`. It hashes 52 code/config files plus 2 MIMIC training files.
> - After the freeze, `verify_protocol` passed.
> - Ledger: neither the addendum nor the scoring step has been claimed.
> - Right before the freeze, the builder added two items: the Codex review entry under post-hoc decisions, and the care-unit CITL diagnostic. Both match this document.

## Question

Two hypoxemia event cells are deployed as follows:
- the frozen MIMIC static model;
- plus online Platt scaling (OPS).

In a retrospective replay reconstructed by observation time, do they pass the event gate on SICdb and HiRID?

## Scope

- **Scored cells:** `hypoxemia_event@1h` and `hypoxemia_event@3h` only (SpO₂ ≤ 92% at t + h). Task type `binary_event_probability`.
- **Databases:** SICdb and HiRID.
- **Previous protocol:** v10 (`adaptive_model_lockbox_protocol_v10_20261027.json`), whose hash is included in the v11 hash list.

## Prior use of these two databases (disclosure)

- **Already scored under v10:** both databases were scored on continuous cells, and the results are known:
  - SICdb 44/47, HiRID 36/38;
  - both include `o2sat@12h`.
- **SpO₂ in v10:** current SpO₂ and its history were model features in v10. The 1h/3h future SpO₂ columns were never used as labels and never viewed.
- **How this protocol should be described:** a newly frozen binary-task evaluation on external databases that have already been used. **Not** "never-touched databases."
- **HiRID source:** HiRID is the third-party copy and cannot be verified against official checksums (see `HIRID_LOCKBOX_PROVENANCE_20261028.md`).
- **SICdb source:** not yet confirmed.

## Deployment form

**Static model and OPS**
- Static model: `_fit_hypoxemia`, refit on each of 5 MIMIC training samples.
- OPS: Gupta & Ramdas (2023), Algorithm 1.
  - γ = 0.1, A0 = 100·I, radius 100.
  - Starts from (1, 0), i.e. the unmodified frozen model.

**Replay**
- Each hospital is its own deployment on the shared deployment clock.
- An outcome is used to update OPS only after the clock passes anchor + h + |label offset|.
- Warm-up stays are not scored.

**Deployment parameters (must match exactly at scoring time)**

| Parameter | Value |
|---|---|
| mimic_template | `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet` |
| development_report | `mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json` |
| event_config | `site_adaptive_event_amendment_20261028.json` |
| samples | 5 |
| max_rows_per_database | 60000 |

## Gate

1. **Brier:**
   - Must beat two baselines:
     - persistence (current SpO₂ ≤ 92);
     - MIMIC prevalence after the same OPS update.
   - Pass criterion: upper bound of the patient-level paired bootstrap < 0.
2. **Calibration:**
   - The scored stream is pooled. Point estimates must satisfy CITL ∈ [−0.2, 0.2] and slope ∈ [0.8, 1.25].
   - These are research tolerances, not a statistical equivalence test.
   - If calibration is not identifiable, or any prediction is invalid, the sample fails.
3. **Robust:** at least 4 of the 5 MIMIC samples must pass both 1 and 2.
4. **Diagnostics (not gated):**
   - patient-cluster bootstrap CIs;
   - calibration by hospital, by care unit, and by stream phase.

## Decision rule

- Robust in both SICdb and HiRID: an **independently replicated** event cell within the claim scope.
- Robust in only one: candidate.
- Any registry change requires user approval.

## Claim scope

- **Retrospective replay** reconstructed from final values by observation time. It does not mean the results would have been available at that moment in a prospective setting.
  - HiRID may include corrections entered later.
  - SICdb's SpO₂ is an hourly aggregate, so a threshold on it is not the same as instantaneous hypoxemia.
  - In HiRID, one patient ID corresponds to one admission.
- **Pooled calibration:** supports only site-level claims. It does not mean every care unit is well calibrated.
- **Replay properties disclosed in v10** still apply.

## Post-hoc decisions (disclosure)

1. The calibration tolerances were set after the static calibration check (eICU, NWICU, Emory).
2. MIMIC-only selection under the pre-declared rule chose static (4 vs 4 streams). OPS was adopted afterwards by amendment (`site_adaptive_event_amendment_20261028.json`).
3. Codex pre-freeze review: all 7 findings fixed (`CODEX_REVIEW_EVENT_PREFREEZE_20261029.md`).

## Development reference (in-sample, not a prediction)

- **Robust cells:**

  | | eICU | NWICU | Emory |
  |---|---|---|---|
  | OPS | 2/2 | 2/2 | 2/2 |
  | Static | 0/2 | 0/2 | 0/2 |

  Source: `site_adaptive_event_{eICU,NWICU,Emory}_20261029.json`.
- **Diagnostics:**
  - eICU:
    - in each hospital's first third of the stream, OPS slope is 0.75–0.86 (OPS has not adapted yet);
    - in the last third, slope recovers to 0.92–0.97;
    - about one third of eICU hospitals with estimable calibration fall outside the tolerances.
  - The single-site databases are stable from the first third.
  - Care-unit CITL ranges up to ±0.66 (NWICU).

## Execution order (each step requires user approval)

1. **Freeze:** `build_event_lockbox_protocol_v11.py` writes the protocol JSON (write-once).
   - It hashes 52 code/config files, including v10.
   - It also hashes the MIMIC 1h/3h training files.
2. **Addendum:** for each database:

   ```
   lockbox_addendum.py --protocol <v11> --database <db> --files <extractor, its test, mapping file, the 1h/3h parquet files and reports>
   ```

   - The 1h/3h parquet files must be byte-identical to those in the v10 addendum. Scoring checks this.
3. **Scoring, once per database:**

   ```
   site_adaptive_event_audit.py --database <db> --external-template <db>_observation_transitions_{horizon}h_v1.parquet --protocol <v11> --output site_adaptive_event_lockbox_<db>_v11.json
   ```

   - Run from the project folder.
   - Ledger step: `binary_<db>`.
   - Once claimed, the step cannot be rerun. If it crashes after the claim, that database's evaluation ends unless the user decides otherwise, and the crash is reported.

`independent_validation: false`; `causal_claim_allowed: false`; `clinical_promotion_allowed: false`.

## Execution log

- **2026-10-29: freeze.** The protocol JSON was written (sha256 `4ff30975…2ac0`). `verify_protocol` passed.
- **2026-10-29: addenda written.** `addendum_SICdb` (15 files) and `addendum_HiRID` (18 files) were recorded with the user's approval.
  - The file lists are the same as v10's.
  - All digests are byte-identical to the v10 addendum, including the 1h/3h parquet files.
- **Scoring (`binary_SICdb`, `binary_HiRID`):** not run. Needs separate user approval.
- **2026-10-29: scoring done.** `binary_SICdb` and `binary_HiRID` were scored once each with the user's approval.
  - Both exited with code 0.
  - Each step's ledger receipt records its output file.
  - Results: see `ADAPTIVE_EVENT_LOCKBOX_RESULTS_V11_20261029.md`.
