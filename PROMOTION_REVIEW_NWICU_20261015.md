# Promotion Review: 26 NWICU One-Shot Full-Pass Cells (2026-10-15)

The 49-cell development report is marked `development_rescue_complete_pending_fresh_lockbox`
("these cells require a fresh untouched lockbox before formal external promotion"). The NWICU
one-shot under `NWICU_EXTERNAL_VALIDATION_PROTOCOL_20261014.md` is that lockbox. 26/40 evaluable
cells passed its primary endpoint.

## Caveats

- The one-shot used the **v1** value-conditional interval, which has since been removed
  (over-coverage). The current **v2** interval has only non-independent re-checks (NWICU, eICU).
- The NWICU point gate is patient-axis only (single hospital). The eICU point gate also requires
  the hospital axis.
- MIMIC training data is the v2 rebuild (current extractor), not the frozen 2026-08 files.

## Evidence tiers

| Tier | Meaning | Cells |
|---|---|---|
| **A (13)** | NWICU one-shot full pass **and** v2 passes on NWICU and eICU | hypoxemia_event@1h, hypoxemia_event@3h, map@3h, respiratory_rate@3h, heart_rate@6h, respiratory_rate@6h, potassium@12h, bicarbonate@24h, glucose@24h, potassium@24h, sodium@24h, creatinine@48h, heart_rate@48h |
| **B (10)** | NWICU one-shot and v2 NWICU pass; eICU (disease cohorts) fails | hemoglobin@3h, platelets@3h, potassium@3h, creatinine@6h, map@6h, bicarbonate@12h, map@12h, o2sat@12h, respiratory_rate@12h, potassium@48h |
| **C (3)** | NWICU one-shot pass only; v2 NWICU re-check fails | creatinine@12h, hemoglobin@12h, bun@24h |

## Proposed decision (requires the user's approval)

- **A:** promote point forecast and v2 interval as "externally replicated on two databases".
- **B:** promote point forecast and v2 interval with scope "general ICU population"; flag that the
  interval under-covers in disease-enriched cohorts.
- **C:** promote the point forecast only; the interval stays uncalibrated (`lower` / `upper` null).

Promotion would be recorded in a new registry file with the same schema as
`three_level_11_external_expert_anchor_registry_20260817.json`. Nothing has been written yet.

`causal_claim_allowed: false`, `clinical_promotion_allowed: false` (research runtime only).

## Update 2026-10-16: tiers re-based on multi-sample robustness

Approved by the user on 2026-10-16. Tiers now use the five-sample robustness rule from
`MULTI_SAMPLE_ROBUSTNESS_FINDINGS_20261016.md` (`build_external_expert_promotion_registry.py`):
map@3h, respiratory_rate@3h, respiratory_rate@6h, bicarbonate@24h, creatinine@48h A → B;
glucose@24h A → C; bicarbonate@12h B → C. Result: A 7, B 14, C 5.
The registry is now `external_expert_promotion_registry_20261016.json`; the single-sample
2026-10-15 registry was deleted as superseded.
