# Multi-Sample Robustness of External Gates (2026-10-16)

Script: `multi_sample_external_audit.py`. Every cell was refit on 5 MIMIC patient samples of 15,000
rows; sample 0 reproduces the original audit sample, and all four runs reproduced the original
single-sample results exactly. Predeclared rule: a cell is **robust** when it passes the full gate
(point gate and v2 value-conditional interval) in at least 80% of samples. Both external databases
were already used, so these are non-independent re-checks.

## Single-sample vs robust counts

| Data version | NWICU single → robust | eICU single → robust | Robust on both |
|---|---|---|---|
| Promotion version (MIMIC v2, original eICU/NWICU files) | 30 → **29** | 16 → **9** | 8 |
| v4 (harmonized definitions, chemistry-only glucose) | 29 → **28** | 17 → **13** | **10** |

- NWICU (general ICU, like MIMIC) results are stable: one cell lost in each version.
- **eICU single-sample passes were inflated by sampling luck** (promotion version 16 → 9).
- **Measurement harmonization holds up under resampling.** Robust eICU cells rise from 9 to 13,
  and cells robust on both databases from 8 to 10. Respiratory rate at 3h, 6h and 12h becomes
  robust on eICU with nurse-charted vitals.

Robust on both databases, v4: hypoxemia_event@1h, hypoxemia_event@3h, respiratory_rate@3h,
heart_rate@6h, respiratory_rate@6h, respiratory_rate@12h, heart_rate@24h, map@24h, sodium@24h,
heart_rate@48h.

## Promoted cells (registry pinned to the promotion version)

Rule per tier: A must be robust on NWICU and eICU; B robust on NWICU; C point gate passing in
≥ 80% of NWICU samples.

| Cell | Tier | NWICU full share | eICU full share | Result | Proposed |
|---|---|---:|---:|---|---|
| map@3h | A | 0.8 | 0.6 | not robust on eICU | **A → B** |
| respiratory_rate@3h | A | 1.0 | 0.6 | not robust on eICU | **A → B** |
| respiratory_rate@6h | A | 0.8 | 0.6 | not robust on eICU | **A → B** |
| bicarbonate@24h | A | 0.8 | 0.2 | not robust on eICU | **A → B** |
| creatinine@48h | A | 1.0 | 0.4 | not robust on eICU | **A → B** |
| glucose@24h | A | 0.4 | 0.2 | robust on neither; NWICU point share 1.0 | **A → C** |
| bicarbonate@12h | B | 0.6 | 0.0 | not robust on NWICU; NWICU point share 1.0 | **B → C** |

The other 19 promoted cells meet their tier rule. All 5 C cells, including the 2 proposed, have a
NWICU point-gate share of 1.0.

Proposed tiers after review: A 7 (hypoxemia_event@1h, hypoxemia_event@3h, heart_rate@6h,
potassium@12h, potassium@24h, sodium@24h, heart_rate@48h), B 14, C 5. Not applied; needs the
user's approval.

`causal_claim_allowed: false`. No registry was changed.
