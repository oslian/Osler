# Event Lockbox v11 Results (2026-10-29)

- **Protocol:** `adaptive_event_lockbox_protocol_v11_20261029.json` (sha256 `4ff30975…2ac0`).
- **Scoring:** each database was scored once, with the user's approval. Ledger steps: `binary_SICdb` and `binary_HiRID`.
- **Outputs:**
  - `site_adaptive_event_lockbox_{SICdb,HiRID}_v11.json`
  - logs: `site_adaptive_event_lockbox_{SICdb,HiRID}_v11.log`
  - runner: `site_adaptive_event_lockbox_v11_20261029.sh`

## Primary endpoint: robust cells per database (deployment form = static + OPS)

| | SICdb | HiRID |
|---|---|---|
| online_platt (deployment form) | **2/2** (5/5 samples per cell) | **2/2** (5/5 samples per cell) |
| static (reference, not the deployment form) | 0/2 | 1/2 (1h: 4/5; 3h: 2/5) |

**Decision rule.** `hypoxemia_event@1h` and `hypoxemia_event@3h` are robust on both SICdb and HiRID. Within the claim scope, they are therefore **independently replicated event cells**.

**What "independently replicated" covers here:**
- The method is the frozen static MIMIC model plus online Platt scaling.
- The evidence is a retrospective replay reconstructed by observation time.
- Calibration is checked pooled at the site level.

**Limits that still apply:**
- `independent_validation: false`.
- `causal_claim_allowed: false`.
- Registry changes require user approval.

## Gate details (online_platt; ranges over 5 samples)

| Cell | Patients | Brier Δ vs persistence (upper bound) | Brier Δ vs prevalence + OPS (upper bound) | CITL | Slope | O/E |
|---|---|---|---|---|---|---|
| SICdb 1h | 7,677 | −0.009 to −0.006 (≤ −0.005) | −0.013 to −0.010 (≤ −0.009) | −0.005 to 0.011 | 0.991–0.998 | 0.997–1.008 |
| SICdb 3h | 7,690 | −0.015 to −0.012 (≤ −0.010) | −0.009 to −0.007 (≤ −0.005) | −0.004 to 0.007 | 0.992–1.002 | 0.997–1.005 |
| HiRID 1h | 11,445 | −0.022 to −0.021 (≤ −0.019) | −0.011 to −0.010 (≤ −0.009) | −0.004 to 0.008 | 0.989–1.000 | 0.997–1.006 |
| HiRID 3h | 11,620 | −0.028 to −0.025 (≤ −0.023) | −0.008 to −0.006 (≤ −0.005) | −0.002 to 0.002 | 0.988–0.997 | 0.998–1.002 |

Event rates:

| | 1h | 3h |
|---|---|---|
| SICdb | 5.6% | 5.9% |
| HiRID | 7.7% | 8.2% |

Static model calibration:
- **SICdb:** CITL −0.72 to −0.27, slope 1.22–1.76. It fails calibration, which is consistent with what the development databases showed.
- **HiRID:** much closer to calibrated.

## Diagnostics (not gated)

- **Patient-cluster bootstrap 95% CI** (pooled across all four cells and samples):
  - slope 0.95–1.04;
  - CITL −0.065 to 0.076.
- **Calibration by stream phase:**
  - first-third slope 0.975–1.014 in every cell;
  - no early under-adaptation like eICU's (0.75–0.86).
  - Each of these is a single site with a large stream, so OPS adapts quickly.
- **Calibration by care unit:**
  - HiRID has only one care-unit label. Care-unit heterogeneity is therefore **not testable** in HiRID.
  - SICdb has 4 units:
    - 1h: all within tolerance (CITL −0.17 to 0.18).
    - 3h: CITL −0.33 to 0.25. Six of 20 unit × sample combinations fall outside tolerance (INBD, INIC).
  - This agrees with the development finding: pooled site-level calibration does not mean each unit is calibrated.

## Interpretation and limits

1. **Prior use of the databases.** Both were already used under v10 for the continuous cells. This was a new, frozen binary-task evaluation, not a first look at untouched databases.
2. **Post-hoc decisions.** OPS and the calibration tolerances were adopted post hoc after development; the development databases are in-sample for those choices. This lockbox is the first evaluation of these choices on held-out data.
3. **Where the gain comes from.** Static already beats both comparators on Brier in every sample, at both sites (point_pass 100%). OPS mainly corrects calibration.
   - Against static, its Brier is up to about 5% lower in SICdb and about 0–1.5% lower in HiRID.
   - In HiRID 3h, sample 0, OPS is slightly higher than static (0.06389 vs 0.06386).
4. **Data caveats.**
   - SICdb SpO₂ is an hourly aggregate.
   - The HiRID copy is from a third party and cannot be verified against official checksums.
   - SICdb provenance is still not confirmed.
5. **Retrospective only.** This does not show what would happen in prospective deployment. Real-time data latency and later corrections are not modeled.

## Registry update (user-approved, 2026-10-29)

The registry was rebuilt as `adaptive_replication_registry_20261029.json`, schema v2. It now has two sources:
- the v10 continuous cells, under method `point_and_dtaci_adaptive`;
- the v11 event cells, under method `online_platt`.

**Status counts:** 37 independently replicated (previously 35), 10 candidate, 8 not promoted.

**Integrity checks:**
- The two hypoxemia cells are accepted only after the SHA-256 of their lockbox output matches the output recorded in the v11 ledger receipt `binary_<db>`.
- The 53 v10 cells are identical to the previous registry.

**Other changes and records:**
- `adaptive_cell_weights.py` now reads the continuous source of the registry. Its exported cells are still the same 35 as the published weights.
- `adaptive_replication_registry_20261028.json` is kept as a historical record: `adaptive_cell_weights_20261028/MANIFEST.json`, the manifest of the published weights, cites its sha256.
