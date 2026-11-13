# Why Short-Horizon Glucose Got Worse After Harmonization (2026-10-15)

After harmonization the eICU glucose target is laboratory-only, and short-horizon coverage fell
(1h 0.856 → 0.769, 3h 0.842 → 0.785, 6h 0.802 → 0.760) while 24h and 48h improved.

## 1. Not the new `glucose_poc` input

Same harmonized eICU cohorts, MIMIC recipe with and without the `glucose_poc` feature (MIMIC v2
vs v3 data): 1h 0.777 vs 0.769, 3h 0.771 vs 0.785, 6h 0.758 vs 0.760, 12h 0.802 vs 0.801, 24h 0.888
vs 0.887, 48h 0.872 vs 0.874. The input has essentially no effect.

## 2. The eICU cohort became small and selected

Laboratory-only label pairs at short horizons are rare in eICU: 1h 333 rows (was 8,510), 3h
1,022 (was 15,004), 6h 1,483 (was 13,791). At 24h and 48h the cohort size is unchanged. These are
patients whose laboratory glucose is redrawn within hours, a volatile subgroup. The earlier
source diagnostic already showed laboratory→laboratory eICU rows at 0.76–0.79.

## 3. MIMIC "laboratory glucose" includes blood gas; eICU does not

- MIMIC ICU laboratory glucose: chemistry 562,894 results, blood gas (50809) 182,862 (24.5%).
- eICU laboratory glucose names: `glucose` (chemistry), `bedside glucose`, `glucose - CSF`; no
  blood-gas glucose.
- NWICU: 100001 chemistry and 100045 blood gas.

MIMIC held-out coverage by source pair (v2 interval, `glucose_source_mimic_diagnostic.py`):

| Cell | Blood gas → blood gas (share, coverage) | Chemistry → chemistry (share, coverage) | All |
|---|---|---|---|
| glucose@1h | 32%, 0.923 | 46%, **0.858** | 0.891 |
| glucose@3h | 20%, 0.942 | 56%, 0.892 | 0.907 |
| glucose@6h | 13%, 0.921 | 66%, 0.884 | 0.899 |
| glucose@12h | 11%, 0.934 | 65%, 0.904 | 0.913 |
| glucose@24h | 9%, 0.926 | 71%, 0.897 | 0.903 |
| glucose@48h | 9%, 0.899 | 71%, 0.890 | 0.887 |

The MIMIC interval is calibrated on a blood-gas/chemistry mixture: blood-gas pairs (frequent,
stable redraws with arterial blood gases) are over-covered and chemistry pairs under-covered, most
strongly at short horizons. eICU has only chemistry glucose, so it inherits the under-covered side.

## Decomposition at 1h

About 0.04 of the gap is mixture miscalibration: MIMIC chemistry→chemistry is 0.858, against
0.891 overall. The rest (0.858 → 0.769) is eICU selection: only 333 redraw pairs, a more volatile
subgroup.

## Proposed refinement (needs approval)

Define the glucose target as **chemistry glucose only** in every database and move blood-gas
glucose to a separate input (`glucose_bg`): MIMIC 50809, NWICU 100045. This changes the definition
approved on 2026-10-15 ("laboratory or blood gas") and needs another MIMIC and NWICU rebuild. The
expected effect is partial: it removes the mixture miscalibration (about 0.04 at 1h), not the
eICU selection effect. Short-horizon glucose cells would probably still fail on eICU.

`causal_claim_allowed: false`. MIMIC-only diagnostic plus non-independent eICU re-checks.

## Update 2026-10-16: chemistry-only glucose target implemented (v4 data)

Approved and implemented: MIMIC `glucose` = 50931, 52569 (chemistry); `glucose_bg` = 50809, 52027
(blood gas, separate input). NWICU `glucose` = 100001; `glucose_bg` = 100045. eICU unchanged (no
blood-gas glucose). All 15 eICU modules are now in `eicu_harmonized_20261015/` (68 files).

eICU (v2 interval, v3 → v4 data):

| Cell | Coverage | Point gate | Full |
|---|---|---|---|
| glucose@1h | 0.769 → **0.811** | fail → **pass** | fail |
| glucose@3h | 0.785 → 0.779 | fail | fail |
| glucose@6h | 0.760 → 0.784 | fail | fail |
| glucose@12h | 0.801 → 0.819 | fail | fail |
| glucose@24h | 0.887 → **0.816** | pass → **fail** | pass → **fail** |
| glucose@48h | 0.874 → 0.888 | pass | fail → **pass** |

Totals: eICU full pass 18 → 17 (glucose@48h gained; glucose@24h and potassium@12h lost); NWICU
29 → 29. Non-glucose cells changed by at most 0.007.

As predicted, the short-horizon gain at 1h is about +0.04 (mixture miscalibration removed); the
eICU selection effect remains.

### glucose@24h regression and sampling sensitivity

- Not the new inputs: removing `glucose_bg`, `glucose_poc`, or both leaves eICU MAE at 41.7–42.0
  against persistence 41.8. MIMIC fingerstick and eICU bedside glucose agree with laboratory
  glucose similarly (median |POC − lab| 17 vs 12 mg/dL; r = 0.79 vs 0.86).
- **Sampling sensitivity.** Refitting on different 15,000-row MIMIC samples:
  v3 eICU MAE 36.2–36.7, coverage 0.861–0.887; v4 eICU MAE 37.3–41.9, coverage 0.816–0.858 (40,000
  rows: 37.3, 0.842). The chemistry-only MIMIC cohort (16% fewer pairs) yields a less stable
  recipe, and v3's 0.887 "pass" was a favourable sample: other samples fall below 0.87.

### General implication

All audits in this series fit each recipe on one 15,000-row MIMIC sample. Near the gates, that
choice alone moves coverage by about 0.01–0.03 and MAE by up to about 4 mg/dL for glucose@24h.
Future audits and promotion decisions should report pass frequency over several MIMIC samples
(or use larger caps) rather than a single fit.
