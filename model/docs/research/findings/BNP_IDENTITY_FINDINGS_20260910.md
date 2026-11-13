# Analyte Identity Repair

The mapping incorrectly assigned MIMIC item50963 to BNP. The local MIMIC3.1
`d_labitems` dictionary calls it NTproBNP, Blood. This is a confirmed identity
error, not a conclusion drawn from a high median. The extractor now maps it to
`nt_pro_bnp`; canonical units retain pg/mL without aliasing to BNP. The existing
53-variable model has no NT-proBNP slot and must not fill BNP with it.

BNP and NT-proBNP are distinct tests; see the primary
[Mayo NT-proBNP catalog](https://www.mayocliniclabs.com/test-catalog/overview/615897)
and [BNP catalog](https://www.mayocliniclabs.com/test-catalog/Overview/800006).
There is no fixed numeric conversion in this repair.

## Matched Test

`BNP_IDENTITY_PROTOCOL_20260910.md`, fixed source-supported identity candidates,
same6,616 MIMIC pairs/889 patients; calcium protection remains in BOTH arms.
Only82 cached observations are relabeled. Depending on seed,320-321 future
predictions change through the history window. Patient-equal MAE mg/dL:

| Mode | Before median | After median | Significant improvement |
|---|---:|---:|---:|
| Whole unbounded | 33.323 | 33.337 | 0/3 |
| Whole unbounded-balanced | 33.324 | 33.339 | 0/3 |

All paired95% CIs cross0. Correct identity did not improve glucose accuracy in
this test, so it is NOT supported as the main residual error source. The
mapping remains corrected regardless of predictive utility. eICU BNP was not
changed. No retraining, runtime weights, calibrator or registry change.

## Cache / Reproducibility

`artifacts/bnp_identity_repair_20260910.json` records exact results and bindings.
The standard local availability MIMIC cache is now corrected (0 BNP,82
NT-proBNP records). The original is explicitly archived at
`/private/tmp/availability_jepa_mimic_before_bnp_identity.parquet` for historical
audit reproduction. Earlier reports used the old cache; do not reinterpret
their numbers as corrected-input evidence. Future runs of the fixed extractor
also preserve the correct identity. No row-level data entered version control.

## Broader Screen

A raw accepted-lab median/scaler screen across all supported checkpoint
variables identifies additional shifts, not automatically errors. MIMIC direct
bilirubin median2.9 versus scaler center0.2 and scale0.148 can also reflect
selective testing of sicker patients; no unit mismatch was established and no
correction was guessed. The complete item-label inventory also contains both
calculated and measured methods (e.g., osmolality, hematocrit, bicarbonate).
Method provenance must be distinguished before calling all these labels direct
measurements; their predictive effect was not tested here.

296 tests pass. Candidate status remains; whole-vs-own transfer and full gates
remain unresolved. This repair addresses an actual extractor error, not a
claim that all generalization problems are now solved.
