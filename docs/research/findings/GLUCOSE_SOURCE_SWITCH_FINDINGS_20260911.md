# Glucose source-label mixing

Joined all178,892 accepted glucose events to native eICU labid, one-to-one.
145,830 are named bedside glucose and33,062 glucose. Names are source
categories, not independently verified sample material or assay method.
The prior canonical extraction retained IDs but discarded labname. Future
eICU extraction now preserves source_test_name; it is provenance, not a new
encoder feature. Existing caches/checkpoints are not overwritten or relabeled.

All5,438 measured current/future pairs resolved their source category. Current
identity uses stay, known-at-query time, actual measured timestamp and value;
future identity uses the stored exact event ID.

| Current source | Future source | Rows |
|---|---|---:|
| bedside glucose | bedside glucose | 3057 |
| bedside glucose | glucose | 1359 |
| glucose | glucose | 583 |
| glucose | bedside glucose | 439 |

1,798/5,438 (33.06%) switch source names. This is not itself a mislabeled target
or proof of measurement artifact. Aggregate: `artifacts/glucose_source_switch_20260911.json`.

Whole-vs-own comparison on fixed checkpoints:
- Source same-category gain0/7,harm3/7; switching gain0/7,harm1/7.
- Hospital same-category gain0/7,harm0/7, numerical benefit4/7.
- Hospital switching gain0/7,harm1/7, numerical harm7/7; paired MAE increases
  .241-3.674mg/dL. Hospital support148same,109switch patients, with overlap.

Same-category source failure remains, so mixing cannot explain all failure.
Hospital association motivates preserving provenance, not claiming causality
or choosing only same-source labels after seeing outcomes. Future assay/source
is unknown at forecast time; these are retrospective audit strata and MUST
NOT drive inference routing. No unseen-label feature or treatment claim.

Reproducibility: source metadata is private parquet joined by labid from the
local lab.csv.gz; original experiment45bindings verified before scoring and
the new metadata hash is bound in the aggregate. Row-level data remain private.
Original shared-model predictions are unchanged. Extractor provenance change
requires a fresh cache for later source-aware fitting; no accuracy repair or
runtime, interval, retention, external-lockbox or promotion claim.
