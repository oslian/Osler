# Coverage Reconstruction: Complete Input Guide / 完整輸入指南

## 1. Scope / 先確認模型用途

This package includes actual epoch-10 weights, fit-only scalers, matching source,
an event-CSV history builder, prediction CLI and synthetic example. It reconstructs
a masked **current** value from other known measurements. It does NOT forecast
3h/6h/12h outcomes, attach calibrated intervals, recommend treatment or replace
the validated forecasting runtime.

這是「遮住一個指標，用其他已知觀測估計當下值」的研究候選，不是未來
預測器。`candidate_only=true`、`causal_claim_allowed=false`。
四個獨立訓練 fold 為0、1、3、4；fold2擬合與fold0等價。不是ensemble，
沒有經驗證的新病人fold選擇器，不可用測試答案挑fold。

The September30 archive could load weights but omitted its documented history
builder. This input-complete release fixes that delivery gap without changing
model tensors. Completed five-fold results are included in
`RECONSTRUCTION_CONTINUATION_OUTER_FINDINGS_20260930.md`: reused development,
NOT a new untouched lockbox or a clinical accuracy guarantee.

## 2. Download and run / 下載到實際輸出

Download from this SAME release:

- `coverage_candidate_source.tar.gz`
- `coverage_fold0_real_epoch10.pt` (fold0 for this reproducible example)
- `MANIFEST.json`

Use the attached archive, NOT GitHub's automatic Source code ZIP. Extract into
an empty directory and put the checkpoint there. Run commands from that root.
No real patient rows, credentials, optimizer or private cache is included.

```sh
tar -xzf coverage_candidate_source.tar.gz
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.load_coverage_candidate coverage_fold0_real_epoch10.pt
python -m unittest tests.test_coverage_input -v
python -m scripts.predict_coverage_csv \
  --checkpoint coverage_fold0_real_epoch10.pt \
  --events examples/coverage_synthetic_events.csv \
  --stay-id demo-1 \
  --anchor '2100-01-01 02:00:00' \
  --target bicarbonate
```

The demo deliberately contains no bicarbonate measurement. No dummy target value
is required. All demo rows are invented to test execution, NOT prediction accuracy.

Before loading, verify downloaded assets against the manifest:

```sh
python -c 'import json,hashlib,pathlib; m=json.load(open("MANIFEST.json")); [print(n, "OK" if hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==v["sha256"] else "HASH MISMATCH") for n,v in m["files"].items() if pathlib.Path(n).exists()]'
```

Do not continue on a mismatch. Download any missing asset you intend to use.
The loader uses `weights_only=True` and strict state loading.

For a full packaging parity check, download all four checkpoints and run:

```sh
python -m scripts.verify_coverage_bundle --report BUNDLE_VERIFICATION.json
```

It checks all asset hashes, all47 trained outputs across all4 folds, exact input
tensor/prediction parity with the training builder, and that project modules
load from this extracted directory rather than an author's checkout. Only
synthetic observations are used; this does not establish predictive accuracy.

## 3. CSV columns / 每欄要填什麼

Input is a **long-form canonical measurement-event CSV**, not an arbitrary wide
spreadsheet or an unconverted MIMIC/eICU table. One row means one measurement
becoming available. `--stay-id` selects exactly one encounter from the file.

| Column | Content / 填法 |
|---|---|
| `stay_id` | Local encounter identifier, e.g. `patient-1-stay-1`. Same admission uses the same ID; it is not a numeric model feature. |
| `source_kind` | Exactly `lab_available` or `bedside_available`. Drugs/orders are rejected. |
| `source_id` | Stable original measurement ID within stay and source kind. Revisions retain the ID with later availability. A genuine new test gets a new ID even if its value is unchanged. |
| `event_time` | Time the result actually became available, NOT automatically specimen/chart time. |
| `measured_at` | Actual collection/measurement time; cannot be after `event_time`. |
| `variable` | Exact canonical name from `--schema`, such as `glucose`, `map`, `creatinine`. No guessed aliases. |
| `value` | Finite physical-unit number before normalization. Measured zero is `0`; omit unmeasured rows, never fill missingness with zero. |
| `unit` | Exact canonical unit from `--schema`, case-sensitive. Convert/qualify upstream. Mismatches fail instead of guessing. |
| `quality_valid` | `true` only after identity, unit and quality qualification; otherwise `false`. Do not mark everything true to bypass validation. |

```csv
stay_id,source_kind,source_id,event_time,measured_at,variable,value,unit,quality_valid
patient-1-stay-1,lab_available,lab-100,2100-01-01 01:00:00,2100-01-01 00:40:00,creatinine,1.2,mg/dL,true
patient-1-stay-1,bedside_available,chart-200,2100-01-01 01:10:00,2100-01-01 01:10:00,map,80,mmHg,true
```

List all53 input names, units and which47 are trained reconstruction outputs:

```sh
python -m scripts.predict_coverage_csv \
  --checkpoint coverage_fold0_real_epoch10.pt --schema
```

An input variable is not automatically an eligible target. Unsupported `--target`
fails. This package is not the older urine-output forecast interface.

## 4. Time, missingness and history / 時間、缺失、歷史

- All timestamps must be timezone-naive in one documented source time basis.
  Normalize timezone/DST upstream; do not mix shifted/unshifted dates.
- Anchor is the current-value query time. Only `event_time <= anchor` is used;
  future results cannot enter even if already in the CSV.
- Replay window is `(anchor - 48 hours, anchor]`, matching training.
- Acquisition-to-availability lag must be at most1h for an observation to enter
  a token. Earlier valid event groups can remain in the48h history; they are NOT
  copied forward and mistaken for new measurements.
- Known revisions/conflicts are resolved before replay. Simultaneous measurements
  share one event group; true timestamps are retained.
- The target, its complete history and near-algebraic siblings are always hidden.
  Target-only groups disappear after masking. No independent valid history means
  an explicit error/no estimate, not a fabricated value.
- Actual zero has `mask=1`. Missing/invalid/stale observations are not presented
  as physical zero. No generic6h carry-forward is synthesized.
- Native clocks keep the trained convention: hours since `2100-01-01`, measurement
  ages in days, event clocks in days/weeks. The builder computes them. Do not
  manually shift dates or concatenate clock features to improve a result.

Do not invent a target event to force a label at anchor. Inference needs no target
truth. Accuracy evaluation DOES need an independent appropriately timed measured
target and the audit's eligibility policy. Arbitrary sparse/new-patient query
times are not automatically validated use cases.

## 5. Python API / 程式介面

```python
import torch
from scripts.load_coverage_candidate import load
from whole_body_jepa.coverage_input import read_events, build_input

model, checkpoint = load('coverage_fold0_real_epoch10.pt')
events = read_events('examples/coverage_synthetic_events.csv')
x = build_input(events, checkpoint, 'demo-1', '2100-01-01 02:00:00', 'bicarbonate')
with torch.inference_mode():
    normalized = model(x['tokens'], x['event_hours'], x['anchor_hour'], x['target'])[0]
c = checkpoint['variables'].index(x['target'])
physical_value = float(normalized * checkpoint['scaler']['scale'][c]
                       + checkpoint['scaler']['location'][c])
```

`build_input` already applies fit-only normalization. Do NOT normalize its output
again. The included parity test compares its tensors with the original
`native_reconstruction_examples` training builder plus `transform_example`.

For N=53 variables and M checkpoint modules, width is `3*N + M + 8`:
values[N], masks[N], measurement ages[N], clock[1], module presence[M], zero
treatment fields[6], final clock[1]. Order comes from the checkpoint. Presence
is recomputed after masking. You do NOT fill the old311feature vector, body
latent, belief latent, `target_future`, organ or `base_point`; those belong to
a different model interface.

## 6. Output / 如何解讀

JSON contains `current_estimate`, canonical `unit`, anchor, independent event
group count, newest independent event age, source fold and candidate status.
`lower`/`upper` are null; `interval_status=not_calibrated`. Null is not a zero-width
interval. No90% coverage is claimed for this reconstruction package.

The neural estimate is not silently clipped to look physiologically plausible.
Implausible results show candidate limitations and are not usable clinical advice.
No treatment-effect claims are allowed.

## 7. Common errors / 常見問題

| Error | Check |
|---|---|
| Missing module/native builder | Use this input-complete source archive and its extracted root; do not mix releases. |
| Unit mismatch | Run `--schema`; a glucose label alone does not establish mg/dL versus mmol/L. |
| No independent observed history | Other values may be missing, stale, conflicted, outside48h or siblings that must be hidden. Do not bypass masking. |
| Wrong width/state loading | Use matching code/weights; do not manually build a311vector. |
| Target is not trained | Check `reconstruction_target` in schema. Unsupported tasks need another validated model. |
| No events known at anchor | Check availability times, encounter ID and anchor; never substitute future results. |

## 8. What remains upstream / 資料來源端仍須完成

Hospital-specific extraction and unit/availability qualification remain necessary.
The bridge does not guess MIMIC item IDs, eICU names, specimen times, source timezone
or clinical quality. Raw clinical datasets require separate licensed access and
are not bundled. Historical training scripts provide provenance, not one-command
full-cohort retraining without private data. The runnable CSV path above requires
only the download assets and your canonical event CSV, not the author's SQLite,
patient-index PT files, local model registry or private filesystem.
