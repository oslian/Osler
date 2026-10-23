# MIMIC-IV Waveform Precision Candidate

Status: candidate-only waveform data and feature pipeline validated. A first
held-out precision audit has now run, and **no waveform-derived feature is
validated for 6h factual forecasting yet**.

## Why This Exists

The table-based observation layer has reached the point where additional
same-source ICU rows are no longer the main accuracy lever. The remaining
high-precision signal source is MIMIC-IV Waveform: ECG, arterial pressure,
plethysmography, respiration, oxygen saturation, and related high-frequency
signals.

This is a separate modality, so it must enter through its own fail-closed
pipeline rather than being mixed into validated table features.

## Current Implementation

- `osler_jepa/waveform.py`
  - parses WFDB header text;
  - classifies signal labels into coarse physiologic groups;
  - defines candidate waveform feature families;
  - exposes a readiness contract that keeps prediction authority closed.
- `mimiciv_waveform_manifest.py`
  - scans `.hea` files under a waveform root;
  - emits aggregate signal coverage only;
  - does not read raw waveform samples;
  - does not emit row-level records or patient identifiers.
- `osler_jepa/observation_layer.py`
  - reports `mimiciv_waveform_precision_candidate` as candidate-only.

## Candidate Signal Groups

- ECG: beat-to-beat heart rate, RR variability, tachy/brady burden.
- Arterial pressure: beat-to-beat SBP/DBP/MAP, MAP variability, hypotension
  burden, pulse-pressure variability.
- Plethysmography: perfusion/pleth variability and desaturation burden.
- Respiration/capnography: respiratory variability, apnea/low-ventilation
  burden, ETCO2 variability when present.

## Promotion Gate

Before any waveform-derived feature can be used for factual prediction, it must
pass the same discipline as the rest of Osler-JEPA:

- baseline: existing table-based factual forecast;
- candidate: table-based forecast plus waveform-derived features;
- placebo: table-based forecast plus capacity-matched random features;
- validation: patient-heldout plus careunit/time or hospital-heldout where
  available;
- targets: heart rate, MAP/SBP/DBP, oxygen saturation, respiratory rate;
- authority: no causal, counterfactual, clinical, runtime, checkpoint, or active
  rule authority.

## Current Result

The MIMIC-IV Waveform Database local mirror is present at:

```text
physionet.org/files/mimic4wdb/0.1.0
```

Aggregate WFDB header manifest, run 2026-07-05:

| Metric | Count |
|---|---:|
| headers scanned | 4,059 |
| parse errors | 0 |
| ECG coverage | 3,597 |
| respiration coverage | 3,689 |
| pleth coverage | 3,369 |
| arterial pressure coverage | 1,072 |
| central venous pressure coverage | 429 |
| capnography coverage | 4 |

The header parser was also corrected for two WFDB details:

- multi-segment master headers list segment names, not signal labels;
- ECG subchannels such as `ECG #0` are canonicalized to `ECG`.

This means the data source is real and has enough ECG/pleth/respiration coverage
to justify the next engineering step. It still has **no factual prediction
authority** because no raw waveform features or held-out accuracy audit have
been run.

The manifest command is:

```bash
python3 mimiciv_waveform_manifest.py \
  --waveform-root physionet.org/files/mimic4wdb/0.1.0 \
  --output /tmp/mimiciv_waveform_manifest.json
```

## Next Engineering Step

Raw feature extraction has been opened as a bounded candidate pipeline:

- `osler_jepa/waveform_features.py`
  - computes robust channel summaries from bounded waveform windows;
  - emits ECG, arterial pressure, pleth, and respiration candidate features;
  - keeps prediction authority closed.
- `mimiciv_waveform_feature_extract.py`
  - reads local WFDB records through the optional `wfdb` dependency;
  - writes row-level feature candidates only to caller-provided output paths;
  - should use `/tmp` or another non-repository path for smoke outputs.

Bounded feature smoke, run 2026-07-05:

| Metric | Value |
|---|---:|
| records read | 100 |
| read failures | 8 |
| requested window | 60 seconds |
| median seconds read | 20.49 |
| candidate features emitted | 38 |
| ECG feature coverage | 87 / 100 |
| pleth feature coverage | 92 / 100 |
| respiration feature coverage | 97 / 100 |
| arterial pressure feature coverage | 19 / 100 |

The smoke command was:

```bash
.venv/bin/python mimiciv_waveform_feature_extract.py \
  --waveform-root physionet.org/files/mimic4wdb/0.1.0 \
  --seconds 60 \
  --max-records 100 \
  --features-output /tmp/mimiciv_waveform_features_smoke.csv \
  --summary-output /tmp/mimiciv_waveform_features_smoke_summary.json
```

This proves the raw reader and bounded feature path work. It still does **not**
prove that waveform features improve forecasts.

## Formal 6h Precision Audit

Formal held-out audit, run 2026-07-05:

| Metric | Value |
|---|---:|
| segment index rows | 2,686 |
| aligned stays | 171 |
| aligned subjects | 170 |
| candidate transition anchors | 689 |
| waveform feature rows | 689 |
| waveform read failures | 0 |
| pre-anchor waveform window | 60 seconds |
| candidate waveform features | 90 |
| ECG anchor coverage | 689 |
| pleth anchor coverage | 687 |
| respiration anchor coverage | 674 |
| arterial pressure anchor coverage | 199 |
| validated targets | 0 |

Target-level result:

| Target | Rows | Subjects | Random splits selected candidate | Held-out beats baseline | Held-out beats placebo |
|---|---:|---:|---:|---:|---:|
| heart rate | 688 | 127 | 0 / 7 | 0 / 7 | 0 / 7 |
| MAP | 643 | 124 | 0 / 7 | 0 / 7 | 0 / 7 |
| oxygen saturation | 686 | 127 | 0 / 7 | 0 / 7 | 0 / 7 |
| respiratory rate | 685 | 127 | 0 / 7 | 0 / 7 | 0 / 7 |

The formal audit command was:

```bash
.venv/bin/python mimiciv_waveform_precision_audit.py \
  --cohort /tmp/mimiciv_observation_transitions_6h_treatment_context.parquet \
  --waveform-root physionet.org/files/mimic4wdb/0.1.0 \
  --mimic-dir /Users/chunyouchang/Downloads/mimic-iv-3.1 \
  --output /tmp/mimiciv_waveform_precision_audit_peak_fullanchors.json \
  --seconds 60 \
  --max-anchors 10000 \
  --bootstrap-samples 300 \
  --min-pairs 80 \
  --min-subjects 20
```

This is a real negative result, not a pipeline failure. The local MIMIC4WDB
subset provides usable ECG/pleth/respiration data, and the reader extracted
features without failures. But in this subset, simple 60-second robust and peak
features do **not** add stable 6h factual forecast signal beyond the existing
table-plus-treatment-context baseline.

## Boundary After Accuracy Gate

Waveform remains candidate-only:

- no prediction authority;
- no runtime authority;
- no treatment, causal, or counterfactual authority;
- no checkpoint promotion;
- no active rule authority.

The result does **not** prove that waveforms are useless. It only closes this
specific candidate: local MIMIC4WDB subset + 60-second pre-anchor windows +
simple robust/peak features + 6h targets. Plausible future waveform work would
need a different target or signal layer, such as 1h forecasts/nowcasts, stronger
signal-quality and beat detectors, or a larger waveform-linked cohort.

The current environment does not have the optional `wfdb` Python package
installed by default. It was installed into the local project `.venv` for this
work and is listed in `requirements.txt`.

## Full MIMIC Short-Horizon Audit

After locating the complete MIMIC-IV v3.1 tables under
`/Users/chunyouchang/Downloads/mimic-iv-3.1`, the short-horizon experiment was
rerun on the waveform-linked cohort rather than the MIMIC demo. The waveform
master headers were aligned to ICU stays by `subject_id + hadm_id`; extracting
the numeric directory name as `stay_id` was explicitly rejected because it
does not identify the ICU stay.

The extractor was made horizon-configurable and memory-bounded:

- `mimiciv_observation_transition_extract.py` now supports `--horizon-hours`
  and preserves the original 6h default;
- `mimiciv_observation_transition_extract_batched.py` processes stay batches and
  merges local parquet parts, avoiding the full-table out-of-memory failure;
- waveform-linked support: 171 stays / 170 subjects / 2,686 waveform segments.

Full linked-cohort results:

| Horizon | Transition rows | Waveform feature rows | Candidate targets | Validated targets |
|---|---:|---:|---:|---:|
| 1h | 7,811 | 4,151 | heart rate, MAP, SpO2, respiratory rate | 0 |
| 3h | 2,689 | 1,396 | heart rate, MAP, SpO2, respiratory rate | 0 |

The 1h respiratory-rate candidate showed a useful signal in some care-unit
held-out comparisons, but it did not pass the full random patient-split plus
care-unit/time gate. The 3h results were similarly suggestive in isolated
comparisons but not promotion-grade. Therefore waveform remains
`candidate-only`; this is a stronger negative result than the earlier demo
run, because it has the correct full MIMIC waveform-linked cohort.

During the full audit, a numerical-stability issue was found in the shared
feature preparation: near-zero discovery variance could create enormous
standardized values in time-held-out rows. `eicu_full_variable_coverage_audit.py`
now applies a scale floor and finite-value clipping before Ridge fitting. The
1h audit was rerun after this correction; no target was promoted.

## Full-Target Waveform Coverage Sweep

To check whether waveform work was being limited by the original four-target
list, the complete linked transition tables were swept across all 57 available
targets at 1h and 3h. The sweep used the same 96 waveform features, patient
split logic, care-unit/time heldout checks, and placebo comparison. It was a
coverage sweep, not a relaxed promotion gate.

| Horizon | Transition rows | Targets evaluated | Waveform read failures | Validated waveform increments |
|---|---:|---:|---:|---:|
| 1h | 7,811 | 57 | 0 | 0 |
| 3h | 2,689 | 57 | 0 | 0 |

The full table-based target layer still has usable factual targets in these
cohorts, but adding the current 60-second waveform summaries did not pass the
complete gate for any target. Some isolated heldout comparisons were positive
(especially respiratory rate), but they were not stable across all required
patient splits and therefore remain candidate-only.

This closes the question "are we only testing four waveform targets?" with a
negative but useful answer: the current waveform feature family is not adding
validated predictive directions in this linked cohort. The table-based full
MIMIC target expansion is recorded separately in
`MIMICIV_FULL_TARGET_COVERAGE_FINDINGS.md`.

## Beat-Level Encoder Follow-up

The next design candidate was implemented and evaluated without changing the
cohort or the promotion gate:

- `osler_jepa/waveform_encoder.py` adds bounded ECG beat/RR, arterial-pressure
  pulse, pleth pulse, respiration-cycle, and signal-quality features;
- `mimiciv_waveform_precision_audit.py --feature-mode beat` fuses those 88
  waveform features with the existing table observation baseline;
- the comparison remains baseline vs candidate vs capacity-matched placebo;
- the audit was rerun at 1h and 3h on the same 171 waveform-linked stays.

| Horizon | Target | Random patient splits selected | Held-out beats baseline | Care-unit signal | Time signal | Validated |
|---|---|---:|---:|---:|---:|---:|
| 1h | heart rate | 2/7 | 1/7 | no | no | no |
| 1h | MAP | 0/7 | 0/7 | no | no | no |
| 1h | SpO2 | 0/7 | 0/7 | no | no | no |
| 1h | respiratory rate | 3/7 | 0/7 | yes | no | no |
| 3h | heart rate | 0/7 | 0/7 | no | yes | no |
| 3h | MAP | 0/7 | 0/7 | no | no | no |
| 3h | SpO2 | 0/7 | 0/7 | no | 0/7 | no |
| 3h | respiratory rate | 0/7 | 0/7 | no | no | no |

The first beat-level run exposed an extractor error rather than a model result:
the generic peak detector counted QRS plus same-beat T-wave/artefact peaks as
separate ECG beats. On the 1h linked cohort, the pre-fix ECG rate had median
about 131 bpm, MAE about 42 bpm against the same-anchor table HR, and correlation
about -0.20. The fix uses an 8--20 Hz QRS-focused band and a 0.45 s refractory
period. After the fix, ECG median was about 85 bpm, MAE about 5--6 bpm, and
lead-level correlation was about 0.80. The old pre-fix audit must not be used as
evidence against the waveform design.

After the QRS fix, the encoder reads successfully (4,151/4,151 feature rows at
1h and 1,396/1,396 at 3h). Respiratory rate at 1h remains the strongest
near-miss: its care-unit heldout MAE improved from 3.778 to 3.557, but it did
not pass all patient-heldout and time-heldout requirements. No target passed
the complete promotion gate yet.

The candidate therefore remains research-only. The next meaningful design
change would be a learned multi-scale waveform/table fusion model or a
waveform-specific nowcast task; simply adding more summary columns would not
be justified by this audit.

## Same-Anchor Waveform Nowcast

The waveform-specific nowcast task was then evaluated separately from future
forecasting. The candidate used only pre-anchor beat-level features and was
compared with a robustly scaled table-only Ridge model and an equal-capacity
random-feature placebo. Target values were restricted to physiologically
plausible ranges before fitting to prevent MIMIC table outliers from dominating
the audit.

| Target | Patient-heldout | Care-unit heldout | Time heldout | Validated |
|---|---:|---:|---:|---:|
| heart rate | 7/7 | pass | pass | yes |
| respiratory rate | 7/7 | pass | pass | yes |
| MAP | 4/7 | fail | pass | no |
| SpO2 | 0/7 | fail | fail | no |

For the external heldouts, HR MAE was 5.90 versus 16.94 for the table-only
baseline in the care-unit split, and 5.26 versus 14.78 in the chronological
split. Respiratory-rate MAE was 3.51 versus 4.46 and 3.79 versus 4.40 in the
same two splits. These are factual same-anchor observation results, not future
forecasting or causal treatment results.

This is the first waveform capability allowed to enter the validated
observation layer: HR and respiratory rate nowcasting only. Future 1h/3h
waveform forecasting remains candidate-only.

## Nowcast-to-Forecast Bridge

The validated same-anchor nowcasts were then fed into a second, independent
forecast model as `wave_nowcast_<target>` state-completion features. The
nowcast model was refit inside every patient/care-unit/time discovery split;
the future forecast model never received future waveform or future table
columns.

| Horizon | Target | Patient-heldout candidate beats baseline | Patient-heldout candidate beats placebo | Care-unit | Time | Promoted |
|---|---|---:|---:|---:|---:|---:|
| 1h | heart rate | 3/7 | 3/7 | pass | pass | no |
| 1h | respiratory rate | 6/7 | 6/7 | pass | pass | no |
| 3h | heart rate | 0/7 | 0/7 | pass | pass | no |
| 3h | respiratory rate | 4/7 | 5/7 | fail | fail | no |

This supports a bounded conclusion: validated waveform nowcasts carry useful
short-horizon information into RR forecasting, but the effect is not yet
stable enough for forecast promotion and decays by 3h. The validated layer is
therefore limited to same-anchor HR/RR completion; the bridge remains
candidate-only.

## Quality-Gated Forecast Fallback

The near-miss was also tested with a fixed pre-anchor quality gate. When ECG or
respiration quality failed the gate, the bridge automatically fell back to the
table forecast; otherwise it used the waveform-nowcast bridge. Coverage was
reported rather than hidden.

At 1h, respiratory rate improved to 7/7 against the table baseline, but only
6/7 against placebo. Heart rate reached 4/7. At 3h, respiratory rate reached
4/7 against both comparators and heart rate only 1/7. Therefore quality-gated
abstention reduces error but does not unlock forecast promotion. This closes
the current waveform design loop without granting future forecast authority.
