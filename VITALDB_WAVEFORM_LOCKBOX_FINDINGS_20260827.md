# VitalDB independent waveform lockbox findings

## Question

Can the frozen MIMIC-IV raw-waveform JEPA identify incident sustained hypoxemia
in a larger independent hospital waveform cohort?

This is a factual event-risk audit. It does not estimate treatment effects and
does not grant clinical decision authority.

## Data boundary

- Development checkpoint: MIMIC-IV Waveform Database, 161 patients.
- Independent lockbox: VitalDB open dataset, Seoul National University Hospital.
- Lockbox domain: intraoperative surgical care, not ICU care.
- Frozen event: new sustained SpO2 <= 92 within 15 minutes after a stable
  10-minute pre-anchor window.
- Anchor sampling is deterministic and does not use the future event label.
- No VitalDB result was used to tune the checkpoint, feature schema, horizon,
  threshold, temperature, or pass gate.
- Row-level lockbox data and raw tensors remain outside the repository.

Official data interfaces:

- <https://vitaldb.net/docs/?documentId=API%2FPython_Library.md>
- <https://api.vitaldb.net/docs/>

## Support audit

The numeric-only preregistration audit scanned 500 cases before raw-waveform
download:

| Item | Result |
|---|---:|
| Supported cases | 499 |
| Eligible anchors | 3,889 |
| Events | 334 |
| Event prevalence | 8.59% |

Artifact: `artifacts/vitaldb_event_support_500.json`.

## Final lockbox

| Item | Result |
|---|---:|
| Patients | 318 |
| Anchors | 2,505 |
| Events | 218 |
| Event prevalence | 8.70% |
| Raw sampling rate | 50 Hz |
| API read failures after retry | 0 |

Artifact: `artifacts/vitaldb_waveform_lockbox_320.json`.

The portable signal contract used ECG II, PLETH, capnography as the respiratory
waveform, and arterial pressure when available. Numeric history used HR, RR,
SpO2, MAP, SBP, DBP, FiO2, and PEEP. All inputs were available at or before the
anchor.

## Frozen external result

| Input | Brier | AUC | ECE |
|---|---:|---:|---:|
| Full frozen raw-waveform JEPA | 0.07953 | 0.6200 | 0.0330 |
| Raw waveform masked | 0.08881 | 0.5934 | 0.0999 |
| Raw waveform shuffled between patients | 0.07991 | 0.5767 | 0.0320 |
| External-cohort climatology | 0.07945 | 0.5000 | 0.0000 |

Patient-level paired bootstrap:

| Comparison | Delta Brier | 95% CI | Pass |
|---|---:|---:|---:|
| Full vs raw masked | -0.00909 | [-0.01100, -0.00710] | yes |
| Full vs shuffled raw | -0.00030 | [-0.00129, 0.00067] | no |

Artifact: `artifacts/vitaldb_raw_waveform_15m_external_validation.json`.

## Interpretation

The raw waveform is not empty: masking it materially worsens the model, and the
unshuffled waveform improves ranking over shuffled patient waveforms. However,
the patient-specific raw content does not significantly improve paired Brier
error over the equal-distribution shuffled placebo, and its Brier score is
slightly worse than the external-cohort climatology. The frozen model therefore
has a weak portable ranking signal but not a validated calibrated event-risk
gain.

The external cohort removed the earlier small-sample objection: 318 patients and
218 events exceed the prespecified support floor. The remaining failure is not
explained by lack of independent cases. It is more consistent with a model that
uses waveform availability or broad morphology but does not yet extract stable,
patient-specific precursors of the future event across ICU and operating-room
domains.

The checkpoint remains `candidate_only` because:

1. its original internal MIMIC gate failed;
2. the VitalDB full-vs-shuffled paired Brier gate failed;
3. VitalDB is a surgical setting and cannot alone establish ICU runtime validity.

## Reproduction

```bash
python -m scripts.audit_vitaldb_event_support \
  --cases-index /tmp/vitaldb_cases.json \
  --tracks-index /tmp/vitaldb_tracks.json \
  --report artifacts/vitaldb_event_support_500.json \
  --max-cases 500

python -m scripts.build_vitaldb_waveform_lockbox \
  --cases-index /tmp/vitaldb_cases.json \
  --tracks-index /tmp/vitaldb_tracks.json \
  --cohort-output /tmp/vitaldb_waveform_lockbox.parquet \
  --raw-output /tmp/vitaldb_waveform_lockbox.npz \
  --report artifacts/vitaldb_waveform_lockbox_320.json \
  --max-cases 320 \
  --raw-interval-seconds 0.02

python -m scripts.validate_vitaldb_waveform_lockbox \
  --checkpoint models/raw_waveform_event_jepa_15m_candidate.pt \
  --cohort /tmp/vitaldb_waveform_lockbox.parquet \
  --raw-cache /tmp/vitaldb_waveform_lockbox.npz \
  --report artifacts/vitaldb_raw_waveform_15m_external_validation.json
```

## Stop rule

Do not tune on these 318 lockbox patients and report them again as external
validation. A further model iteration requires a separate development partition
and a third untouched lockbox.
