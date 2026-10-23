# MIMIC-IV Full Target Observation Coverage

Status: validated factual observation coverage for the full MIMIC-IV v3.1
6-hour transition cohort. This expands the whole-body observation contract; it
does not grant causal, counterfactual, clinical, runtime, or treatment
authority.

## Scope

The audit used the local full-cohort transition artifact:

```text
/private/tmp/mimiciv_obs_6h.parquet
```

The cohort contains 640,164 transition rows, 63,307 subjects, 90,190 ICU
stays, and 17 care units. Fifty-five numeric targets had both current and
future support and were evaluated. The audit used:

- seven patient-heldout seeds;
- care-unit heldout evaluation;
- chronological heldout evaluation;
- capacity-matched Ridge baselines and bootstrap comparisons;
- split-conformal 90% interval checks for forecast targets.

The raw report is local-only:

```text
/private/tmp/mimiciv_cross_database_coverage_6h_all_targets.json
```

## Promotion Results

| Capability | Validated count |
|---|---:|
| Numeric targets evaluated for nowcast | 55 |
| Numeric targets evaluated for 6h forecast | 55 |
| Validated same-time nowcast targets | 19 |
| Validated 6h factual forecast targets | 18 |
| Validated 6h conformal interval targets | 14 |

### Newly added nowcast directions

These targets were not in the previous whole-body nowcast cell map and now
pass the full MIMIC heldout contract:

- C-reactive protein (`crp`)
- fibrinogen (`fibrinogen`)
- ionized calcium (`ionized_calcium`)
- magnesium (`magnesium`)
- serum osmolality (`serum_osmolality`)

The nowcast is a current-state completion only. It does not imply that the
target can be forecast at 6 hours.

### Newly added 6h forecast directions

The following global MIMIC target cells were added to the whole-body future
trajectory map:

- anion gap (`anion_gap`)
- calcium (`calcium`)
- chloride (`chloride`)
- magnesium (`magnesium`)
- minute volume (`minute_volume`)
- arterial carbon dioxide (`paco2`)
- arterial oxygen (`pao2`)
- phosphate (`phosphate`)
- urine output (`urine_output`)

The complete validated 6h forecast set is:

```text
anion_gap, bicarbonate, calcium, chloride, heart_rate, hemoglobin,
magnesium, map, minute_volume, paco2, pao2, ph, phosphate, potassium,
respiratory_rate, temperature, urine_output, wbc
```

The 14 cells with calibrated 90% intervals are:

```text
anion_gap, bicarbonate, calcium, heart_rate, magnesium, map, minute_volume,
paco2, pao2, ph, phosphate, potassium, respiratory_rate, temperature
```

Chloride, hemoglobin, urine output, and WBC have validated point forecasts in
this audit but do not receive a numeric interval from this run.

## Waveform Sweep Boundary

The waveform-linked cohort was also swept across all 57 available transition
targets at 1h and 3h using 96 waveform features. It had 171 aligned stays / 170
subjects, no waveform read failures, and zero waveform increments passed the
full promotion gate. Therefore the new directions above come from the full
table-based MIMIC observation layer, not from waveform features.

Waveform remains candidate-only. A negative waveform increment does not remove
the table-based target capability.

## Safety Boundary

- same-time nowcast is allowed only for the validated current-state cells;
- future movement is allowed only for validated target-horizon cells;
- persistence or missing remains the fallback elsewhere;
- interval bounds are shown only for the 14 calibrated cells;
- no row-level values or patient identifiers belong in committed artifacts;
- causal and counterfactual claims remain disabled;
- no checkpoint, active rule, or clinical decision authority is granted.
