# MIMIC-IV Hormone Data Integration Findings

Status: hormone observations are now connected to a local MIMIC-IV transition
artifact, but no hormone target is promoted for factual prediction or nowcast.

## What Was Added

`mimiciv_observation_transition_extract.py` now recognizes these MIMIC-IV lab
itemids:

- cortisol;
- PTH;
- TPO antibody;
- TSH;
- total T4;
- free T4.

The targeted augmentation script is:

```text
mimiciv_hormone_transition_augment.py
```

It preserves the existing 640,164-row MIMIC observation cohort and adds the
hormone columns using the existing `t` / `t_plus` alignment contract. The
result is local-only:

```text
/private/tmp/mimiciv_obs_6h_hormone_augmented.parquet
/private/tmp/mimiciv_obs_6h_hormone_augmented_report.json
```

## Paired Transition Support

The augmented cohort has the following current/future support at the 6-hour
anchor contract:

| Target | Current pairs | Future pairs | Current + future pairs |
|---|---:|---:|---:|
| Cortisol | 1,205 | 1,888 | 30 |
| TSH | 2,333 | 2,812 | 9 |
| Free T4 | 504 | 763 | 4 |
| Total T4 | 346 | 549 | 3 |
| PTH | 184 | 278 | 3 |
| TPO antibody | 7 | 17 | 0 |

This shows the central limitation: hormone labs exist, but they are usually
single measurements rather than repeated short-horizon ICU observations.

## Gate Results

The hormone target audit used seven patient-heldout seeds, care-unit heldout,
chronological heldout and capacity-matched placebo comparisons.

- no hormone nowcast target passed the full gate;
- no hormone 6-hour forecast target had enough paired support to be evaluated
  as a stable forecast;
- TPO antibody was too sparse even for the minimum nowcast support;
- cortisol had enough nowcast rows to run the gate, but did not beat the
  baseline consistently;
- TSH and free T4 had usable current observations, but no robust improvement
  over baseline/placebo.

The aggregate report is:

```text
/private/tmp/mimiciv_hormone_target_audit.json
```

## Interpretation

The correct conclusion is not that hormone physiology is disconnected. It is
that the current MIMIC ICU anchor process observes hormone assays too
infrequently to learn a reliable 6-hour dynamic forecast or a validated
same-time population relation.

The data can still support:

- descriptive hormone-axis stratification;
- candidate latent-state research;
- longer-horizon or encounter-level analysis;
- targeted cohorts where repeated endocrine assays are deliberately selected.

It cannot currently support a promoted hormone forecast cell in
`whole_body_state_forecast`.

## Safety Boundary

- hormone concentration is an observed lab only when a timestamped assay is
  present;
- missing hormone values remain missing;
- inferred ADH/insulin/PTH states are not hormone measurements;
- no causal, counterfactual, clinical, runtime, checkpoint or active-rule
  authority is granted.
