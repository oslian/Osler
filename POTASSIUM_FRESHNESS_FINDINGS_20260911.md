# Potassium Freshness Localization

Fixed14weights and identical predictions. All pooled reports exactly reproduce
the previous external evaluation. Bins use only pre-anchor known measurement
age, not future changes. This is a descriptive diagnosis, not routing selection.

| Role / age | Patients / rows | Whole beats own | Whole beats persistence |
|---|---|---:|---:|
| eICU <=1h | 145 / 351 | 0/7 | 7/7 |
| eICU (1,3]h | 115 / 272 | 0/7 | 7/7 |
| eICU (3,6]h | 98 / 196 | 0/7 | 7/7 |
| MIMIC <=1h | 678 / 3670 | 0/7 | 7/7 |
| MIMIC (1,3]h | 623 / 1698 | 0/7 | 7/7 |
| MIMIC (3,6]h | 157 / 306 | 0/7 | 0/7 |

Whole versus own MIMIC harms:1/7 fresh,2/7 middle,0/7old. All groups supported.
Patients may appear in multiple bins; counts do not sum to unique cohort size.

Median target age: eICU1.333h,MIMIC.817h. Median draw-to-draw duration:
7.533h/6.567h, distinct from the6h query horizon. This is expected under the
documented last-known-value contract, not proof of future leakage. MIMIC CVICU
age median.1h versus MICU.9h confirms observation differences. Merely filtering
to fresh observations does not restore whole-versus-own benefit. Subgroup
composition and treatment differ; age is not established as their causal driver.

The stale MIMIC subgroup flags a deployment uncertainty concern, not a license
to optimize an abstention cutoff on opened validation data. No runtime, weight,
interval or promotion change. No sole root cause or completed repair claimed.

Artifact: `artifacts/potassium_freshness_20260911.json`.
Protocol: `POTASSIUM_FRESHNESS_PROTOCOL_20260911.md`.
Command: `python -m scripts.evaluate_potassium_matched_encoder --freshness`.
