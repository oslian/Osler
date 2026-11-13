# Native MAP Refinement

## Seven-Split Completion

All 14 matched refinement fits completed. All direct report dependencies were
hash-verified by `scripts/summarize_map_native_refinement.py`.

| Median patient-equal MAE | New source patients | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.44082 | 11.73923 |
| Original own | 8.84140 | 10.21067 |
| Refined own | 8.71165 | 10.15199 |
| Original full | 9.22298 | 10.57510 |
| Refined full | 9.18431 | 10.55747 |

Full versus original full significantly improves 4/7 source splits and 0/7
hospital splits, with no significant harms. Own refinement improves 5/7 and
2/7 respectively. Refined full versus refined own still has zero gains, with
7/7 source harms and 6/7 hospital harms. Full beats persistence 7/7 both roles.

Thus this additional optimization protocol offers partial source improvement,
not a stable repair of the full-history increment or hospital accuracy. It
does not prove convergence or exclude other optimization mechanisms: AdamW
was reset, and the protocol is not an uninterrupted 64-epoch experiment.
Earlier evidence already located harm in the selected model's fitting
population and value objective. Timing, MAE scoring alone, and this additional
training budget are insufficient explanations/repairs. Representation,
non-target event cadence and optimization remain incompletely separated.

The seven splits share patients and the hospital evaluation is reused, not
seven independent replications or a new lockbox. No calibration, runtime,
weights in the release, or promotion changed. Evidence:
`artifacts/map_native_query_refine_summary_20260912.json`.

## Historical First Split

The pending status below is superseded by the completed seven-split result.

Seed7 paired refinement completed under the prewritten protocol: both arms
start from their source-tune-selected checkpoint, reset AdamW, then run 32
additional epochs with matched fresh source-patient draws. This is not an
uninterrupted 64-epoch run; original optimizer state was unavailable.

| MAE | New source patients | Held-out hospitals |
|---|---:|---:|
| Original own | 8.80806 | 10.21067 |
| Refined own | 8.68509 | 10.15199 |
| Original full | 9.16712 | 10.56384 |
| Refined full | 9.07457 | 10.55196 |

Refined full-original full: source -0.09255 [-0.15374,-0.03217], hospital
-0.01187 [-0.11824,+0.09220]. Own also improves source significantly but not
hospital. Refined full-own remains harmful: source +0.38948 [0.23359,0.54151],
hospital +0.39997 [0.07150,0.71802]. Both estimates are patient-equal; paired
bootstrap follows the existing policy. Full remains better than persistence.

Additional refinement helps source accuracy in this split but does not repair
the full-history increment. One of seven splits only; no general convergence,
compute-insufficiency or optimizer-reset causal verdict. Reused evaluation is
not new confirmation. Other seeds remain pending. No runtime/calibration or
promotion change. Evidence: `artifacts/map_native_query_refine_7_20260912.json`.
