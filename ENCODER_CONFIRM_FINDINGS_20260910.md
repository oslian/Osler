# Seven-Seed Supported Decoder Confirmation

Completed28 fits: own/whole x bounded/unbounded x seven source seeds. Source
support protection in every arm, corrected BNP cache, unchanged source-only
fit/tune selection. Reports: `artifacts/encoder_confirm_20260910.json` and
`artifacts/encoder_confirm_strata_20260910.json`, both complete.

Median patient-equal glucose6h MAE (mg/dL):

| Arm | eICU hospital | MIMIC validation |
|---|---:|---:|
| Persistence | 45.878 | 37.449 |
| Own bounded | 42.112 | 34.536 |
| Own unbounded | 38.091 | 31.812 |
| Whole bounded | 42.400 | 35.423 |
| Whole unbounded | 38.331 | 33.087 |

Both identity arms improve bounded7/7 and persistence7/7 in pooled patient
bootstrap CIs in both domains. Whole vs own incremental gain remains0/7 for
both decoder modes in both domains. Cap removal has reproducible benefit,
but this is not proof of incremental whole-body prediction.

## Remaining Heterogeneity

Supported eICU unit/year strata show7/7 identity-vs-bounded improvement for
both views. Six small care units remain unsupported (<30 distinct patients).
MIMIC own-unbounded vs persistence passes7/7 in CVICU,CCU,MICU,MICU/SICU,SICU,
but only1/7 in TSICU (83 patients). Whole vs persistence passes7/7 CVICU,MICU,
SICU;6/7 MICU/SICU;1/7 CCU;0/7 TSICU. No supported MIMIC care unit shows
significant whole-vs-own benefit in any seed.

All five MIMIC periods show own-unbounded vs persistence7/7, but cap removal
vs own-bounded is significant0/7 in2008-2010. Whole vs persistence is1/7 in
2014-2016. These are coarse anchor-year-group subgroup audits, not pure
forward-time holdouts. Do not select models from these subgroup outcomes.

The output limit is a confirmed contributor, but cannot explain all residual
failure: even own does not stably beat persistence in TSICU. Case mix, unknown
future inputs, representation transfer and limited optimization remain
hypotheses, not uniquely diagnosed causes.

## Boundaries And Reproduction

The separate subgroup protocol was added after training began, before reading
subgroup results. Original training protocol hash still matches. Every
training/query cache is hash-bound; local prediction NPZs follow scored-frame
order and subgroup evaluation checks the frame hash. Candidate checkpoints
include activation and input-support metadata. No raw rows committed.

Seven seeds reuse validation patients; CIs are exploratory, unadjusted for
multiple comparisons. MIMIC is repeatedly inspected validation, not untouched.
No complete care-unit/time gate, conformal calibration, other-target retention,
all-current-router comparison or new lockbox. No runtime changes or promotion.
296 tests pass via unittest; pytest is not installed.
