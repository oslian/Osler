# Continued Expanded Selection Results

Completed source-only replay of epochs9-19, retaining frozen0-8scores. All3323
broad source selection patients, original90/additional3233 scopes,33trained
targets and original selected-group scores unchanged. Original90 per-target
scores reproduce every new epoch within1e-7; complete unique pairs and bindings
checked. Six untrained targets remain pending.

The original90, additional3233, all3323, and equal-group macro rules ALL select
epoch16 across the full0-19trajectory. Broadmacro epoch7=1.557181 versus
epoch16=1.547097; equal-group1.686014->1.676308. Expanded selection does not
offer a different common checkpoint that repairs the observed evaluation results.
At16,32/33full-broad targets numerically beat epoch0 population initialization;
troponin remains just above it. This is not32promotions or process-control proof.

## Creatinine And Lipase: Mean Signs Are Not Established Harms

Epoch7->16paired patient bootstrap,4000draws,95%descriptive intervals:

| Target/scope | Patients | Mean MAE change | 95% interval |
|---|---:|---:|---:|
| Creatinine original | 83 | -.003413 | [-.048545,.042711] |
| Creatinine all | 3095 | +.001295 | [-.004921,.007381] |
| Lipase original | 7 | -.106748 | [-.573991,.250617] |
| Lipase all | 455 | +.025994 | [-.013388,.064864] |
| Glucose all | 3127 | -.016569 | [-.023055,-.009943] |
| BUN all | 3086 | -.010057 | [-.016679,-.003238] |

Changes are normalized MAE, negative better. One observation per patient-target;
pair identities/truths checked and means reproduce epoch summaries1e-10. Intervals
are post-selection and not adjusted for33targets/epoch search: no promotion or
formal subgroup inference. Small7patient lipase bootstrap is especially weak.

The expanded source means forcreatinine/lipase now agree in direction with the
evaluation regression, but all these source intervals crosszero. We cannot claim
the larger sample proves damage or that smaller selection definitively hid a
true harm. More defensible: the original tiny improvements were unsupported,
and cohort selection changes their descriptive signs. Conversely glucose/BUN
show more consistent source gains in this replay, still not full validation.

## Decision

Do not retune the selector or construct an oracle per-target epoch mixture.
The same commonepoch16 remains selected; increasing source selection support
does not rescue the previous full gate. Stop treating tiny source differences
as evidence of specific mechanisms. Any retention repair must distinguish
meaningful source changes from sampling uncertainty and preserve the shared
state and patient/site validation. Full39-target training remains unfinished.

Evidence:
- artifacts/native_reconstruction_continued_selection_20260920.json
- artifacts/native_reconstruction_continued_selection_pairs_20260920.json

No new optimizer steps, runtime, registry, interval or promotion changes.
